import json
import math

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {value: key for key, value in QTYPES.items()}
TEMP_MIN, TEMP_MAX = 0.5, 5.0


def render(value):
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)
    )


def render_options(question):
    question_type, criteria = question["t"], question.get("crit")
    if question_type == "choice":
        return [
            key if value is None or value == "" else f"{key}: {render(value)}"
            for key, value in criteria.items()
        ]
    if question_type == "score":
        return [
            f"level {index}: {render(value)}" for index, value in enumerate(criteria)
        ]
    criteria = criteria or {}
    return [
        f"false: {render(criteria['false'])}"
        if criteria.get("false") not in (None, "")
        else "false: no, the statement does not hold",
        f"true: {render(criteria['true'])}"
        if criteria.get("true") not in (None, "")
        else "true: yes, the statement holds",
    ]


def build_sequence(tokenizer, state, question, max_length, head_max_length):
    mask_token = tokenizer.mask_token
    options = render_options(question)
    instructions = str(question["ins"]).replace(mask_token, " ")
    head = tokenizer(
        f"{question['t']} question: {instructions}", add_special_tokens=False
    )["input_ids"]
    option_ids = [
        [tokenizer.mask_token_id]
        + tokenizer(" " + option.replace(mask_token, " "), add_special_tokens=False)[
            "input_ids"
        ][:48]
        for option in options
    ]
    budget = head_max_length - sum(map(len, option_ids))
    if budget < 16:
        size = max(4, (head_max_length - 16) // max(1, len(option_ids)))
        option_ids = [ids[:size] for ids in option_ids]
        budget = head_max_length - sum(map(len, option_ids))
    input_ids = (
        [tokenizer.cls_token_id] + head[: max(8, budget)] + [tokenizer.sep_token_id]
    )
    markers = []
    for ids in option_ids:
        markers.append(len(input_ids))
        input_ids.extend(ids)
    input_ids.append(tokenizer.sep_token_id)
    available = max(0, max_length - len(input_ids) - 1)
    serialized = (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    )
    state_ids = tokenizer(
        serialized.replace(mask_token, " "), add_special_tokens=False
    )["input_ids"][:available]
    input_ids.extend(state_ids)
    input_ids.append(tokenizer.sep_token_id)
    return input_ids[:max_length], [marker for marker in markers if marker < max_length]


class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers, action_count):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(
            hidden_size,
            max(1, hidden_size // 64),
            4 * hidden_size,
            0.1,
            batch_first=True,
            norm_first=True,
        )
        self.head = (
            nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            if head_layers
            else None
        )
        self.type_emb = nn.Embedding(3, hidden_size)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.act_head = nn.Sequential(
            nn.Linear(hidden_size + 4, 256), nn.GELU(), nn.Linear(256, action_count)
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(
        self, input_ids, attention_mask, marker_pos, marker_mask, question_type
    ):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        hidden = hidden + self.type_emb(question_type)[:, None, :]
        if self.head is not None:
            padding_mask = ~attention_mask.bool()
            for layer in self.head.layers:
                hidden = layer(hidden, src_key_padding_mask=padding_mask)
        index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
        logits = (
            self.scorer(torch.gather(hidden, 1, index))
            .squeeze(-1)
            .float()
            .masked_fill(~marker_mask, -1e4)
        )
        probabilities = torch.softmax(logits.detach(), -1)
        count = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-9))).sum(
            -1
        ) / torch.log(count)
        if probabilities.size(-1) >= 2:
            top = probabilities.topk(2, -1).values
        else:
            top = probabilities.topk(1, -1).values
            top = torch.cat([top, torch.zeros_like(top)], -1)
        features = torch.stack(
            [top[:, 0], top[:, 0] - top[:, 1], entropy, count / 255.0], -1
        )
        return logits, self.act_head(torch.cat([hidden[:, 0].float(), features], -1))


def build_model(config, encoder_directory):
    encoder_config = AutoConfig.from_pretrained(
        str(encoder_directory), local_files_only=True
    )
    encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
    return DecisionModel(
        encoder, config["head_layers"], len(config.get("act_costs", {})) + 1
    )


def confidence(probabilities):
    count = len(probabilities)
    if count < 2:
        return 1.0
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - entropy / math.log(count), 0.0, 1.0))


def temperature_bucket(question_type, count):
    size = (
        "2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else "11+"
    )
    return f"{QTYPE_NAMES[question_type]}:{size}"


def clamp_temperature(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, value)) if math.isfinite(value) else 1.0


def collate(items, pad_token_id):
    batch_size = len(items)
    sequence_length = max(len(item["ids"]) for item in items)
    marker_count = max(len(item["markers"]) for item in items)
    input_ids = torch.full(
        (batch_size, sequence_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    marker_positions = torch.zeros((batch_size, marker_count), dtype=torch.long)
    marker_mask = torch.zeros((batch_size, marker_count), dtype=torch.bool)
    for index, item in enumerate(items):
        length, count = len(item["ids"]), len(item["markers"])
        input_ids[index, :length] = torch.tensor(item["ids"])
        attention_mask[index, :length] = 1
        marker_positions[index, :count] = torch.tensor(item["markers"])
        marker_mask[index, :count] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_positions,
        "marker_mask": marker_mask,
        "qtype": torch.tensor([item["qtype"] for item in items]),
    }
