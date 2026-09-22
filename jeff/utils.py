import json
import math
from bisect import bisect_left

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {value: key for key, value in QTYPES.items()}
TEMP_MIN, TEMP_MAX = 0.5, 5.0
NOUL_DEFAULTS = {
    "false": "no, the statement does not hold",
    "true": "yes, the statement holds",
}


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
            key if value in (None, "") else f"{key}: {render(value)}"
            for key, value in criteria.items()
        ]
    if question_type == "score":
        return [
            f"level {index}: {render(value)}" for index, value in enumerate(criteria)
        ]
    criteria = criteria or {}
    return [
        f"{key}: {render(criteria[key]) if criteria.get(key) not in (None, '') else default}"
        for key, default in NOUL_DEFAULTS.items()
    ]


def build_sequence(tokenizer, state, question, max_length, head_max_length):
    mask = tokenizer.mask_token

    def encode(text, limit=None):
        ids = tokenizer(
            str(text).replace(mask, " "), add_special_tokens=False
        ).input_ids
        return ids if limit is None else ids[:limit]

    options = render_options(question)
    head = encode(f"{question['t']} question: {question['ins']}")
    option_ids = [
        [tokenizer.mask_token_id] + encode(" " + option, 48) for option in options
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
    input_ids.extend(encode(render(state), max(0, max_length - len(input_ids) - 1)))
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
            hidden = self.head(hidden, src_key_padding_mask=~attention_mask.bool())
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
        top_size = min(2, probabilities.size(-1))
        top = probabilities.topk(top_size, -1).values
        if top_size < 2:
            top = torch.cat([top, top.new_zeros(top.size(0), 2 - top_size)], -1)
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
    size = ("2", "3-5", "6-10", "11+")[bisect_left((2, 5, 10), count)]
    return f"{QTYPE_NAMES[question_type]}:{size}"


def clamp_temperature(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, value)) if math.isfinite(value) else 1.0


def collate(items, pad_token_id):
    sequence_length = max(len(item["ids"]) for item in items)
    marker_count = max(len(item["markers"]) for item in items)
    batch = {
        "input_ids": torch.full(
            (len(items), sequence_length), pad_token_id, dtype=torch.long
        ),
        "attention_mask": torch.zeros((len(items), sequence_length), dtype=torch.long),
        "marker_pos": torch.zeros((len(items), marker_count), dtype=torch.long),
        "marker_mask": torch.zeros((len(items), marker_count), dtype=torch.bool),
        "qtype": torch.tensor([item["qtype"] for item in items]),
    }
    for index, item in enumerate(items):
        length, count = len(item["ids"]), len(item["markers"])
        batch["input_ids"][index, :length] = torch.as_tensor(item["ids"])
        batch["attention_mask"][index, :length] = 1
        batch["marker_pos"][index, :count] = torch.as_tensor(item["markers"])
        batch["marker_mask"][index, :count] = True
    return batch
