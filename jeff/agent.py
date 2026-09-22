import json
import warnings
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from .utils import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    build_model,
    build_sequence,
    clamp_temperature,
    collate,
    confidence,
    render_options,
    temperature_bucket,
)

DEFAULT_MODEL_DIRECTORY = Path(__file__).resolve().parent.parent / "model"
REQUIRED_MODEL_FILES = (
    "rl_agent_config.json",
    "model.safetensors",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def resolve_model_directory(model_directory=None):
    directory = (
        (DEFAULT_MODEL_DIRECTORY if model_directory is None else Path(model_directory))
        .expanduser()
        .resolve()
    )
    missing = [
        name for name in REQUIRED_MODEL_FILES if not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Invalid local model directory {str(directory)!r}; missing: {', '.join(missing)}"
        )
    return directory


def _mps_available():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(device):
    available = {"cuda": torch.cuda.is_available, "mps": _mps_available}
    if device is None:
        device = next((name for name, ok in available.items() if ok()), "cpu")
    resolved = torch.device(device)
    is_ok = available.get(resolved.type)
    if not is_ok or is_ok():
        return resolved
    return torch.device("cpu")


class Agent:
    def _device_info(self):
        kind = self.device.type
        cuda = kind == "cuda"
        if kind in ("cpu", "mps"):
            dtype = torch.float32
        elif cuda and torch.cuda.get_device_capability(self.device)[0] < 8:
            dtype = torch.float16
        else:
            dtype = (
                torch.bfloat16
                if self.config.get("amp_dtype") == "bf16"
                else torch.float16
            )
        return dtype, kind, kind == "cpu", kind == "cuda"

    def __init__(self, model_directory=None, device=None):
        self.model_directory = resolve_model_directory(model_directory)
        with (self.model_directory / "rl_agent_config.json").open(
            encoding="utf-8"
        ) as file:
            self.config = json.load(file)
        required = ("encoder", "head_layers", "max_len", "head_max_len")
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(
                f"Invalid model config in {str(self.model_directory)!r}; missing: {', '.join(missing)}"
            )
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_directory / "tokenizer"), local_files_only=True
        )
        self.model = build_model(self.config, self.model_directory / "encoder")
        weights = load_file(str(self.model_directory / "model.safetensors"))
        self.model.load_state_dict(weights, strict=True)
        if hasattr(self.model.encoder.config, "reference_compile"):
            self.model.encoder.config.reference_compile = False
        raw_temperatures = self.config.get("temperature", [1.0, 1.0, 1.0])
        raw_buckets = self.config.get("temperature_by_options", {})
        self.temperatures = [clamp_temperature(value) for value in raw_temperatures]
        self.temperature_buckets = {
            key: clamp_temperature(value) for key, value in raw_buckets.items()
        }
        rejected = [
            f"{key}={value}"
            for key, value in raw_buckets.items()
            if clamp_temperature(value) != float(value)
        ]
        if rejected:
            warnings.warn(
                f"Model temperatures outside [{TEMP_MIN}, {TEMP_MAX}] were clamped: {', '.join(rejected)}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.dtype, _, cpu, _ = self._device_info()
        try:
            self.model.to(self.device).eval()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            if cpu:
                raise
            self._fallback_to_cpu(f"Could not place model on {self.device}: {error}")

    def _fallback_to_cpu(self, reason):
        warnings.warn(f"{reason}; using CPU", RuntimeWarning, stacklevel=3)
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.model.to(self.device).eval()

    @staticmethod
    def normalize_question(question):
        question_type = question.get("type")
        if question_type not in QTYPES:
            raise ValueError(f"Unsupported question type: {question_type!r}")
        if "instructions" not in question:
            raise ValueError("Question requires 'instructions'")
        criteria = question.get("criteria")
        if question_type == "choice":
            if isinstance(criteria, list):
                criteria = {value: None for value in criteria}
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice question requires non-empty 'criteria'")
        elif question_type == "score" and (
            not isinstance(criteria, list) or not criteria
        ):
            raise ValueError("Score question requires non-empty 'criteria'")
        instructions = question["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions, ensure_ascii=False)
        return {"t": question_type, "ins": instructions, "crit": criteria}

    @torch.no_grad()
    def predict(self, state, questions):
        if not questions:
            raise ValueError("At least one question is required")
        question_ids = list(questions)
        items = []
        for question_id in question_ids:
            question = self.normalize_question(questions[question_id])
            sequence, markers = build_sequence(
                self.tokenizer,
                state,
                question,
                self.config["max_len"],
                self.config["head_max_len"],
            )
            if len(markers) != len(render_options(question)):
                raise ValueError(
                    f"Question {question_id!r} options exceed head_max_len={self.config['head_max_len']}"
                )
            items.append(
                {
                    "ids": sequence,
                    "markers": markers,
                    "qtype": QTYPES[question["t"]],
                    "question": question,
                }
            )
        batch = collate(items, self.tokenizer.pad_token_id)
        try:
            logits, actions = self._forward(batch)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            message = str(error).lower()
            _, kind, cpu, _ = self._device_info()
            if cpu or not any(term in message for term in ("memory", kind)):
                raise
            self._fallback_to_cpu(f"Inference failed on {self.device}: {error}")
            logits, actions = self._forward(batch)
        answers = {}
        for question_id, item, values, action_values in zip(
            question_ids, items, logits, actions
        ):
            question = item["question"]
            count = len(item["markers"])
            question_type = item["qtype"]
            temperature = self.temperature_buckets.get(
                temperature_bucket(question_type, count),
                self.temperatures[question_type],
            )
            scaled = values[:count] / temperature
            probabilities = np.exp(scaled - scaled.max())
            probabilities /= probabilities.sum()
            answers[question_id] = self._answer(question, probabilities, action_values)
        return {
            "model": "jeff-rl-agent",
            "answers": answers,
            "usage": {
                "input_tokens": int(batch["attention_mask"].sum()),
                "output_tokens": 0,
            },
        }

    @staticmethod
    def _answer(question, probabilities, action_values):
        question_type = question["t"]
        answer = {
            "type": question_type,
            "action": {"act_probability": round(float(action_values[0]), 4)},
        }
        if question_type == "choice":
            labels = list(question["crit"])
            answer.update(
                choice=labels[int(probabilities.argmax())],
                probabilities={
                    label: round(float(value), 4)
                    for label, value in zip(labels, probabilities)
                },
                confidence=round(confidence(probabilities), 4),
            )
        elif question_type == "score":
            answer.update(
                score=round(
                    float((np.arange(len(probabilities)) * probabilities).sum()), 4
                ),
                legend={
                    str(index): criterion
                    for index, criterion in enumerate(question["crit"])
                },
                probabilities={
                    str(index): round(float(value), 4)
                    for index, value in enumerate(probabilities)
                },
                confidence=round(confidence(probabilities), 4),
            )
        else:
            value = float(probabilities[1])
            answer.update(
                noul=round(value, 4), confidence=round(max(value, 1.0 - value), 4)
            )
        return answer

    def _forward(self, batch):
        _, _, _, cuda = self._device_info()
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=cuda,
        ):
            logits, actions = self.model(
                *(value.to(self.device) for value in batch.values())
            )
        return logits.float().cpu().numpy(), torch.softmax(
            actions.float(), -1
        ).cpu().numpy()

    system_one = predict


def load(model_directory=None, device=None):
    return Agent(model_directory, device)
