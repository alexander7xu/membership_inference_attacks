from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Iterable


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: Iterable[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(normalized_prediction == normalize_answer(answer) for answer in answers)
    )


def token_f1(prediction: str, answers: Iterable[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for answer in answers:
        gold_tokens = normalize_answer(answer).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if not pred_tokens or not gold_tokens:
            score = float(pred_tokens == gold_tokens)
        elif num_same == 0:
            score = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def perplexity(loss: float) -> float:
    if math.isnan(loss):
        return float("nan")
    return float(math.exp(min(loss, 20.0)))
