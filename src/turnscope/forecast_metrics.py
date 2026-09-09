"""Conversation-weighted prefix losses and explicit any-alert decision metrics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from .forecast_data import ForecastExample


def weighted_auc(
    labels: Sequence[bool], scores: Sequence[float], weights: Sequence[float]
) -> float | None:
    """Weighted concordance with half-credit for tied scores, in O(n log n)."""
    positives = math.fsum(weight for label, weight in zip(labels, weights, strict=True) if label)
    negatives = math.fsum(
        weight for label, weight in zip(labels, weights, strict=True) if not label
    )
    if not positives or not negatives:
        return None
    buckets: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for label, score, weight in zip(labels, scores, weights, strict=True):
        buckets[score][int(label)] += weight
    before = numerator = 0.0
    for negative, positive in (buckets[score] for score in sorted(buckets)):
        numerator += positive * (before + negative / 2)
        before += negative
    return numerator / (positives * negatives)


def conversation_scores(
    examples: Sequence[ForecastExample], scores: Sequence[float]
) -> dict[str, tuple[bool, float]]:
    result: dict[str, tuple[bool, float]] = {}
    for example, score in zip(examples, scores, strict=True):
        identifier = example.prefix.conversation_id
        if identifier in result:
            label, previous = result[identifier]
            if label != example.label:
                raise ValueError("inconsistent labels within a forecast conversation")
            score = max(previous, score)
        result[identifier] = (example.label, score)
    return result


def decision_counts(
    labels: Sequence[bool], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores, strict=True):
        alert = score >= threshold
        tp += label and alert
        fp += not label and alert
        tn += not label and not alert
        fn += label and not alert
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(labels) if labels else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": recall,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "balanced_accuracy": (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def choose_threshold(
    examples: Sequence[ForecastExample], scores: Sequence[float]
) -> tuple[float, float]:
    """Maximize validation conversation balanced accuracy; ties choose higher threshold."""
    rows = tuple(conversation_scores(examples, scores).values())
    labels, values = [row[0] for row in rows], [row[1] for row in rows]
    if set(labels) != {False, True}:
        raise ValueError("validation requires eligible conversations from both classes")
    buckets: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for label, value in rows:
        buckets[value][int(label)] += 1
    positives = sum(labels)
    negatives = len(labels) - positives
    remaining_positive, removed_negative = positives, 0
    best = (positives * negatives, 0.0)
    # Ascending sweep: >= threshold retains the current bucket, then removes it.
    # Each conversation contributes one maximum, not a separately split prefix.
    for threshold in sorted({0.0, 1.0, *values}):
        # Use the integer common-denominator numerator so rounding cannot
        # override the specified higher-threshold tie break.
        score = remaining_positive * negatives + removed_negative * positives
        best = max(best, (score, threshold))
        negative, positive = buckets.get(threshold, [0, 0])
        remaining_positive -= positive
        removed_negative += negative
    return best[1], best[0] / (2 * positives * negatives)


def forecast_metrics(
    examples: Sequence[ForecastExample], scores: Sequence[float], threshold: float
) -> dict[str, Any]:
    if not examples:
        raise ValueError("at least one eligible forecast prefix is required")
    frequencies = Counter(example.prefix.conversation_id for example in examples)
    weights = [1 / frequencies[example.prefix.conversation_id] for example in examples]
    labels = [example.label for example in examples]
    size = len(frequencies)
    brier = (
        math.fsum(
            weight * (score - int(label)) ** 2
            for label, score, weight in zip(labels, scores, weights, strict=True)
        )
        / size
    )
    clipped = [min(1 - 1e-15, max(1e-15, score)) for score in scores]
    log_loss = (
        -math.fsum(
            weight * (math.log(score) if label else math.log1p(-score))
            for label, score, weight in zip(labels, clipped, weights, strict=True)
        )
        / size
    )
    rows = tuple(conversation_scores(examples, scores).values())
    decisions = decision_counts([row[0] for row in rows], [row[1] for row in rows], threshold)
    first_alert: dict[str, int] = {}
    for example, score in zip(examples, scores, strict=True):
        if example.label and score >= threshold and example.lead_turns is not None:
            first_alert.setdefault(example.prefix.conversation_id, example.lead_turns)
    return {
        "prefixes": len(examples),
        "conversations": size,
        "positive_conversations": sum(row[0] for row in rows),
        "conversation_weighted_prefix_brier": brier,
        "conversation_weighted_prefix_log_loss": log_loss,
        "conversation_weighted_prefix_roc_auc": weighted_auc(labels, scores, weights),
        "any_alert": decisions,
        "true_positive_first_alerts": len(first_alert),
        "mean_lead_turns_for_true_positives": math.fsum(first_alert.values()) / len(first_alert)
        if first_alert
        else None,
        "threshold": threshold,
    }
