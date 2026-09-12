"""Forecast workflow acceptance with separately labelled numerical fixtures."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from turnscope import neural_forecast as api
from turnscope.models import Conversation, Utterance
from turnscope.neural_forecast import HierarchicalEventForecaster, NeuralForecastConfig
from turnscope.neural_forecast_data import (
    ObservedPrefix,
    ObservedTurn,
    SequencePolicy,
    prepare_sequence_forecasts,
)
from turnscope.neural_forecast_math import (
    FrozenNeuralParameters,
    admit_encoded_turns,
    parameter_shapes,
)
from turnscope.neural_forecast_train import NeuralEpoch, NeuralTrainingConfig, NeuralTrainingResult
from turnscope.neural_token_data import encode_observed_prefix

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def partition(name):
    rows = [
        (True, ["red", "red bright", "red reply", "future attack"]),
        (False, ["blue", "blue calm", "censored end"]),
        (True, ["red warm", "a question", "future attack"]),
    ]
    return [
        Conversation(
            f"{name}-{i}",
            [
                Utterance(
                    str(j),
                    "person",
                    text,
                    BASE + timedelta(seconds=j),
                    metadata={"event": label and j == len(texts) - 1},
                )
                for j, text in enumerate(texts)
            ],
            metadata={"forecast_groups": [f"scope:{name}:{i}"]},
        )
        for i, (label, texts) in enumerate(rows)
    ]


def settings(**changes):
    return replace(NeuralForecastConfig(embedding_dim=3, word_hidden=2, turn_hidden=4), **changes)


def new_model(**changes):
    return HierarchicalEventForecaster(
        config=settings(),
        training_config=NeuralTrainingConfig(epochs=1, batch_conversations=2, seed=5),
        **changes,
    )


def zero_training(train, val, vocabulary, architecture, *, config, numeric_limits, **kwargs):
    """Test-only constant-logit candidate: this is deliberately not neural training."""
    arrays = {
        name: np.zeros(shape, dtype="<f4") for name, shape in parameter_shapes(architecture).items()
    }
    parameters = FrozenNeuralParameters.from_arrays(architecture, arrays, limits=numeric_limits)
    initial = tuple(
        (name, hashlib.sha256(array.tobytes()).hexdigest()) for name, array in arrays.items()
    )
    return NeuralTrainingResult(
        parameters,
        config,
        (NeuralEpoch(1, math.log(2), math.log(2), 1, 0.0),),
        1,
        initial,
        (),
        train.digest,
        val.digest,
        vocabulary.digest,
        len(train.observations),
        len(val.observations),
        len(train.examples),
        len(val.examples),
        1,
        1,
        "synthetic-fixture-no-Torch",
        np.__version__,
        1,
    )


@pytest.fixture
def zero_model(monkeypatch):
    monkeypatch.setattr(api, "train_sequence_model", zero_training)
    return new_model().fit(
        partition("train"), partition("validation"), policy_validation=partition("policy")
    )


@pytest.fixture(scope="module")
def actual_model():
    pytest.importorskip(
        "torch", reason="real fit workflow requires the optional training dependency"
    )
    return new_model().fit(
        partition("actual-train"),
        partition("actual-validation"),
        policy_validation=partition("actual-policy"),
    )


def test_actual_three_partition_fit_predict_transform_and_heldout_evaluation(actual_model):
    model = actual_model
    state = model.state
    assert not state.policy_reuses_model_validation
    assert state.training_groups.isdisjoint(
        state.validation_groups | state.policy_validation_groups
    )
    assert state.validation_groups.isdisjoint(state.policy_validation_groups)
    assert state.training.selected_epoch == 1 and state.training.changed_parameter_names
    assert all(
        word not in state.vocabulary.tokens for word in ("future", "attack", "censored", "end")
    )
    heldout = model.prepare(partition("actual-test"))
    predictions = model.transform(heldout.observations)
    assert predictions == tuple(model.predict(prefix) for prefix in heldout.observations)
    for prefix, prediction in zip(heldout.observations, predictions, strict=True):
        assert prediction.model_digest == model.digest and prediction.input_digest == prefix.digest
        assert prediction.alert == (prediction.probability >= prediction.threshold)
        assert 0 < prediction.probability < 1 and math.isfinite(prediction.logit)
    report = model.evaluate(heldout)
    assert report["metrics"]["conversations"] == 3 and report["metrics"]["prefixes"] == 4
    assert report["metrics"]["conversation_weighted_prefix_brier"] >= 0
    assert report["probability_calibration_claimed"] is False
    assert report["whole_repository_parity_claimed"] is False


def test_threshold_selection_independent_exact_conversation_max_oracle(actual_model):
    model = actual_model
    data = model.prepare(partition("actual-policy"))
    scores = [model.predict(data.prefix(example)).probability for example in data.examples]
    maxima = {}
    for example, score in zip(data.examples, scores, strict=True):
        prior = maxima.get(example.conversation_index, (example.label, 0.0))[1]
        maxima[example.conversation_index] = (example.label, max(prior, score))
    positives = sum(label for label, _ in maxima.values())
    negatives = len(maxima) - positives
    candidates = {0.0, 1.0, *(score for _, score in maxima.values())}
    choices = []
    for threshold in candidates:
        tp = sum(label and score >= threshold for label, score in maxima.values())
        tn = sum(not label and score < threshold for label, score in maxima.values())
        choices.append(((Fraction(tp, positives) + Fraction(tn, negatives)) / 2, threshold))
    score, threshold = max(choices)
    assert model.state.threshold == threshold
    assert model.state.policy_validation_balanced_accuracy == pytest.approx(float(score))


def test_frozen_constant_logit_policy_and_metrics_have_hand_expected_values(zero_model):
    model = zero_model
    assert model.state.threshold == 1.0  # all maxima=.5; higher tied threshold wins
    assert model.state.policy_validation_balanced_accuracy == 0.5
    report = model.evaluate(partition("test"))
    metrics = report["metrics"]
    assert metrics["conversation_weighted_prefix_brier"] == 0.25
    assert metrics["conversation_weighted_prefix_log_loss"] == pytest.approx(math.log(2))
    assert metrics["conversation_weighted_prefix_roc_auc"] == 0.5
    assert {key: metrics["any_alert"][key] for key in ("tp", "fp", "tn", "fn")} == {
        "tp": 0,
        "fp": 0,
        "tn": 1,
        "fn": 2,
    }
    assert metrics["any_alert"]["balanced_accuracy"] == 0.5
    assert metrics["true_positive_first_alerts"] == 0
    assert metrics["mean_lead_turns_for_true_positives"] is None


def test_two_partition_reuse_is_explicit_and_third_partition_cannot_alias(zero_model):
    model = new_model().fit(partition("t"), partition("v"))
    state = model.state
    assert state.policy_reuses_model_validation
    assert state.policy_validation_groups == state.validation_groups
    assert state.policy_validation_digest == state.training.validation_partition_digest
    assert model.evaluate(partition("unseen"))["policy_reuses_model_validation"] is True
    before = model.digest
    with pytest.raises(ValueError, match="policy-validation groups overlap"):
        model.fit(partition("t"), partition("v"), policy_validation=partition("v"))
    assert model.digest == before


def test_excluded_group_overlap_is_rejected_before_training_and_evaluation(zero_model, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("training/inference ran before group validation")

    extra = Conversation(
        "excluded",
        [Utterance("0", "person", "already event", BASE, metadata={"event": True})],
        metadata={"forecast_groups": ["scope:validation:0"]},
    )
    monkeypatch.setattr(api, "train_sequence_model", forbidden)
    with pytest.raises(ValueError, match="overlap"):
        new_model().fit([*partition("another-train"), extra], partition("validation"))
    monkeypatch.setattr(api, "infer_encoded_turns", forbidden)
    with pytest.raises(ValueError, match="evaluation groups overlap"):
        zero_model.evaluate([*partition("heldout"), extra])


def test_cached_model_identity_and_exported_summary_are_immutable(zero_model, monkeypatch):
    model = zero_model
    first = model.digest
    summary = model.training_summary
    summary["config"]["word_hidden"] = 100
    summary["history"][0]["validation_loss"] = 999
    assert model.training_summary["config"]["word_hidden"] == 2
    assert model.digest == first
    with pytest.raises(FrozenInstanceError):
        model.state.threshold = 0.1
    array = model.state.parameters.arrays()["embedding.weight"]
    with pytest.raises(ValueError):
        array.flags.writeable = True

    def forbidden_digest(_self):
        raise AssertionError("prediction rehashed every parameter byte")

    monkeypatch.setattr(FrozenNeuralParameters, "digest", property(forbidden_digest))
    result = model.predict(model.prepare(partition("q")).observations[0])
    assert result.model_digest == first and model.digest == first


@pytest.mark.parametrize("stage", ["optimizer", "policy-inference"])
def test_failed_refit_keeps_exact_prior_model(zero_model, monkeypatch, stage):
    model = zero_model
    old_state, digest = model.state, model.digest
    query = model.prepare(partition("q")).observations[0]
    prediction = model.predict(query)

    def failure(*args, **kwargs):
        raise ValueError("injected candidate failure")

    monkeypatch.setattr(
        api, "train_sequence_model" if stage == "optimizer" else "_dataset_scores", failure
    )
    with pytest.raises(ValueError, match="injected candidate failure"):
        model.fit(partition("new-train"), partition("new-validation"))
    assert (
        model.state is old_state and model.digest == digest and model.predict(query) == prediction
    )


def test_future_and_labels_do_not_enter_predict_views_or_vocabulary(zero_model):
    source = partition("train")
    changed = [
        Conversation(
            row.id,
            [
                *row.utterances[:-1],
                replace(row.utterances[-1], text="a wholly different future phrase"),
            ],
            row.metadata,
        )
        for row in source
    ]
    clone = new_model().fit(changed, partition("validation"), policy_validation=partition("policy"))
    assert clone.state.vocabulary == zero_model.state.vocabulary
    assert clone.state.training.parameters.digest == zero_model.state.training.parameters.digest
    assert (
        clone.state.training.training_partition_digest
        != zero_model.state.training.training_partition_digest
    )
    query = Conversation("query", source[0].utterances[:2], {"forecast_groups": "not inspected"})
    first = zero_model.predict_conversation(query)
    query.utterances[0].metadata["event"] = {"unknown": "metadata"}
    assert zero_model.predict_conversation(query) == first
    with pytest.raises(ValueError, match="ObservedPrefix"):
        zero_model.predict(query)


def test_empty_and_unknown_predictions_report_retention_without_confidence_claims(zero_model):
    empty = ObservedPrefix("empty", (ObservedTurn("0", BASE, ""), ObservedTurn("1", BASE, "")))
    result = zero_model.predict(empty).to_dict()
    assert result["raw_tokens"] == result["retained_tokens"] == result["known_tokens"] == 0
    assert result["retained_token_fraction"] == 1 and result["known_retained_token_fraction"] == 0
    unknown = replace(empty, turns=tuple(replace(turn, text="unknown") for turn in empty.turns))
    assert zero_model.predict(unknown).known_tokens == 0
    assert zero_model.transform([]) == ()


def test_single_prediction_checks_configured_affine_cap_before_inference(zero_model, monkeypatch):
    query = zero_model.prepare(partition("query")).observations[0]
    encoded = encode_observed_prefix(query, zero_model.state.vocabulary)
    work = admit_encoded_turns(
        zero_model.state.parameters.architecture, encoded.turns
    ).affine_multiplications
    limited = new_model()
    limited._state = replace(
        zero_model.state,
        config=replace(zero_model.state.config, max_inference_affine_multiplications=work - 1),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy inference before configured affine admission")

    monkeypatch.setattr(api, "infer_encoded_turns", forbidden)
    with pytest.raises(ValueError, match="max_inference_affine"):
        limited.predict(query)


@pytest.mark.parametrize("kind", ["conversations", "turns", "tokens", "bytes", "affine"])
def test_transform_aggregate_failures_precede_all_inference(zero_model, monkeypatch, kind):
    query = ObservedPrefix("q", (ObservedTurn("0", BASE, "red"), ObservedTurn("1", BASE, "blue")))
    bounds = zero_model.state.data_limits
    config = zero_model.state.config
    changes = {
        "conversations": {"max_conversations": 3},
        "turns": {"max_source_turns": 3},
        "tokens": {"max_tokens": 3},
        "bytes": {
            "max_source_bytes": sum(
                len(word.encode()) for word in zero_model.state.vocabulary.tokens
            )
            + 1
        },
    }
    if kind in changes:
        bounds = replace(bounds, **changes[kind])
    else:
        encoded = encode_observed_prefix(query, zero_model.state.vocabulary)
        work = admit_encoded_turns(
            zero_model.state.parameters.architecture, encoded.turns
        ).affine_multiplications
        config = replace(config, max_inference_affine_multiplications=work + 1)
    limited = new_model()
    limited._state = replace(zero_model.state, data_limits=bounds, config=config)
    queries = [query, query]
    if kind == "conversations":
        queries = [query] * 4
    elif kind == "bytes":
        queries = [query] * (bounds.max_source_bytes // 10 + 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("inference happened before whole-collection admission")

    monkeypatch.setattr(api, "infer_encoded_turns", forbidden)
    with pytest.raises(ValueError, match="exceeds"):
        limited.transform(queries)


def test_policy_budget_fails_before_training_and_dataset_eval_before_any_inference(
    zero_model, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("expensive work before aggregate admission")

    monkeypatch.setattr(api, "train_sequence_model", forbidden)
    limited = HierarchicalEventForecaster(config=settings(max_inference_affine_multiplications=1))
    with pytest.raises(ValueError, match="policy validation exceeds"):
        limited.fit(partition("train"), partition("validation"))
    limited._state = replace(
        zero_model.state,
        config=replace(zero_model.state.config, max_inference_affine_multiplications=1),
    )
    monkeypatch.setattr(api, "infer_encoded_turns", forbidden)
    with pytest.raises(ValueError, match="inference dataset exceeds"):
        limited.evaluate(partition("heldout"))


@pytest.mark.parametrize(
    "change",
    [
        {"config": {}},
        {"training_config": {}},
        {"policy": {}},
        {"data_limits": {}},
        {"numeric_limits": {}},
        {"config": settings(max_turn_tokens=129)},
        {"policy": SequencePolicy(min_turns=65)},
    ],
)
def test_forecaster_configuration_contracts(change):
    with pytest.raises(ValueError):
        HierarchicalEventForecaster(**change)


def test_not_fitted_short_prefix_bad_policy_class_and_leaked_eval_inputs(zero_model):
    with pytest.raises(ValueError, match="not been fitted"):
        new_model().predict(zero_model.prepare(partition("q")).observations[0])
    with pytest.raises(ValueError, match="min_turns"):
        zero_model.predict(ObservedPrefix("q", (ObservedTurn("0", BASE, "x"),)))
    with pytest.raises(ValueError, match="both classes"):
        new_model().fit(partition("t"), partition("v"), policy_validation=partition("p")[:1])
    with pytest.raises(ValueError, match="overlap"):
        zero_model.evaluate(partition("train"))
    with pytest.raises(ValueError, match="eligible"):
        zero_model.evaluate([])
    with pytest.raises(ValueError, match="ObservedPrefix"):
        zero_model.transform([None])
    short = prepare_sequence_forecasts(partition("new"), policy=SequencePolicy(min_turns=1))
    with pytest.raises(ValueError, match="before configured min_turns"):
        zero_model.evaluate(short)


def test_actual_fit_parameters_predict_in_fresh_process_that_forbids_torch(actual_model):
    """Ephemeral test-only JSON transfer, not a production model serialization API."""
    model = actual_model
    state = model.state
    result = state.training
    training = {key: value for key, value in asdict(result).items() if key != "parameters"}
    # Avoid ever accepting pickle: the child receives only our bounded tiny
    # test candidate's explicit numbers, records, and closed constructor fields.
    payload = {
        "config": state.config.to_dict(),
        "policy": state.policy.to_dict(),
        "data_limits": state.data_limits.to_dict(),
        "numeric_limits": state.parameters.limits.to_dict(),
        "vocabulary": state.vocabulary.to_dict(),
        "architecture": state.parameters.architecture.to_dict(),
        "arrays": {name: array.tolist() for name, array in state.parameters.arrays().items()},
        "training": training,
        "state": {
            "threshold": state.threshold,
            "policy_validation_balanced_accuracy": state.policy_validation_balanced_accuracy,
            "training_groups": sorted(state.training_groups),
            "validation_groups": sorted(state.validation_groups),
            "policy_validation_groups": sorted(state.policy_validation_groups),
            "policy_validation_digest": state.policy_validation_digest,
            "policy_validation_conversations": state.policy_validation_conversations,
            "policy_validation_prefixes": state.policy_validation_prefixes,
            "policy_reuses_model_validation": state.policy_reuses_model_validation,
        },
        "query": model.prepare(partition("fresh-process")).observations[0].to_dict(),
    }
    script = r"""
import importlib.abc, json, sys
class ForbidTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("frozen prediction tried to import Torch")
sys.meta_path.insert(0, ForbidTorch())
import numpy as np
from turnscope.neural_forecast import (
    HierarchicalEventForecaster, NeuralForecastConfig, NeuralForecastState)
from turnscope.neural_forecast_data import ObservedPrefix, SequenceLimits, SequencePolicy
from turnscope.neural_forecast_math import (
    FrozenNeuralParameters, NeuralArchitecture, NeuralNumericLimits)
from turnscope.neural_forecast_train import NeuralEpoch, NeuralTrainingConfig, NeuralTrainingResult
from turnscope.neural_token_data import SequenceVocabulary
data = json.load(sys.stdin)
parameters = FrozenNeuralParameters.from_arrays(NeuralArchitecture(**data["architecture"]),
    {name: np.array(value, dtype="<f4") for name, value in data["arrays"].items()},
    limits=NeuralNumericLimits(**data["numeric_limits"]))
training = data["training"]
training["config"] = NeuralTrainingConfig(**training["config"])
training["history"] = tuple(NeuralEpoch(**row) for row in training["history"])
training["initial_parameter_sha256"] = tuple(
    tuple(row) for row in training["initial_parameter_sha256"])
training["changed_parameter_names"] = tuple(training["changed_parameter_names"])
training = NeuralTrainingResult(parameters=parameters, **training)
extra = data["state"]
for name in ("training_groups", "validation_groups", "policy_validation_groups"):
    extra[name] = frozenset(extra[name])
state = NeuralForecastState(
    NeuralForecastConfig(**data["config"]), SequencePolicy.from_dict(data["policy"]),
    SequenceLimits.from_dict(data["data_limits"]),
    SequenceVocabulary.from_dict(data["vocabulary"]), training, **extra)
model = HierarchicalEventForecaster()
model._state = state
print(json.dumps({"prediction": model.predict(ObservedPrefix.from_dict(data["query"])).to_dict(),
    "torch_imported": any(name == "torch" or name.startswith("torch.") for name in sys.modules)}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
        env=environment,
    )
    output = json.loads(completed.stdout)
    assert output["torch_imported"] is False
    assert (
        output["prediction"] == model.predict(ObservedPrefix.from_dict(payload["query"])).to_dict()
    )
