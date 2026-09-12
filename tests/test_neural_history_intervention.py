"""Original prefix-local intervention oracles, not official corpus/model quality tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from turnscope import neural_forecast as forecast
from turnscope.forecast_metrics import forecast_metrics
from turnscope.models import Conversation, Utterance
from turnscope.neural_forecast import HierarchicalEventForecaster, NeuralForecastConfig
from turnscope.neural_forecast_math import (
    FrozenNeuralParameters,
    infer_encoded_turns,
    parameter_shapes,
)
from turnscope.neural_forecast_train import NeuralEpoch, NeuralTrainingConfig, NeuralTrainingResult
from turnscope.neural_token_data import encode_observed_prefix

_PATH = Path(__file__).resolve().parents[1] / "benchmarks/neural_history_intervention.py"
_SPEC = importlib.util.spec_from_file_location("_neural_history_intervention_tests", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
intervention = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = intervention
_SPEC.loader.exec_module(intervention)
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def oracle_order(conversation_id, ids):
    pairs = []
    for index in range(len(ids) - 1):
        raw = canonical(
            {
                "format": "turnscope.prefix-history-permutation.v1",
                "seed": "turnscope-history-order/2026-09-12",
                "conversation_id": conversation_id,
                "endpoint_turn_id": ids[-1],
                "historical_turn_id": ids[index],
            }
        )
        pairs.append((hashlib.sha256(raw).digest(), index))
    return (*[pair[1] for pair in sorted(pairs)], len(ids) - 1)


def partition(name, *, repeated=False):
    return [
        Conversation(
            f"{name}-{i}",
            [
                Utterance(
                    f"turn-{j}",
                    "person",
                    ("same same" if repeated else text),
                    BASE + timedelta(seconds=j),
                    metadata={"event": i == 0 and j == 5},
                )
                for j, text in enumerate(
                    ("red 🌿", "blue calm", "red blue", "{{template}} blue", "red", "FUTURE_SECRET")
                )
            ],
            metadata={"forecast_groups": [f"group-{name}-{i}"]},
        )
        for i in range(2)
    ]


def numerical_candidate(train, val, vocabulary, architecture, *, config, numeric_limits, **_kwargs):
    """Explicitly untrained fixture: deterministic small arrays, not fake fit evidence."""
    arrays = {}
    for index, (name, shape) in enumerate(parameter_shapes(architecture).items()):
        arrays[name] = np.array(
            [math.sin(position + 7 * index) / 5 for position in range(math.prod(shape))],
            dtype="<f4",
        ).reshape(shape)
    arrays["embedding.weight"][0] = 0
    parameters = FrozenNeuralParameters.from_arrays(architecture, arrays, limits=numeric_limits)
    hashes = tuple(
        (name, hashlib.sha256(value.tobytes()).hexdigest()) for name, value in arrays.items()
    )
    return NeuralTrainingResult(
        parameters,
        config,
        (NeuralEpoch(1, 1.0, 1.0, 1, 0.0),),
        1,
        hashes,
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
        "authored-numerical-fixture-no-training",
        np.__version__,
        1,
    )


def fixture_model(monkeypatch, word_layers=1, turn_layers=1):
    model = HierarchicalEventForecaster(
        config=NeuralForecastConfig(
            embedding_dim=3,
            word_hidden=2,
            turn_hidden=4,
            word_layers=word_layers,
            turn_layers=turn_layers,
        ),
        training_config=NeuralTrainingConfig(epochs=1, seed=17),
    )
    with monkeypatch.context() as patch:
        patch.setattr(forecast, "train_sequence_model", numerical_candidate)
        model.fit(
            partition("train"), partition("validation"), policy_validation=partition("policy")
        )
    return model


def fully_recomputed(state, data):
    scores, originals, transformed, orders = [], [], [], []
    changed_indices = changed_content = eligible_history = 0
    for example in data.examples:
        observation = data.observations[example.conversation_index]
        encoded = encode_observed_prefix(observation, state.vocabulary).turns[
            : example.endpoint + 1
        ]
        ids = tuple(turn.id for turn in observation.turns[: example.endpoint + 1])
        order = oracle_order(observation.conversation_id, ids)
        moved = tuple(encoded[position] for position in order)
        scores.append(
            forecast._clip(infer_encoded_turns(state.parameters, moved).probabilities[-1])
        )
        originals.append(encoded)
        transformed.append(moved)
        orders.append(order)
        changed_indices += order != tuple(range(len(ids)))
        changed_content += moved != encoded
        eligible_history += len(ids) >= 3
    return scores, {
        "original_prefix_inventory_sha256": hashlib.sha256(
            b"".join(canonical(v) + b"\n" for v in originals)
        ).hexdigest(),
        "transformed_prefix_inventory_sha256": hashlib.sha256(
            b"".join(canonical(v) + b"\n" for v in transformed)
        ).hexdigest(),
        "ordering_inventory_sha256": hashlib.sha256(
            b"".join(canonical(v) + b"\n" for v in orders)
        ).hexdigest(),
        "index_order_changed_prefixes": changed_indices,
        "encoded_content_changed_prefixes": changed_content,
        "prefixes_with_two_historical_turns": eligible_history,
    }


@pytest.mark.parametrize("length", (1, 2, 3, 6, 256))
def test_independent_utf8_hash_order_current_last_and_future_ids_absent(length):
    ids = tuple(f"消息-{{{{{i}}}}}" for i in range(length))
    expected = oracle_order("discussion-🌿", ids)
    actual = intervention.prefix_history_order("discussion-🌿", ids)
    assert actual == expected and actual[-1] == length - 1 and sorted(actual) == list(range(length))
    larger = (*ids, "future-that-must-not-be-hashed")
    assert intervention.prefix_history_order("discussion-🌿", larger[:length]) == actual
    if length <= 2:
        assert actual == tuple(range(length))


@pytest.mark.parametrize("word_layers", (1, 2))
@pytest.mark.parametrize("turn_layers", (1, 2))
def test_cached_scores_match_full_recomputation_all_layer_pairs(
    monkeypatch, word_layers, turn_layers
):
    model = fixture_model(monkeypatch, word_layers, turn_layers)
    data = model.prepare(partition("heldout"))
    before = model.digest
    expected_scores, expected_audit = fully_recomputed(model.state, data)
    captured = []
    original = intervention._cached_scores

    def record_scores(*args):
        values = original(*args)
        captured.extend(values)
        return values

    monkeypatch.setattr(intervention, "_cached_scores", record_scores)
    report = intervention.evaluate_history_intervention(model, data)
    np.testing.assert_allclose(captured, expected_scores, rtol=0, atol=2e-15)
    for name, value in expected_audit.items():
        assert report["audit"][name] == value
    expected_metrics = forecast_metrics(
        forecast._metric_examples(data), expected_scores, model.state.threshold
    )
    assert report["metrics"]["threshold"] == model.state.threshold
    for metric in ("conversation_weighted_prefix_brier", "conversation_weighted_prefix_log_loss"):
        assert report["metrics"][metric] == pytest.approx(
            expected_metrics[metric], abs=1e-14, rel=0
        )
    assert report["metrics"]["any_alert"] == expected_metrics["any_alert"]
    assert report["prediction_inventory_sha256"] == hashlib.sha256(canonical(captured)).hexdigest()
    assert report["model_digest"] == model.digest == before
    assert report["partition_digest"] == data.digest
    assert report["weights_refitted"] is report["threshold_refitted"] is False
    assert report["out_of_distribution_diagnostic"] is True
    assert report["causal_effect_claimed"] is report["whole_repository_parity_claimed"] is False
    encoded_json = json.dumps(report)
    assert all(observation.conversation_id not in encoded_json for observation in data.observations)
    assert "FUTURE_SECRET" not in encoded_json and "{{template}}" not in encoded_json


def test_unique_word_encodings_once_and_turn_recurrence_restarts_per_original_endpoint(monkeypatch):
    model = fixture_model(monkeypatch, 2, 2)
    data = model.prepare(partition("heldout"))
    calls = []
    original = intervention._gru_direction

    def spy(np_module, sequence, arrays, level, layer, hidden, reverse):
        calls.append((level, layer, reverse, len(sequence)))
        return original(np_module, sequence, arrays, level, layer, hidden, reverse)

    monkeypatch.setattr(intervention, "_gru_direction", spy)
    report = intervention.evaluate_history_intervention(model, data)
    unique_turns = sum(len(item.turns) for item in data.observations)
    assert len([row for row in calls if row[0] == "word"]) == unique_turns * 2 * 2
    assert len([row for row in calls if row[0] == "turn"]) == len(data.examples) * 2
    expected_lengths = [example.endpoint + 1 for example in data.examples for _ in range(2)]
    assert [row[3] for row in calls if row[0] == "turn"] == expected_lengths
    assert report["audit"]["unique_completed_turns"] == unique_turns
    assert report["audit"]["permutation_positions"] == sum(expected_lengths) // 2


def test_identical_encoded_content_counts_index_changes_without_claiming_content_changes(
    monkeypatch,
):
    model = fixture_model(monkeypatch)
    data = model.prepare(partition("repeated", repeated=True))
    report = intervention.evaluate_history_intervention(model, data)
    _, expected = fully_recomputed(model.state, data)
    assert expected["index_order_changed_prefixes"] > 0
    assert (
        report["audit"]["index_order_changed_prefixes"] == expected["index_order_changed_prefixes"]
    )
    assert report["audit"]["encoded_content_changed_prefixes"] == 0
    assert report["audit"]["eligible_prefixes"] == len(data.examples)
    assert (
        report["audit"]["original_prefix_inventory_sha256"]
        == report["audit"]["transformed_prefix_inventory_sha256"]
    )


def test_future_turn_id_or_text_cannot_change_earlier_intervened_score(monkeypatch):
    model = fixture_model(monkeypatch)
    source = partition("future-isolation")
    earlier = model.prepare(source)
    outputs, _ = fully_recomputed(model.state, earlier)
    revised_turns = list(source[0].utterances)
    revised_turns[-2] = replace(
        revised_turns[-2], id="changed-last-observed-id", text="blue blue blue"
    )
    source[0] = replace(source[0], utterances=revised_turns)
    changed = model.prepare(source)
    later, _ = fully_recomputed(model.state, changed)
    # Only the final eligible prefix of conversation zero includes the changed turn.
    first_conversation_count = sum(example.conversation_index == 0 for example in earlier.examples)
    np.testing.assert_allclose(
        later[: first_conversation_count - 1],
        outputs[: first_conversation_count - 1],
        rtol=0,
        atol=1e-15,
    )
    admitted_old, _ = intervention._admit(
        model.state, earlier, intervention.HistoryInterventionLimits()
    )
    admitted_new, _ = intervention._admit(
        model.state, changed, intervention.HistoryInterventionLimits()
    )
    assert admitted_old[0].orders[:-1] == admitted_new[0].orders[:-1]


def test_budget_inventory_exact_then_one_less_rejects_before_orders_or_numpy(monkeypatch):
    model = fixture_model(monkeypatch)
    data = model.prepare(partition("budgets"))
    required = sum(example.endpoint + 1 for example in data.examples)
    limits = intervention.HistoryInterventionLimits(required)
    assert (
        intervention.evaluate_history_intervention(model, data, limits=limits)["audit"][
            "permutation_positions"
        ]
        == required
    )
    monkeypatch.setattr(
        intervention,
        "prefix_history_order",
        lambda *_a: pytest.fail("orders constructed before admission"),
    )
    monkeypatch.setattr(
        intervention, "_numpy", lambda: pytest.fail("arrays allocated before admission")
    )
    with pytest.raises(ValueError, match="max_permutation_positions"):
        intervention.evaluate_history_intervention(
            model, data, limits=replace(limits, max_permutation_positions=required - 1)
        )


def test_affine_and_cache_workspace_formula_exact_and_tight_limits(monkeypatch):
    model = fixture_model(monkeypatch, 2, 2)
    data = model.prepare(partition("work"))
    report = intervention.evaluate_history_intervention(model, data)
    state = model.state
    positions = sum(
        len(turn)
        for item in data.observations
        for turn in encode_observed_prefix(item, state.vocabulary).turns
    )
    steps = sum(example.endpoint + 1 for example in data.examples)
    word = positions * 6 * 2 * ((3 + 2) + (4 + 2))
    turn = steps * 3 * 4 * ((4 + 4) + (4 + 4))
    head = len(data.examples) * 2 * 5
    assert report["audit"]["affine_multiplications"] == word + turn + head
    assert report["audit"]["word_affine_multiplications"] == word
    assert report["audit"]["turn_affine_multiplications"] == turn
    maximum = report["audit"]["maximum_estimated_workspace_bytes"]
    new_params = replace(
        state.parameters, limits=replace(state.parameters.limits, max_workspace_bytes=maximum)
    )
    model._state = replace(
        state,
        config=replace(state.config, max_inference_affine_multiplications=word + turn + head),
        training=replace(state.training, parameters=new_params),
    )
    assert intervention.evaluate_history_intervention(model, data)["audit"] == report["audit"]
    monkeypatch.setattr(intervention, "_numpy", lambda: pytest.fail("NumPy reached rejected work"))
    for cap in ("affine", "workspace"):
        if cap == "affine":
            model._state = replace(
                state,
                config=replace(
                    state.config, max_inference_affine_multiplications=word + turn + head - 1
                ),
            )
            match = "max_inference_affine_multiplications"
        else:
            bad = replace(
                state.parameters,
                limits=replace(state.parameters.limits, max_workspace_bytes=maximum - 1),
            )
            model._state = replace(state, training=replace(state.training, parameters=bad))
            match = "max_workspace_bytes"
        with pytest.raises(ValueError, match=match):
            intervention.evaluate_history_intervention(model, data)


def test_evaluation_pins_one_state_even_if_owner_replaces_model_during_admission(monkeypatch):
    model = fixture_model(monkeypatch)
    data = model.prepare(partition("heldout"))
    original_state = model.state
    expected = intervention.evaluate_history_intervention(model, data)
    original_order = intervention.prefix_history_order

    def replace_owner(*args):
        model._state = replace(original_state, threshold=0.123)
        return original_order(*args)

    monkeypatch.setattr(intervention, "prefix_history_order", replace_owner)
    actual = intervention.evaluate_history_intervention(model, data)
    assert actual == expected and model.digest != original_state.digest


def test_empty_and_overlapping_source_groups_reject_before_arrays(monkeypatch):
    model = fixture_model(monkeypatch)
    empty = model.prepare([])
    overlap = model.prepare(partition("train"))
    monkeypatch.setattr(intervention, "_numpy", lambda: pytest.fail("invalid data reached arrays"))
    for data, match in ((empty, "eligible"), (overlap, "overlap")):
        with pytest.raises(ValueError, match=match):
            intervention.evaluate_history_intervention(model, data)


@pytest.mark.parametrize("value", (True, 0, -1, 1.0, 8_000_001, "2", None))
def test_permutation_inventory_budget_is_strict(value):
    with pytest.raises(ValueError):
        intervention.HistoryInterventionLimits(value)


def test_actual_tiny_trained_main_uses_its_unchanged_threshold_and_all_endpoints():
    torch = pytest.importorskip("torch", reason="optional actual CPU training oracle")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = HierarchicalEventForecaster(
            config=NeuralForecastConfig(embedding_dim=3, word_hidden=2, turn_hidden=4),
            training_config=NeuralTrainingConfig(epochs=1, batch_conversations=2, seed=17),
        ).fit(
            partition("actual-train"),
            partition("actual-val"),
            policy_validation=partition("actual-policy"),
        )
        data = model.prepare(partition("actual-heldout"))
        expected, _ = fully_recomputed(model.state, data)
        before = model.digest
        report = intervention.evaluate_history_intervention(model, data)
        assert model.state.training.changed_parameter_names
        assert model.state.training.history[0].optimizer_steps == 1
        assert report["model_digest"] == model.digest == before
        assert report["metrics"]["threshold"] == model.state.threshold
        assert report["metrics"]["prefixes"] == len(data.examples)
        metrics = forecast_metrics(forecast._metric_examples(data), expected, model.state.threshold)
        assert report["metrics"]["any_alert"] == metrics["any_alert"]
        assert report["metrics"]["conversation_weighted_prefix_log_loss"] == pytest.approx(
            metrics["conversation_weighted_prefix_log_loss"], abs=1e-14, rel=0
        )
    finally:
        torch.set_num_threads(threads)
