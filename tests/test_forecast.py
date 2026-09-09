from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction

import pytest

from turnscope import (
    Conversation,
    ForecastExample,
    ForecastPrefix,
    PrefixEventForecaster,
    Utterance,
    prepare_forecast_examples,
)
from turnscope.cli import main
from turnscope.forecast_artifact import checksum
from turnscope.forecast_metrics import choose_threshold, forecast_metrics, weighted_auc
from turnscope.io import conversation_to_dict

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def conversation(identifier, texts, events=None, *, times=None, groups=None):
    events = events or [False] * len(texts)
    times = times or list(range(len(texts)))
    return Conversation(
        identifier,
        [
            Utterance(
                str(i),
                "speaker",
                text,
                BASE + timedelta(seconds=times[i]),
                metadata={"event": events[i]},
            )
            for i, text in enumerate(texts)
        ],
        metadata={"forecast_groups": groups or ["page:" + identifier]},
    )


def partition(name):
    return [
        conversation(name + "-p", ["red", "red", "future_attack"], [False, False, True]),
        conversation(name + "-n", ["blue", "blue", "future_neutral"]),
    ]


def trained():
    return PrefixEventForecaster().fit(partition("train"), partition("val"))


def resign(artifact):
    artifact["sha256"] = checksum(
        {key: value for key, value in artifact.items() if key != "sha256"}
    )
    return artifact


def test_independent_two_class_hand_arithmetic():
    model = trained()
    assert model.state.vocabulary == ("blue", "red")
    # One prefix per conversation, positive red=2 and negative blue=2.
    # Add-one likelihoods: P(red|positive)=3/4; P(red|negative)=1/4.
    positive = model.predict(conversation("observed", ["red", "red"]))
    assert positive.probability == pytest.approx(9 / 10)
    assert model.predict(conversation("observed", ["blue", "blue"])).probability == pytest.approx(
        1 / 10
    )
    assert model.state.positive_prior == 0.5
    assert model.state.threshold == pytest.approx(0.9)
    assert model.state.validation_balanced_accuracy == 1
    metrics = model.evaluate(partition("test"))
    assert metrics["model"]["conversation_weighted_prefix_brier"] == pytest.approx(0.01)
    assert metrics["model"]["conversation_weighted_prefix_log_loss"] == pytest.approx(
        -math.log(0.9)
    )
    assert metrics["model"]["conversation_weighted_prefix_roc_auc"] == 1
    assert metrics["model"]["any_alert"]["tp"] == metrics["model"]["any_alert"]["tn"] == 1
    assert metrics["model"]["mean_lead_turns_for_true_positives"] == 1
    assert metrics["training_prior_baseline"]["conversation_weighted_prefix_brier"] == 0.25
    assert metrics["training_prior_baseline"]["conversation_weighted_prefix_roc_auc"] == 0.5


def test_multiple_prefixes_have_one_conversation_total_weight():
    documents = [
        conversation("p", ["red", "red", "red", "secret"], [False, False, False, True]),
        conversation("n", ["blue", "blue", "other"]),
    ]
    model = PrefixEventForecaster().fit(documents, partition("val"))
    # Positive prefixes counts2 and3 each weight1/2 ->2.5 red tokens.
    # P(red|positive)=(2.5+1)/(2.5+2)=7/9; negative=1/4.
    # Posterior for two red tokens: (28/9)^2 / (1+(28/9)^2).
    assert model.state.training_conversations == 2
    assert model.state.training_prefixes == 3
    assert model.state.positive_prior == 0.5
    assert model.predict(conversation("q", ["red", "red"])).probability == pytest.approx(784 / 865)


def test_authentic_future_boundaries_exclude_attacked_and_terminal_prefixes():
    document = conversation(
        "p", ["a", "b", "c", "attack", "after"], [False, False, False, True, False]
    )
    dataset = prepare_forecast_examples([document])
    assert [(item.prefix.turns, item.lead_turns) for item in dataset.examples] == [(2, 2), (3, 1)]
    assert all(item.label for item in dataset.examples)
    assert all(
        "attack" not in item.prefix.counts and "after" not in item.prefix.counts
        for item in dataset.examples
    )
    assert not hasattr(dataset.examples[0].prefix, "label")
    negative = prepare_forecast_examples([conversation("n", ["a", "b", "c", "last"])])
    assert [item.prefix.turns for item in negative.examples] == [2, 3]
    assert all(item.label is False and item.lead_turns is None for item in negative.examples)
    early = prepare_forecast_examples(
        [conversation("p", ["attack", "x", "y"], [True, False, False])]
    )
    assert early.examples == ()
    assert early.excluded_conversations == 1


def test_equal_time_blocks_and_headers_are_not_prediction_boundaries():
    document = conversation(
        "p", ["one", "two", "three", "attack"], [False, False, False, True], times=[0, 1, 1, 2]
    )
    assert [item.prefix.turns for item in prepare_forecast_examples([document]).examples] == [3]
    tied_event = conversation("p", ["one", "two", "attack"], [False, False, True], times=[0, 1, 1])
    assert prepare_forecast_examples([tied_event]).examples == ()
    source = conversation(
        "c", ["section_secret", "red", "red", "attack"], [False, False, False, True]
    )
    source.utterances[0].metadata.clear()
    source.utterances[0].metadata["is_section_header"] = True
    dataset = prepare_forecast_examples([source])
    assert len(dataset.examples) == 1
    assert dataset.examples[0].prefix.turns == 2
    assert dataset.examples[0].prefix.counts == {"red": 2}


def test_no_future_label_or_validation_vocabulary_leakage_and_atomic_refit():
    model = trained()
    before = model.to_dict()
    changed = partition("train")
    changed[0].utterances[-1].metadata["irrelevant_future_label"] = "secret"
    changed[0] = Conversation(
        changed[0].id,
        [
            *changed[0].utterances[:-1],
            replace(changed[0].utterances[-1], text="validationonly future secret"),
        ],
        metadata=changed[0].metadata,
    )
    assert PrefixEventForecaster().fit(changed, partition("val")).to_dict() == before
    validation = [
        conversation("v-p", ["val_only", "val_only", "a"], [False, False, True]),
        conversation("v-n", ["val_other", "val_other", "b"]),
    ]
    other = PrefixEventForecaster().fit(partition("train"), validation)
    assert other.state.vocabulary == model.state.vocabulary
    assert other.state.log_probabilities == model.state.log_probabilities
    assert other.state.threshold == 1  # Both scores tied; conservative validation tie-break.
    with pytest.raises(ValueError, match="both classes"):
        model.fit(partition("train"), [partition("val")[0]])
    assert model.to_dict() == before
    model.evaluate(partition("test"))
    assert model.to_dict() == before
    observed = conversation("observed", ["red", "red"])
    observed.metadata["target"] = True
    observed.utterances[0].metadata["event"] = True
    assert model.predict(observed).probability == pytest.approx(0.9)


def test_group_and_conversation_overlap_refused_even_when_no_prefixes():
    train, val = partition("train"), partition("val")
    val[0].metadata["forecast_groups"] = train[0].metadata["forecast_groups"]
    with pytest.raises(ValueError, match="groups overlap"):
        PrefixEventForecaster().fit(train, val)
    model = trained()
    for seen in (partition("train"), partition("val")):
        with pytest.raises(ValueError, match="groups overlap"):
            model.evaluate(seen)
    excluded = conversation("new", ["attacked"], [True], groups=["page:train-p"])
    with pytest.raises(ValueError, match="groups overlap"):
        model.evaluate([excluded])
    duplicate = partition("train")
    with pytest.raises(ValueError, match="unique"):
        prepare_forecast_examples([duplicate[0], duplicate[0]])


@pytest.mark.parametrize("groups", [None, [], [1], [""], ["x", "x"], "page:x"])
def test_missing_or_invalid_group_identity(groups):
    item = partition("x")[0]
    item.metadata["forecast_groups"] = groups
    with pytest.raises(ValueError, match="group"):
        prepare_forecast_examples([item])


@pytest.mark.parametrize("value", [None, 0, 1, "True"])
def test_strict_event_annotations(value):
    item = partition("x")[0]
    item.utterances[0].metadata["event"] = value
    with pytest.raises(ValueError, match="boolean event"):
        prepare_forecast_examples([item])


def test_snapshot_immutability_and_deterministic_training_order():
    source = partition("train")
    model = PrefixEventForecaster().fit(source, partition("val"))
    assert (
        PrefixEventForecaster().fit(reversed(source), reversed(partition("val"))).to_dict()
        == model.to_dict()
    )
    example = prepare_forecast_examples(source).examples[0]
    source[0].utterances[0].metadata["event"] = True
    with pytest.raises(TypeError):
        example.prefix.counts["x"] = 1
    with pytest.raises(FrozenInstanceError):
        example.label = False
    with pytest.raises(TypeError):
        model.state.log_probabilities[0]["x"] = 0
    with pytest.raises(FrozenInstanceError):
        model.state.threshold = 0
    exported = model.to_dict()
    exported["state"]["log_probabilities"][0][0] = 99
    assert model.to_dict() != exported


def test_prefix_budgets_validation_and_empty_word_baseline():
    for kwargs in ({"max_prefixes": 1}, {"max_prefix_cells": 1}, {"max_tokens": 1}):
        with pytest.raises(ValueError, match="budget"):
            prepare_forecast_examples(partition("x"), **kwargs)
    with pytest.raises(ValueError, match="nondecreasing"):
        prepare_forecast_examples([conversation("x", ["a", "b", "c"], times=[2, 1, 3])])
    item = partition("x")[0]
    duplicate = Conversation(item.id, [item.utterances[0]] * 2, item.metadata)
    with pytest.raises(ValueError, match="unique"):
        prepare_forecast_examples([duplicate])
    item.utterances[0].metadata["is_section_header"] = 1
    with pytest.raises(ValueError, match="boolean"):
        prepare_forecast_examples([item])
    with pytest.raises(TypeError, match="Conversation"):
        prepare_forecast_examples([None])
    with pytest.raises(ValueError, match="positive integer"):
        prepare_forecast_examples([], max_tokens=False)
    with pytest.raises(ValueError, match="non-empty"):
        prepare_forecast_examples([], event_field="")
    punctuation = [
        conversation("p", ["!", "!", "a"], [False, False, True]),
        conversation("n", ["!", "!", "b"]),
    ]
    model = PrefixEventForecaster().fit(punctuation, partition("val"))
    assert model.state.vocabulary == ()
    assert model.predict(conversation("q", ["anything", "unknown"])).probability == 0.5
    assert PrefixEventForecaster.from_dict(model.to_dict()).to_dict() == model.to_dict()
    with pytest.raises(ValueError, match="eligible"):
        PrefixEventForecaster().fit([], [])
    with pytest.raises(ValueError, match="both classes"):
        PrefixEventForecaster().fit([partition("train")[0]], partition("val"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": False},
        {"alpha": "1"},
        {"alpha": 0},
        {"alpha": float("nan")},
        {"alpha": float("inf")},
        {"alpha": 10**500},
        {"max_features": True},
        {"max_features": 100001},
        {"min_turns": 0},
        {"max_tokens": -1},
        {"event_field": ""},
        {"event_field": "is_section_header"},
    ],
)
def test_invalid_forecaster_configuration(kwargs):
    with pytest.raises(ValueError):
        PrefixEventForecaster(**kwargs)


def test_inference_input_contracts_and_oov():
    model = trained()
    assert model.predict(conversation("q", ["Straße", "unseen"])).to_dict() == {
        "probability": 0.5,
        "alert": False,
        "threshold": model.state.threshold,
        "tokens": 2,
        "known_tokens": 0,
        "coverage": 0.0,
    }
    observed = conversation("q", ["RED unknown", "red"])
    assert model.predict(observed).to_dict()["coverage"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="fitted"):
        PrefixEventForecaster().predict(observed)
    with pytest.raises(TypeError, match="Conversation"):
        model.predict(None)
    with pytest.raises(TypeError, match="ForecastPrefix"):
        model.score_prefix(None)
    with pytest.raises(ValueError, match="min_turns"):
        model.predict(conversation("q", ["red"]))
    with pytest.raises(ValueError, match="unique"):
        model.predict(Conversation("q", [observed.utterances[0]] * 2))
    with pytest.raises(ValueError, match="budget"):
        model.score_prefix(ForecastPrefix("q", "u", 2, BASE, {"red": 2000001}))
    with pytest.raises(ValueError, match="counts"):
        ForecastPrefix("q", "u", 2, BASE, {"red": -1})
    with pytest.raises(ValueError, match="eligible"):
        model.evaluate([])


def example(identifier, label, turns=2, lead=1):
    return ForecastExample(
        ForecastPrefix(identifier, str(turns), turns, BASE, {}), label, lead if label else None
    )


def test_independent_weighted_auc_and_conversation_metrics_oracle():
    rows = [
        example("p", True, 2, 3),
        example("p", True, 3, 2),
        example("n", False),
        example("m", False),
    ]
    scores = [0.2, 0.8, 0.4, 0.8]
    weights = [0.5, 0.5, 1, 1]
    # Exhaustive positive-negative pairs, weighted and with half credit on ties.
    numerator = sum(
        weights[i]
        * weights[j]
        * (float(scores[i] > scores[j]) + 0.5 * float(scores[i] == scores[j]))
        for i in (0, 1)
        for j in (2, 3)
    )
    assert weighted_auc([row.label for row in rows], scores, weights) == pytest.approx(
        numerator / 2
    )
    result = forecast_metrics(rows, scores, 0.5)
    assert result["conversation_weighted_prefix_brier"] == pytest.approx(
        (0.5 * 0.8**2 + 0.5 * 0.2**2 + 0.4**2 + 0.8**2) / 3
    )
    assert result["any_alert"]["tp"] == result["any_alert"]["fp"] == result["any_alert"]["tn"] == 1
    assert result["mean_lead_turns_for_true_positives"] == 2
    assert forecast_metrics([rows[0]], [0.2], 0.5)["conversation_weighted_prefix_roc_auc"] is None
    assert forecast_metrics([rows[0]], [0.2], 0.5)["any_alert"]["precision"] is None


def test_threshold_sweep_matches_exhaustive_hand_candidates():
    rows = [example("p", True), example("n", False), example("m", False), example("p", True, 3)]
    scores = [0.1, 0.4, 0.8, 0.9]
    # Conversation maxima(.9,.4,.8) separate perfectly only at .9.
    assert choose_threshold(rows, scores) == (0.9, 1.0)
    assert choose_threshold([example("p", True), example("n", False)], [0.5, 0.5]) == (1.0, 0.5)
    with pytest.raises(ValueError, match="inconsistent"):
        forecast_metrics([example("same", True), example("same", False)], [0.1, 0.2], 0.5)


def test_threshold_math_ties_use_exact_integer_objective():
    rows = [example(str(i), i in {1, 5}) for i in range(8)]
    scores = [(i + 1) / 9 for i in range(8)]
    threshold, score = choose_threshold(rows, scores)
    assert threshold == 6 / 9
    assert score == float(Fraction(7, 12))


def test_rechecksummed_numerical_bounds_reject_unsafe_or_impossible_models():
    artifact = trained().to_dict()
    artifact["state"]["log_probabilities"] = [[0, -1e308], [-1e308, 0]]
    with pytest.raises(ValueError, match="numeric bounds"):
        PrefixEventForecaster.from_dict(resign(artifact))
    artifact = trained().to_dict()
    artifact["state"]["positive_prior"] = 1e-12
    with pytest.raises(ValueError, match="class prior"):
        PrefixEventForecaster.from_dict(resign(artifact))


def test_observation_mask_is_consistent_but_event_labels_are_not_features():
    model = trained()
    source = conversation(
        "q", ["blue blue blue", "red", "red", "attack"], [False, False, False, True]
    )
    source.utterances[0].metadata["is_section_header"] = True
    prepared = model.prepare([source]).examples[0].prefix
    observed = Conversation("q", source.utterances[:3])
    assert model.predict(observed) == model.score_prefix(prepared)
    source.utterances[1].metadata["event"] = True
    assert model.predict(observed).probability == pytest.approx(0.9)
    with pytest.raises(ValueError, match="must differ"):
        prepare_forecast_examples([source], event_field="event", skip_field="event")
    source.utterances[0].metadata["is_section_header"] = "yes"
    with pytest.raises(ValueError, match="mask must be boolean"):
        model.predict(observed)
    with pytest.raises(ValueError, match="nondecreasing"):
        model.predict(conversation("x", ["a", "b"], times=[2, 1]))


def test_strict_artifact_roundtrip_and_atomic_save(tmp_path, monkeypatch):
    model = trained()
    path = tmp_path / "model.json"
    model.save(path)
    restored = PrefixEventForecaster.load(path)
    assert restored.to_dict() == model.to_dict()
    assert restored.evaluate(partition("test")) == model.evaluate(partition("test"))
    from turnscope import forecast_artifact

    monkeypatch.setattr(forecast_artifact, "MAX_FORECAST_BYTES", 2)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="32 MiB"):
        model.save(path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="32 MiB"):
        PrefixEventForecaster.load(path)
    with pytest.raises(ValueError, match="cannot load"):
        PrefixEventForecaster.load(tmp_path / "missing.json")
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="cannot load"):
        PrefixEventForecaster.load(path)


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "format", "v999"),
        (None, "extra", 1),
        (None, "sha256", "bad"),
        ("config", "extra", 1),
        ("state", "vocabulary", ["RED"]),
        ("state", "vocabulary", ["x", "x"]),
        ("state", "log_probabilities", [[0]]),
        ("state", "log_probabilities", [[0, 0], [0, 0]]),
        ("state", "log_probabilities", [[True, 0], [0, 0]]),
        ("state", "log_probabilities", [[float("nan"), 0], [0, 0]]),
        ("state", "log_probabilities", [[10**500, 0], [0, 0]]),
        ("state", "positive_prior", 0),
        ("state", "positive_prior", 0.3),
        ("state", "positive_prior", 1e308),
        ("state", "training_prefixes", False),
        ("state", "validation_conversations", 1),
        ("state", "threshold", -0.1),
        ("state", "validation_balanced_accuracy", 2),
        ("state", "training_groups", ["not-a-hash"]),
        ("state", "validation_groups", []),
        ("state", "extra", 1),
    ],
)
def test_corrupt_artifact_fields(section, key, value):
    artifact = trained().to_dict()
    (artifact if section is None else artifact[section])[key] = value
    with pytest.raises(ValueError):
        PrefixEventForecaster.from_dict(artifact)


def test_rechecksummed_bad_state_and_malformed_json(tmp_path):
    artifact = trained().to_dict()
    artifact["state"]["validation_groups"] = artifact["state"]["training_groups"]
    with pytest.raises(ValueError, match="overlap"):
        PrefixEventForecaster.from_dict(resign(artifact))
    path = tmp_path / "bad.json"
    for raw in ('{"x":1,"x":2}', "NaN", "{"):
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError):
            PrefixEventForecaster.load(path)


def test_cli_full_workflow_and_path_protection(tmp_path, capsys):
    paths = {
        name: tmp_path / (name + ".json") for name in ("train", "val", "test", "model", "output")
    }
    for name in ("train", "val", "test"):
        paths[name].write_text(
            json.dumps([conversation_to_dict(item) for item in partition(name)]), encoding="utf-8"
        )
    assert (
        main(["forecast", "fit", str(paths["train"]), str(paths["val"]), str(paths["model"])]) == 0
    )
    assert json.loads(capsys.readouterr().out)["validation_balanced_accuracy"] == 1
    assert (
        main(
            [
                "forecast",
                "evaluate",
                str(paths["model"]),
                str(paths["test"]),
                "-o",
                str(paths["output"]),
            ]
        )
        == 0
    )
    assert json.loads(paths["output"].read_text())["model"]["any_alert"]["accuracy"] == 1
    assert main(["forecast", "predict", str(paths["model"]), str(paths["test"])]) == 0
    assert len(json.loads(capsys.readouterr().out)["predictions"]) == 2
    before = {name: paths[name].read_bytes() for name in ("train", "val", "model")}
    for args in (
        ["fit", str(paths["train"]), str(paths["train"]), str(paths["model"])],
        ["fit", str(paths["train"]), str(paths["val"]), str(paths["train"])],
        ["predict", str(paths["model"]), str(paths["test"]), "-o", str(paths["model"])],
    ):
        assert main(["forecast", *args]) == 2
        assert "differ" in capsys.readouterr().err
    assert before == {name: paths[name].read_bytes() for name in before}
    paths["test"].write_text(json.dumps([conversation_to_dict(partition("test")[0])] * 2))
    paths["output"].write_text("existing")
    assert (
        main(
            [
                "forecast",
                "predict",
                str(paths["model"]),
                str(paths["test"]),
                "-o",
                str(paths["output"]),
            ]
        )
        == 2
    )
    assert paths["output"].read_text() == "existing"
