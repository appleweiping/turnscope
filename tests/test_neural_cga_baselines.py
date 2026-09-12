"""Hand-computed comparator oracles; no published corpus or neural trainer runs."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from turnscope.models import Conversation, Utterance
from turnscope.neural_forecast_data import (
    SequenceForecastDataset,
    SequenceLimits,
    prepare_sequence_forecasts,
)
from turnscope.neural_token_data import fit_sequence_vocabulary

_PATH = Path(__file__).resolve().parents[1] / "benchmarks/neural_cga_baselines.py"
_SPEC = importlib.util.spec_from_file_location("_neural_cga_baselines_tests", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
baseline = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = baseline
_SPEC.loader.exec_module(baseline)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def conversation(identifier, label, observed, *, group=None, future="UNOBSERVED_SECRET"):
    texts = [*observed, future]
    return Conversation(
        identifier,
        [
            Utterance(
                str(index),
                "PRIVATE_SPEAKER",
                text,
                BASE + timedelta(seconds=index),
                metadata={"event": label and index == len(texts) - 1, "toxicity": 0.97},
            )
            for index, text in enumerate(texts)
        ],
        metadata={"forecast_groups": [group or f"page-{identifier}"]},
    )


def sources(prefix, *, positive=True, future="UNOBSERVED_SECRET"):
    return [
        conversation(f"{prefix}-p", positive, ["red", "", "blue"], future=future),
        conversation(f"{prefix}-n", False, ["blue", "blue"], future=future),
    ]


def dataset(prefix, **kwargs):
    return prepare_sequence_forecasts(sources(prefix, **kwargs))


def fitted():
    return baseline.fit_neural_baselines(dataset("train"), dataset("policy"))


def test_exact_conversation_weighted_counts_and_alpha_one_likelihoods():
    model = fitted()
    state = model.state
    assert state.vocabulary.tokens == ("blue", "red")
    assert state.vocabulary.document_frequencies == (2, 1)
    assert state.vocabulary.documents == 2
    assert state.vocabulary.size == 5  # PAD, UNK, EOS, blue, red
    assert state.class_token_counts == ((0.0, 2.0, 2.0, 0.0), (0.0, 2.5, 0.5, 1.0))
    # Negative has one two-turn prefix. Positive averages its two- and three-turn
    # counts, giving red 1, blue .5 and EOS 2.5; no extra conversation weight.
    assert [math.exp(x) for x in state.log_likelihoods[0]] == pytest.approx(
        [1 / 8, 3 / 8, 3 / 8, 1 / 8]
    )
    assert [math.exp(x) for x in state.log_likelihoods[1]] == pytest.approx(
        [1 / 8, 3.5 / 8, 1.5 / 8, 2 / 8]
    )
    assert state.prior == 0.5  # not 2 positive prefixes / 3 prefixes
    assert model.summary()["provenance"]["weighted_token_mass_negative_positive"] == [4, 4]


def test_hand_computed_probabilities_losses_and_independent_thresholds():
    model = fitted()
    p2, p3, negative = 49 / 67, 343 / 559, 49 / 193
    report = model.evaluate(dataset("test"))
    metrics = report["lexical_nb"]
    # Highest perfect policy threshold is the positive conversation's maximum.
    assert model.state.lexical_threshold == pytest.approx(p2)
    assert model.state.prior_threshold == 1.0
    expected_brier = (((1 - p2) ** 2 + (1 - p3) ** 2) / 2 + negative**2) / 2
    expected_loss = (-(math.log(p2) + math.log(p3)) / 2 - math.log1p(-negative)) / 2
    assert metrics["conversation_weighted_prefix_brier"] == pytest.approx(expected_brier)
    assert metrics["conversation_weighted_prefix_log_loss"] == pytest.approx(expected_loss)
    assert metrics["conversation_weighted_prefix_roc_auc"] == 1
    assert metrics["any_alert"]["balanced_accuracy"] == 1
    assert metrics["any_alert"]["tp"] == metrics["any_alert"]["tn"] == 1
    assert metrics["true_positive_first_alerts"] == 1
    assert metrics["mean_lead_turns_for_true_positives"] == 2
    assert report["prior"]["conversation_weighted_prefix_brier"] == 0.25
    assert report["prior"]["conversation_weighted_prefix_log_loss"] == pytest.approx(math.log(2))
    assert report["prior"]["any_alert"]["balanced_accuracy"] == 0.5
    assert report["prior"]["any_alert"]["fn"] == 1
    assert report["prior"]["any_alert"]["tn"] == 1


def test_threshold_equal_comparison_and_highest_tie_follow_existing_policy():
    model = fitted()
    report = model.evaluate(dataset("test"))
    assert report["lexical_nb"]["any_alert"]["tp"] == 1  # maximum == threshold alerts
    assert report["prior"]["threshold"] == 1  # constant .5: BA .5 at every candidate


def test_prior_uses_eligible_conversations_not_prefixes_or_excluded_sources():
    train = prepare_sequence_forecasts(
        [
            *sources("train"),
            conversation("second-positive", True, ["red", ""]),
            conversation("excluded-negative", False, []),
        ]
    )
    model = baseline.fit_neural_baselines(train, dataset("policy"))
    assert model.state.prior == 2 / 3
    support = model.summary()["provenance"]["training_support"]
    assert support["source_conversations"] == 4
    assert support["eligible_conversations"] == 3
    assert support["excluded_conversations"] == 1


def test_vocabulary_is_refit_exactly_with_document_frequencies_and_pinned_policy():
    train, policy = dataset("train"), dataset("policy")
    model = baseline.fit_neural_baselines(train, policy)
    expected = fit_sequence_vocabulary(
        train,
        min_document_frequency=1,
        max_features=10000,
        max_turn_tokens=128,
        long_turn_policy="head",
    )
    assert model.state.vocabulary.to_dict() == expected.to_dict()
    with pytest.raises(TypeError):
        baseline.fit_neural_baselines(train, policy, vocabulary=expected)
    with pytest.raises(TypeError):
        baseline.fit_neural_baselines(train, policy, neural_threshold=0.2)


def test_no_future_or_model_validation_words_enter_vocabulary():
    model = fitted()
    assert all("secret" not in token for token in model.state.vocabulary.tokens)
    unseen = prepare_sequence_forecasts(
        [
            conversation("q1", True, ["MODELVALONLY", ""]),
            conversation("q2", False, ["MODELVALONLY", ""]),
        ]
    )
    before = model.state.digest
    report = model.evaluate(unseen)
    assert "modelvalonly" not in model.state.vocabulary.tokens
    assert report["encoding"]["unknown_tokens"] == 2
    assert model.state.digest == before
    assert report["model_validation_used"] is False


def test_head128_plus_eos_and_boundary_vocab_document_frequency():
    long = " ".join(["red"] * 128 + ["ONLY_AFTER_HEAD"])
    train = prepare_sequence_forecasts(
        [
            conversation("train-p", True, [long, ""]),
            conversation("train-n", False, ["blue", "blue"]),
        ]
    )
    model = baseline.fit_neural_baselines(train, dataset("policy"))
    assert "only_after_head" not in model.state.vocabulary.tokens
    audit = model.summary()["provenance"]["training_encoding"]
    assert audit["raw_tokens"] == 131
    assert audit["retained_tokens"] == 130
    assert audit["eos_tokens"] == 4
    assert audit["encoded_tokens_including_eos"] == 134
    assert audit["truncated_turns"] == 1
    assert model.state.class_token_counts[1] == (0.0, 2.0, 0.0, 128.0)


def test_blank_turns_are_eos_not_pad_and_unknown_words_stay_features():
    train = prepare_sequence_forecasts(
        [conversation("train-p", True, ["", ""]), conversation("train-n", False, ["", ""])]
    )
    model = baseline.fit_neural_baselines(train, dataset("policy"))
    assert model.state.vocabulary.tokens == ()
    assert model.state.class_token_counts == ((0.0, 2.0), (0.0, 2.0))
    assert model.summary()["smoothing_categories_excluding_pad"] == 2
    assert model.evaluate(dataset("test"))["encoding"]["unknown_tokens"] == 4


def test_only_one_encoder_call_per_unique_observation_no_prefix_materialization(monkeypatch):
    calls = []
    original = baseline.encode_observed_prefix

    def encode(observed, vocabulary, *, limits):
        calls.append(observed.conversation_id)
        return original(observed, vocabulary, limits=limits)

    def no_prefix_copies(*_):
        raise AssertionError("raw prefixes must not be materialized")

    monkeypatch.setattr(baseline, "encode_observed_prefix", encode)
    monkeypatch.setattr(SequenceForecastDataset, "prefix", no_prefix_copies)
    model = fitted()
    model.evaluate(dataset("test"))
    assert len(calls) == len(set(calls)) == 6


def test_changed_future_text_keeps_counts_and_predictions_not_source_audit():
    original = fitted()
    changed = baseline.fit_neural_baselines(
        dataset("train", future="TOTALLY_DIFFERENT_FUTURE_😀"), dataset("policy")
    )
    assert original.state.vocabulary == changed.state.vocabulary
    assert original.state.class_token_counts == changed.state.class_token_counts
    assert original.state.log_likelihoods == changed.state.log_likelihoods
    assert original.state.training_digest != changed.state.training_digest
    assert (
        original.evaluate(dataset("test"))["prediction_inventory_digest"]
        == (changed.evaluate(dataset("test"))["prediction_inventory_digest"])
    )


def test_heldout_label_perturbation_changes_metrics_not_features_or_model():
    model = fitted()
    before = model.state.digest
    first = model.evaluate(dataset("test"))
    second = model.evaluate(dataset("test", positive=False))
    assert first["observation_digest"] == second["observation_digest"]
    assert first["partition_digest"] != second["partition_digest"]
    assert first["prediction_inventory_digest"] == second["prediction_inventory_digest"]
    assert second["lexical_nb"]["any_alert"]["balanced_accuracy"] is None
    assert second["lexical_nb"]["conversation_weighted_prefix_roc_auc"] is None
    assert model.state.digest == before


def test_policy_labels_select_policies_without_changing_fitted_counts():
    train = dataset("train")
    policy = prepare_sequence_forecasts(
        [
            conversation("policy-p", False, ["red", "", "blue"]),
            conversation("policy-n", True, ["blue", "blue"]),
        ]
    )
    first = fitted()
    second = baseline.fit_neural_baselines(train, policy)
    assert first.state.vocabulary == second.state.vocabulary
    assert first.state.class_token_counts == second.state.class_token_counts
    assert first.state.log_likelihoods == second.state.log_likelihoods
    assert first.state.prior == second.state.prior
    assert first.state.lexical_threshold != second.state.lexical_threshold
    assert second.state.lexical_threshold == 1.0  # inverted ordering cannot beat BA .5


@pytest.mark.parametrize("where", ["training", "policy"])
def test_excluded_group_overlap_rejected_before_vocabulary_fit(monkeypatch, where):
    train_rows, policy_rows = sources("train"), sources("policy")
    extra = conversation(
        "excluded", False, [], group="page-policy-p" if where == "training" else "page-train-p"
    )
    (train_rows if where == "training" else policy_rows).append(extra)
    train = prepare_sequence_forecasts(train_rows)
    policy = prepare_sequence_forecasts(policy_rows)
    monkeypatch.setattr(
        baseline,
        "fit_sequence_vocabulary",
        lambda *_a, **_k: pytest.fail("overlap reached vocabulary fitting"),
    )
    with pytest.raises(ValueError, match="overlap"):
        baseline.fit_neural_baselines(train, policy)


@pytest.mark.parametrize("group", ["page-train-p", "page-policy-p"])
def test_heldout_excluded_groups_reject_before_encoding(monkeypatch, group):
    model = fitted()
    data = prepare_sequence_forecasts(
        [*sources("test"), conversation("excluded", False, [], group=group)]
    )
    monkeypatch.setattr(
        baseline, "_encode", lambda *_a, **_k: pytest.fail("overlap reached feature encoding")
    )
    with pytest.raises(ValueError, match="overlap"):
        model.evaluate(data)


@pytest.mark.parametrize("partition", ["training", "policy"])
def test_single_class_fit_rejected(partition):
    train = dataset("train", positive=partition != "training")
    policy = dataset("policy", positive=partition != "policy")
    with pytest.raises(ValueError, match="both classes"):
        baseline.fit_neural_baselines(train, policy)


def test_empty_and_raw_sources_rejected():
    empty = prepare_sequence_forecasts([])
    with pytest.raises(ValueError, match="eligible"):
        fitted().evaluate(empty)
    with pytest.raises(ValueError, match="prepared"):
        baseline.fit_neural_baselines(sources("train"), dataset("policy"))
    with pytest.raises(ValueError, match="eligible"):
        baseline.fit_neural_baselines(dataset("train"), empty)


@pytest.mark.parametrize(
    "field", ["max_encoded_tokens", "max_count_updates", "max_parameter_cells"]
)
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, 10**100])
def test_strict_limits_reject_invalid_numbers(field, value):
    with pytest.raises(ValueError, match="bounded positive"):
        baseline.NeuralBaselineLimits(**{field: value})


@pytest.mark.parametrize(
    "field, message",
    [
        ("max_encoded_tokens", "encoded-token"),
        ("max_count_updates", "count-update"),
        ("max_parameter_cells", "parameter-cell"),
    ],
)
def test_work_limits_raise_not_sample_or_publish_partial_state(field, message):
    with pytest.raises(ValueError, match=message):
        baseline.fit_neural_baselines(
            dataset("train"),
            dataset("policy"),
            limits=baseline.NeuralBaselineLimits(**{field: 1}),
        )


def test_parameter_budget_before_encoding(monkeypatch):
    monkeypatch.setattr(
        baseline, "_encode", lambda *_a, **_k: pytest.fail("parameter admission is late")
    )
    with pytest.raises(ValueError, match="parameter-cell"):
        baseline.fit_neural_baselines(
            dataset("train"),
            dataset("policy"),
            limits=baseline.NeuralBaselineLimits(max_parameter_cells=15),
        )


def test_heldout_token_budget_honored_and_input_remains_unchanged():
    model = baseline.fit_neural_baselines(
        dataset("train"),
        dataset("policy"),
        limits=baseline.NeuralBaselineLimits(max_encoded_tokens=10),
    )
    heldout = prepare_sequence_forecasts([conversation("test", True, ["red " * 20, "blue"])])
    before = heldout.digest, model.state.digest
    with pytest.raises(ValueError, match="encoded-token"):
        model.evaluate(heldout)
    assert (heldout.digest, model.state.digest) == before


def test_sequence_source_bounds_and_closed_bound_types():
    with pytest.raises(ValueError, match="source inventory"):
        baseline.fit_neural_baselines(
            dataset("train"), dataset("policy"), data_limits=SequenceLimits(max_source_turns=1)
        )
    for kwargs in ({"data_limits": {}}, {"limits": {}}):
        with pytest.raises(ValueError, match="typed"):
            baseline.fit_neural_baselines(dataset("train"), dataset("policy"), **kwargs)


def test_frozen_nested_state_and_detached_reports():
    model = fitted()
    before = model.state.digest
    with pytest.raises(FrozenInstanceError):
        model.state.prior = 0.9
    with pytest.raises(TypeError):
        model.state.class_token_counts[0][0] = 5
    with pytest.raises(TypeError):
        model.state.vocabulary._indices["injected"] = 3
    report = model.summary()
    report["protocol"]["max_features"] = 1
    report["provenance"]["training_support"]["prefixes"] = 0
    assert model.state.digest == before
    assert model.summary()["protocol"]["max_features"] == 10000


def test_reports_are_aggregate_not_raw_text_ids_vocab_or_per_prefix_scores():
    model = fitted()
    text = json.dumps([model.summary(), model.evaluate(dataset("test"))])
    for private in (
        "UNOBSERVED_SECRET",
        "PRIVATE_SPEAKER",
        "train-p",
        "page-train",
        '"red"',
        '"blue"',
        "class_token_counts",
        "log_likelihoods",
    ):
        assert private not in text
    assert model.summary()["protocol"]["epoch_selection_used"] is False
    assert model.summary()["model_capacity_matched"] is False


def test_input_mutation_is_rejected_instead_of_bound_to_old_digest(monkeypatch):
    train = dataset("train")
    original = baseline.encode_observed_prefix
    old_groups = train.group_digests

    def mutate(observed, vocabulary, *, limits):
        object.__setattr__(train, "group_digests", old_groups | {"0" * 64})
        return original(observed, vocabulary, limits=limits)

    monkeypatch.setattr(baseline, "encode_observed_prefix", mutate)
    try:
        with pytest.raises(ValueError, match="changed during"):
            baseline.fit_neural_baselines(train, dataset("policy"))
    finally:
        object.__setattr__(train, "group_digests", old_groups)


def test_state_mutation_during_evaluation_does_not_publish_report(monkeypatch):
    model = fitted()
    original = baseline.encode_observed_prefix

    def mutate(observed, vocabulary, *, limits):
        object.__setattr__(model.state, "prior", 0.4)
        return original(observed, vocabulary, limits=limits)

    monkeypatch.setattr(baseline, "encode_observed_prefix", mutate)
    try:
        with pytest.raises(ValueError, match="state changed"):
            model.evaluate(dataset("test"))
    finally:
        object.__setattr__(model.state, "prior", 0.5)


def test_corrupt_pad_or_foreign_encoding_is_rejected(monkeypatch):
    model = fitted()
    original = baseline.encode_observed_prefix

    def corrupt(observed, vocabulary, *, limits):
        value = original(observed, vocabulary, limits=limits)
        object.__setattr__(value, "turns", ((0, 2),))
        return value

    monkeypatch.setattr(baseline, "encode_observed_prefix", corrupt)
    with pytest.raises(ValueError, match="PAD"):
        model.evaluate(dataset("test"))


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_log_odds_fail(value):
    with pytest.raises(ValueError, match="finite"):
        baseline._probability(value)


def test_stable_clipping_extreme_log_odds_and_state_validation():
    assert baseline._probability(-1e308) == 1e-15
    assert baseline._probability(1e308) == 1 - 1e-15
    state = fitted().state
    with pytest.raises(ValueError, match="likelihoods"):
        replace(state, log_likelihoods=((0.0,) * 4,) * 2)
    with pytest.raises(ValueError, match="finite"):
        replace(state, prior=float("nan"))


def test_deterministic_fit_and_evaluation_leave_inputs_untouched():
    train, policy, test = dataset("train"), dataset("policy"), dataset("test")
    before = train.digest, policy.digest, test.digest
    a = baseline.fit_neural_baselines(train, policy)
    b = baseline.fit_neural_baselines(train, policy)
    assert a.state.digest == b.state.digest
    assert a.summary() == b.summary()
    assert a.evaluate(test) == b.evaluate(test)
    assert (train.digest, policy.digest, test.digest) == before


def test_incremental_count_formula_matches_independent_rational_prefix_oracle():
    # This deliberately slow test oracle materializes tiny ASCII count prefixes;
    # production must not. Exact fractions independently verify prefix weights.
    train = prepare_sequence_forecasts(
        [
            conversation("p1", True, ["red blue", "red", "green green", "", "blue"]),
            conversation("p2", True, ["green", "blue red"]),
            conversation("n1", False, ["blue", "", "blue green", "red"]),
        ]
    )
    model = baseline.fit_neural_baselines(train, dataset("policy"))
    table = {word: index + 3 for index, word in enumerate(model.state.vocabulary.tokens)}
    expected = [[Fraction(0)] * (model.state.vocabulary.size - 1) for _ in range(2)]
    for conversation_index, observed in enumerate(train.observations):
        examples = [x for x in train.examples if x.conversation_index == conversation_index]
        for example in examples:
            for turn in observed.turns[: example.endpoint + 1]:
                ids = [table[word] for word in turn.text.split()] + [2]
                for token in ids:
                    expected[int(example.label)][token - 1] += Fraction(1, len(examples))
    for observed, oracle in zip(model.state.class_token_counts, expected, strict=True):
        assert observed == pytest.approx([float(value) for value in oracle], abs=1e-14)
    assert model.state.prior == 2 / 3


def test_equal_time_blocks_have_only_complete_prepared_endpoints():
    rows = sources("train")
    positive = rows[0]
    turns = list(positive.utterances)
    turns[2] = replace(turns[2], timestamp=turns[1].timestamp)
    rows[0] = replace(positive, utterances=tuple(turns))
    train = prepare_sequence_forecasts(rows)
    model = baseline.fit_neural_baselines(train, dataset("policy"))
    # The positive intermediate two-turn prefix would split a simultaneous block.
    # Only its completed three-turn observation remains, with full blue/EOS counts.
    assert [x.endpoint for x in train.examples if x.label] == [2]
    assert model.state.class_token_counts[1] == (0.0, 3.0, 1.0, 1.0)


def test_header_metadata_and_future_text_do_not_become_features():
    rows = sources("train")
    first = rows[0]
    header = Utterance(
        "header",
        "speaker",
        "HEADER_SECRET",
        BASE - timedelta(seconds=1),
        metadata={"is_section_header": True},
    )
    rows[0] = replace(first, utterances=(header, *first.utterances))
    model = baseline.fit_neural_baselines(prepare_sequence_forecasts(rows), dataset("policy"))
    assert model.state.class_token_counts == fitted().state.class_token_counts
    assert "header_secret" not in model.state.vocabulary.tokens
    assert model.summary()["provenance"]["training_support"]["header_turns"] == 1


def test_encoding_and_parameter_limits_admit_exact_boundary():
    model = baseline.fit_neural_baselines(
        dataset("train"),
        dataset("policy"),
        limits=baseline.NeuralBaselineLimits(
            max_encoded_tokens=9, max_count_updates=9, max_parameter_cells=16
        ),
    )
    audit = model.summary()["provenance"]["training_encoding"]
    assert audit["encoded_tokens_including_eos"] == 9
    assert audit["count_updates"] == 9
    assert model.evaluate(dataset("test"))["encoding"] == audit


def test_policy_reports_include_full_metrics_and_never_claim_epoch_selection():
    provenance = fitted().summary()["provenance"]
    assert provenance["policy_validation_lexical_metrics"]["any_alert"]["tp"] == 1
    assert provenance["policy_validation_prior_metrics"]["any_alert"]["fn"] == 1
    assert "conversation_weighted_prefix_log_loss" in provenance["policy_validation_prior_metrics"]


def test_actual_ten_thousand_feature_cap_retains_dropped_training_words_as_unk():
    words = [f"token{index:05d}" for index in range(10240)]
    first = [" ".join(words[offset : offset + 128]) for offset in range(0, 5120, 128)]
    second = [" ".join(words[offset : offset + 128]) for offset in range(5120, 10240, 128)]
    train = prepare_sequence_forecasts(
        [conversation("train-p", True, first), conversation("train-n", False, second)]
    )
    model = baseline.fit_neural_baselines(train, dataset("policy"))
    assert model.state.vocabulary.tokens == tuple(words[:10000])
    assert model.state.vocabulary.document_frequencies == (1,) * 10000
    assert model.state.class_token_counts[0][0] > 0  # dropped negative training words
    assert model.state.class_token_counts[1][0] == 0
    audit = model.summary()["provenance"]["training_encoding"]
    assert audit["unknown_tokens"] == 240
    assert audit["truncated_turns"] == 0
