"""Original API acceptance; zero candidates below are explicitly not training evidence."""

from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction

import pytest

from turnscope import neural_ablation as api
from turnscope.models import Conversation, Utterance
from turnscope.neural_ablation_math import (
    ABLATION_VARIANTS,
    FrozenAblationParameters,
    ablation_parameter_shapes,
    admit_ablation_turns,
)
from turnscope.neural_ablation_train import AblationTrainingResult
from turnscope.neural_forecast_data import (
    ObservedPrefix,
    ObservedTurn,
    SequencePolicy,
    prepare_sequence_forecasts,
)
from turnscope.neural_forecast_math import parameter_shapes
from turnscope.neural_forecast_train import NeuralEpoch, NeuralTrainingConfig
from turnscope.neural_token_data import encode_observed_prefix

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def partition(name):
    patterns = (
        (True, ("red bright", "red answer", "a conflict", "future attack")),
        (False, ("blue calm", "quiet answer", "censored end")),
        (True, ("red", "explain please", "future attack")),
    )
    return [
        Conversation(
            f"{name}-{index}",
            [
                Utterance(
                    str(position),
                    "person",
                    text,
                    BASE + timedelta(seconds=position),
                    metadata={"event": label and position == len(texts) - 1},
                )
                for position, text in enumerate(texts)
            ],
            metadata={"forecast_groups": [f"group:{name}:{index}"]},
        )
        for index, (label, texts) in enumerate(patterns)
    ]


def new_model(variant="order-erased.v1", **kwargs):
    return api.AblationEventForecaster(
        config=api.AblationForecastConfig(variant=variant),
        training_config=NeuralTrainingConfig(epochs=1, batch_conversations=2, seed=17),
        **kwargs,
    )


def zero_candidate(
    training,
    validation,
    vocabulary,
    architecture,
    *,
    variant,
    config,
    numeric_limits,
    data_limits,
    pooling_limits,
    training_limits,
):
    """Authored closed zero-weight candidate: tests plumbing, NEVER optimization."""
    np = pytest.importorskip("numpy")
    arrays = {
        name: np.zeros(shape, dtype="<f4")
        for name, shape in ablation_parameter_shapes(variant, architecture).items()
    }
    parameters = FrozenAblationParameters.from_arrays(
        variant, architecture, arrays, limits=numeric_limits, pooling_limits=pooling_limits
    )
    initial = tuple((row.name, hashlib.sha256(row.data).hexdigest()) for row in parameters.tensors)
    main_initial = tuple(
        (name, hashlib.sha256(np.zeros(shape, dtype="<f4").tobytes()).hexdigest())
        for name, shape in parameter_shapes(architecture).items()
    )
    totals = [0, 0, 0]
    for data, factor in ((training, 3), (validation, 1)):
        for observation in data.observations:
            work = admit_ablation_turns(
                variant,
                architecture,
                encode_observed_prefix(observation, vocabulary).turns,
                limits=numeric_limits,
                pooling_limits=pooling_limits,
            )
            for index, value in enumerate(
                (work.affine_multiplications, work.pooling_additions, work.pooling_scalings)
            ):
                totals[index] += config.epochs * factor * value
    return AblationTrainingResult(
        variant,
        parameters,
        config,
        training_limits,
        tuple(
            NeuralEpoch(
                epoch,
                math.log(2),
                math.log(2),
                math.ceil(len(training.observations) / config.batch_conversations),
                0.0,
            )
            for epoch in range(1, min(config.epochs, config.patience + 1) + 1)
        ),
        1,
        initial,
        main_initial,
        (),
        training.digest,
        validation.digest,
        vocabulary.digest,
        len(training.observations),
        len(validation.observations),
        len(training.examples),
        len(validation.examples),
        max(12 * architecture.parameter_count, 48 * parameters.parameter_count) + 1_000_000,
        12 * architecture.parameter_count,
        *totals,
        "authored-zero-fixture",
        str(np.__version__),
        1,
    )


@pytest.fixture
def zero_model(monkeypatch):
    monkeypatch.setattr(api, "train_ablation_model", zero_candidate)
    return new_model().fit(
        partition("train"), partition("validation"), policy_validation=partition("policy")
    )


@pytest.fixture(scope="module", params=ABLATION_VARIANTS)
def actual_model(request):
    torch = pytest.importorskip("torch", reason="actual CPU optimization dependency")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield new_model(request.param).fit(
            partition("actual-train"),
            partition("actual-validation"),
            policy_validation=partition("actual-policy"),
        )
    finally:
        torch.set_num_threads(previous)


def test_actual_three_mode_fit_predict_transform_and_evaluate(actual_model):
    model = actual_model
    state = model.state
    assert state.training.changed_parameter_names
    assert state.training.selected_epoch == 1
    assert (
        len(state.parameters.tensors)
        == {"current-turn.v1": 16, "mean-word.v1": 9, "order-erased.v1": 5}[state.config.variant]
    )
    assert state.training_groups.isdisjoint(
        state.validation_groups | state.policy_validation_groups
    )
    assert state.validation_groups.isdisjoint(state.policy_validation_groups)
    assert not {"future", "attack", "censored", "end"} & set(state.vocabulary.tokens)
    data = model.prepare(partition("actual-heldout"))
    predictions = model.transform(data.observations)
    assert predictions == tuple(model.predict(row) for row in data.observations)
    for observation, prediction in zip(data.observations, predictions, strict=True):
        assert prediction.variant == state.config.variant
        assert (
            prediction.input_digest == observation.digest
            and prediction.model_digest == model.digest
        )
        assert prediction.alert == (prediction.probability >= state.threshold)
        assert 0 < prediction.probability < 1 and math.isfinite(prediction.logit)
    report = model.evaluate(data)
    assert report["format"] == "turnscope.neural-ablation-report.v1"
    assert report["metrics"]["conversations"] == 3 and report["metrics"]["prefixes"] == 4
    assert report["variant"] == state.config.variant and report["model_digest"] == model.digest
    assert report["policy_reuses_model_validation"] is False
    assert report["probability_calibration_claimed"] is False
    assert report["whole_repository_parity_claimed"] is False


def test_independent_exact_threshold_and_per_conversation_metric_oracles(actual_model):
    model = actual_model
    policy = model.prepare(partition("actual-policy"))
    scores = [model.predict(policy.prefix(item)).probability for item in policy.examples]
    maxima = {}
    for example, score in zip(policy.examples, scores, strict=True):
        prior = maxima.get(example.conversation_index, (example.label, 0.0))[1]
        maxima[example.conversation_index] = example.label, max(prior, score)
    positives = sum(label for label, _ in maxima.values())
    negatives = len(maxima) - positives
    choices = []
    for threshold in {0.0, 1.0, *(score for _, score in maxima.values())}:
        tp = sum(label and score >= threshold for label, score in maxima.values())
        tn = sum(not label and score < threshold for label, score in maxima.values())
        choices.append(((Fraction(tp, positives) + Fraction(tn, negatives)) / 2, threshold))
    accuracy, threshold = max(choices)
    assert model.state.threshold == threshold
    assert model.state.policy_validation_balanced_accuracy == pytest.approx(float(accuracy))
    data = model.prepare(partition("oracle-heldout"))
    rows = {}
    leads = {}
    for example in data.examples:
        probability = model.predict(data.prefix(example)).probability
        rows.setdefault(example.conversation_index, []).append((example.label, probability))
        if example.label and probability >= model.state.threshold:
            leads.setdefault(example.conversation_index, example.lead_turns)
    brier = sum(
        sum((score - int(label)) ** 2 for label, score in values) / len(values)
        for values in rows.values()
    ) / len(rows)
    log_loss = -sum(
        sum(math.log(score) if label else math.log1p(-score) for label, score in values)
        / len(values)
        for values in rows.values()
    ) / len(rows)
    positives = [
        (score, 1 / len(values)) for values in rows.values() for label, score in values if label
    ]
    negatives = [
        (score, 1 / len(values)) for values in rows.values() for label, score in values if not label
    ]
    numerator = sum(
        a_weight * b_weight * (int(a > b) + 0.5 * int(a == b))
        for a, a_weight in positives
        for b, b_weight in negatives
    )
    auc = numerator / (
        sum(weight for _, weight in positives) * sum(weight for _, weight in negatives)
    )
    metrics = model.evaluate(data)["metrics"]
    assert metrics["conversation_weighted_prefix_brier"] == pytest.approx(brier, abs=1e-12)
    assert metrics["conversation_weighted_prefix_log_loss"] == pytest.approx(log_loss, abs=1e-12)
    assert metrics["conversation_weighted_prefix_roc_auc"] == pytest.approx(auc, abs=1e-12)
    assert metrics["true_positive_first_alerts"] == len(leads)
    assert metrics["mean_lead_turns_for_true_positives"] == (
        pytest.approx(sum(leads.values()) / len(leads)) if leads else None
    )


def test_actual_retraining_future_text_change_leaves_all_features_and_weights_identical(
    actual_model,
):
    changed = [
        replace(
            conversation,
            utterances=(
                *conversation.utterances[:-1],
                replace(conversation.utterances[-1], text="different unobservable future text"),
            ),
        )
        for conversation in partition("actual-train")
    ]
    cloned = new_model(actual_model.state.config.variant).fit(
        changed, partition("actual-validation"), policy_validation=partition("actual-policy")
    )
    assert cloned.state.vocabulary == actual_model.state.vocabulary
    assert cloned.state.parameters.digest == actual_model.state.parameters.digest
    assert cloned.state.training.history == actual_model.state.training.history
    assert (
        cloned.state.training.training_partition_digest
        != actual_model.state.training.training_partition_digest
    )
    query = actual_model.prepare(partition("future-mutation-query")).observations[0]
    assert cloned.predict(query).probability == actual_model.predict(query).probability


def test_constant_fixture_tied_policy_and_metrics_have_hand_values(zero_model):
    assert zero_model.state.threshold == 1.0
    assert zero_model.state.policy_validation_balanced_accuracy == 0.5
    metrics = zero_model.evaluate(partition("heldout"))["metrics"]
    assert metrics["conversation_weighted_prefix_brier"] == 0.25
    assert metrics["conversation_weighted_prefix_log_loss"] == pytest.approx(math.log(2))
    assert metrics["conversation_weighted_prefix_roc_auc"] == 0.5
    assert {key: metrics["any_alert"][key] for key in ("tp", "fp", "tn", "fn")} == {
        "tp": 0,
        "fp": 0,
        "tn": 1,
        "fn": 2,
    }
    assert metrics["true_positive_first_alerts"] == 0
    assert metrics["mean_lead_turns_for_true_positives"] is None


def test_single_class_heldout_metrics_preserve_undefined_denominators(zero_model):
    metrics = zero_model.evaluate(partition("negative-only")[1:2])["metrics"]
    assert metrics["positive_conversations"] == 0
    assert metrics["conversation_weighted_prefix_roc_auc"] is None
    assert metrics["any_alert"]["recall"] is None
    assert metrics["any_alert"]["balanced_accuracy"] is None
    assert metrics["any_alert"]["false_positive_rate"] == 0.0


def test_three_partitions_mandatory_and_pairwise_group_disjoint_before_training(monkeypatch):
    monkeypatch.setattr(
        api, "train_ablation_model", lambda *_, **__: pytest.fail("trained before split admission")
    )
    with pytest.raises(TypeError, match="policy_validation"):
        new_model().fit(partition("t"), partition("v"))
    for names in (("same", "same", "p"), ("same", "v", "same"), ("t", "same", "same")):
        with pytest.raises(ValueError, match="distinct groups"):
            new_model().fit(
                partition(names[0]), partition(names[1]), policy_validation=partition(names[2])
            )
    with pytest.raises(ValueError, match="both classes"):
        new_model().fit(partition("t"), partition("v"), policy_validation=partition("p")[:1])


@pytest.mark.parametrize("part", ["train", "validation", "policy"])
def test_excluded_source_groups_block_heldout_evaluation(zero_model, monkeypatch, part):
    excluded = Conversation(
        "excluded",
        [Utterance("0", "person", "already attacked", BASE, metadata={"event": True})],
        metadata={"forecast_groups": [f"group:{part}:0"]},
    )
    monkeypatch.setattr(
        api, "infer_ablation_turns", lambda *_, **__: pytest.fail("inference before group checks")
    )
    with pytest.raises(ValueError, match="overlap"):
        zero_model.evaluate([*partition("new"), excluded])


def test_excluded_source_groups_block_fit_before_optimizer(monkeypatch):
    excluded = Conversation(
        "excluded",
        [Utterance("0", "person", "already attacked", BASE, metadata={"event": True})],
        metadata={"forecast_groups": ["group:v:0"]},
    )
    monkeypatch.setattr(
        api,
        "train_ablation_model",
        lambda *_, **__: pytest.fail("trained despite excluded-group overlap"),
    )
    with pytest.raises(ValueError, match="distinct groups"):
        new_model().fit(
            [*partition("t"), excluded], partition("v"), policy_validation=partition("p")
        )


@pytest.mark.parametrize("phase", ["optimizer", "policy", "threshold"])
def test_atomic_failed_refit_preserves_exact_prior_state(zero_model, monkeypatch, phase):
    model = zero_model
    old, digest = model.state, model.digest
    query = model.prepare(partition("query")).observations[0]
    prior = model.predict(query)
    name = {
        "optimizer": "train_ablation_model",
        "policy": "infer_ablation_turns",
        "threshold": "choose_threshold",
    }[phase]
    with monkeypatch.context() as patch:

        def fail(*_, **__):
            raise ValueError("injected private candidate failure")

        patch.setattr(api, name, fail)
        with pytest.raises(ValueError, match="injected"):
            model.fit(
                partition("next-t"), partition("next-v"), policy_validation=partition("next-p")
            )
    assert model.state is old and model.digest == digest and model.predict(query) == prior


def test_cached_identity_and_summary_cannot_mutate_state(zero_model, monkeypatch):
    state, digest = zero_model.state, zero_model.digest
    summary = zero_model.training_summary
    summary["config"]["variant"] = "wrong"
    summary["history"][0]["validation_loss"] = 999
    summary["initial_parameter_sha256"].clear()
    assert zero_model.training_summary["config"]["variant"] == "order-erased.v1"
    with pytest.raises(FrozenInstanceError):
        state.threshold = 0.0
    with pytest.raises(ValueError):
        state.parameters.arrays()["embedding.weight"].flags.writeable = True

    def forbidden(_self):
        raise AssertionError("prediction rehashed all parameter bytes")

    monkeypatch.setattr(FrozenAblationParameters, "digest", property(forbidden))
    assert (
        zero_model.predict(zero_model.prepare(partition("q")).observations[0]).model_digest
        == digest
    )


def test_observation_only_features_and_future_vocabulary_exclusion(zero_model):
    query = Conversation("query", partition("x")[0].utterances[:2], {"forecast_groups": object()})
    initial = zero_model.predict_conversation(query)
    for row in query.utterances:
        row.metadata["event"] = {"not": "a label"}
        row.metadata["other"] = object()
    assert zero_model.predict_conversation(query) == initial
    with pytest.raises(ValueError, match="ObservedPrefix"):
        zero_model.predict(query)
    changed = [
        replace(
            conversation,
            utterances=(
                *conversation.utterances[:-1],
                replace(conversation.utterances[-1], text="secret changed future"),
            ),
        )
        for conversation in partition("train")
    ]
    cloned = new_model().fit(
        changed, partition("validation"), policy_validation=partition("policy")
    )
    assert cloned.state.vocabulary == zero_model.state.vocabulary
    assert cloned.state.parameters.digest == zero_model.state.parameters.digest
    assert (
        cloned.state.training.training_partition_digest
        != zero_model.state.training.training_partition_digest
    )


@pytest.mark.parametrize("cap", ["affine", "pooling"])
def test_policy_aggregate_budget_is_admitted_before_any_optimization(monkeypatch, cap):
    config = api.AblationForecastConfig(
        "order-erased.v1",
        **{
            f"max_inference_{cap}_multiplications"
            if cap == "affine"
            else "max_inference_pooling_operations": 1
        },
    )
    model = api.AblationEventForecaster(config=config)
    monkeypatch.setattr(
        api,
        "train_ablation_model",
        lambda *_, **__: pytest.fail("training before policy admission"),
    )
    monkeypatch.setattr(
        api,
        "infer_ablation_turns",
        lambda *_, **__: pytest.fail("inference before policy admission"),
    )
    with pytest.raises(ValueError, match=f"max_inference_{cap}"):
        model.fit(partition("t"), partition("v"), policy_validation=partition("p"))


@pytest.mark.parametrize("cap", ["affine", "pooling"])
def test_single_and_aggregate_work_rejected_before_first_inference(zero_model, monkeypatch, cap):
    query = zero_model.prepare(partition("q")).observations[0]
    state = zero_model.state
    encoded = encode_observed_prefix(query, state.vocabulary)
    work = admit_ablation_turns(state.config.variant, state.parameters.architecture, encoded.turns)
    field = (
        "max_inference_affine_multiplications"
        if cap == "affine"
        else "max_inference_pooling_operations"
    )
    amount = work.affine_multiplications if cap == "affine" else work.pooling_operations
    limited = new_model()
    limited._state = replace(state, config=replace(state.config, **{field: amount}))
    assert limited.predict(query).probability == zero_model.predict(query).probability
    monkeypatch.setattr(
        api,
        "infer_ablation_turns",
        lambda *_, **__: pytest.fail("numeric inference before whole admission"),
    )
    with pytest.raises(ValueError, match=field):
        limited.transform([query, query])
    limited._state = replace(state, config=replace(state.config, **{field: amount - 1}))
    with pytest.raises(ValueError, match=field):
        limited.predict(query)
    with pytest.raises(ValueError, match=field):
        limited.evaluate(partition("heldout"))


@pytest.mark.parametrize("kind", ["count", "turns", "tokens", "bytes", "late-invalid"])
def test_all_collection_admission_including_late_invalid_precedes_inference(
    zero_model, monkeypatch, kind
):
    state = zero_model.state
    query = ObservedPrefix("q", (ObservedTurn("0", BASE, "red"), ObservedTurn("1", BASE, "blue")))
    limits = state.data_limits
    queries = [query] * 4
    if kind == "count":
        limits = replace(limits, max_conversations=3)
    elif kind == "turns":
        limits = replace(limits, max_source_turns=12)
        queries = [query] * 7
    elif kind == "tokens":
        limits = replace(limits, max_tokens=10)
        queries = [query] * 6
    elif kind == "bytes":
        limits = replace(limits, max_source_bytes=200)
        queries = [query] * 21
    else:
        queries = [query, None]
    limited = new_model()
    limited._state = replace(state, data_limits=limits)
    monkeypatch.setattr(
        api,
        "infer_ablation_turns",
        lambda *_, **__: pytest.fail("partial inference before invalid final row"),
    )
    with pytest.raises(ValueError):
        limited.transform(iter(queries))


def test_empty_collection_empty_text_unknown_and_short_views(zero_model, monkeypatch):
    empty = ObservedPrefix("empty", (ObservedTurn("0", BASE, ""), ObservedTurn("1", BASE, "")))
    result = zero_model.predict(empty).to_dict()
    assert result["raw_tokens"] == result["retained_tokens"] == result["known_tokens"] == 0
    assert result["retained_token_fraction"] == 1 and result["known_retained_token_fraction"] == 0
    unknown = replace(empty, turns=tuple(replace(turn, text="unseen") for turn in empty.turns))
    assert zero_model.predict(unknown).known_tokens == 0
    with pytest.raises(ValueError, match="min_turns"):
        zero_model.predict(replace(empty, turns=empty.turns[:1]))
    with pytest.raises(ValueError, match="eligible"):
        zero_model.evaluate([])
    monkeypatch.setattr(
        api, "infer_ablation_turns", lambda *_, **__: pytest.fail("empty collection ran inference")
    )
    assert zero_model.transform([]) == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"variant": True},
        {"variant": "main"},
        {"max_turn_tokens": True},
        {"max_turn_tokens": 0},
        {"max_inference_affine_multiplications": True},
        {"max_inference_affine_multiplications": 10**400},
        {"max_inference_pooling_operations": 0},
        {"max_inference_pooling_operations": 1.5},
        {"long_turn_policy": False},
        {"long_turn_policy": "drop"},
    ],
)
def test_config_closed_finite_bounds(changes):
    with pytest.raises(ValueError):
        api.AblationForecastConfig(**{"variant": "mean-word.v1", **changes})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"config": {}},
        {"training_config": {}},
        {"policy": {}},
        {"data_limits": {}},
        {"numeric_limits": {}},
        {"pooling_limits": {}},
        {"training_limits": {}},
        {"config": api.AblationForecastConfig("mean-word.v1", max_turn_tokens=129)},
        {"policy": SequencePolicy(min_turns=65)},
    ],
)
def test_model_typed_configuration_and_cross_limits(kwargs):
    with pytest.raises(ValueError):
        api.AblationEventForecaster(
            **{"config": api.AblationForecastConfig("order-erased.v1"), **kwargs}
        )


def test_unfitted_and_prepared_supervision_boundary(zero_model):
    with pytest.raises(ValueError, match="not been fitted"):
        new_model().transform([])
    prepared = prepare_sequence_forecasts(partition("prepared"), policy=SequencePolicy(min_turns=1))
    with pytest.raises(ValueError, match="before configured min_turns"):
        zero_model.evaluate(prepared)
    # A valid prepared dataset declares its own labels. Raw metadata is not present
    # to rederive the event-field semantics inside an accepted prepared contract.
    supplied = zero_model.prepare(partition("declared"))
    assert zero_model.evaluate(supplied)["partition_digest"] == supplied.digest


def test_state_rejects_typed_count_identity_vocab_and_group_mismatches(zero_model):
    state = zero_model.state
    changes = [
        {"training": {}},
        {"vocabulary": {}},
        {"config": {}},
        {"policy": {}},
        {"data_limits": {}},
        {"config": replace(state.config, variant="mean-word.v1")},
        {"config": replace(state.config, max_turn_tokens=127)},
        {
            "vocabulary": replace(
                state.vocabulary,
                tokens=state.vocabulary.tokens[1:],
                document_frequencies=state.vocabulary.document_frequencies[1:],
            )
        },
        {
            "training": replace(
                state.training, config=replace(state.training.config, min_document_frequency=3)
            )
        },
        {"data_limits": replace(state.data_limits, max_source_turns=6)},
        {"vocabulary": replace(state.vocabulary, documents=state.vocabulary.documents + 1)},
        {
            "vocabulary": replace(
                state.vocabulary, document_frequencies=tuple(1 for _ in state.vocabulary.tokens)
            )
        },
        {"threshold": True},
        {"threshold": float("nan")},
        {"threshold": 10**400},
        {"policy_validation_balanced_accuracy": False},
        {"policy_validation_balanced_accuracy": 1.1},
        {"policy_validation_conversations": True},
        {"policy_validation_prefixes": 2},
        {"policy_validation_digest": "g" * 64},
        {"policy_validation_digest": state.training.training_partition_digest},
        {"training_groups": set(state.training_groups)},
        {"validation_groups": frozenset()},
        {"policy_validation_groups": state.training_groups},
        {
            "policy_validation_groups": frozenset(
                {"g" * 64, *sorted(state.policy_validation_groups)[1:]}
            )
        },
    ]
    for change in changes:
        with pytest.raises(ValueError):
            replace(state, **change)


@pytest.mark.parametrize(
    "field",
    [
        "training_partition_digest",
        "validation_partition_digest",
        "config",
        "training_conversations",
        "validation_conversations",
        "training_prefixes",
        "validation_prefixes",
        "vocabulary_digest",
        "numeric_limits",
        "pooling_limits",
        "training_limits",
        "variant",
        "result_type",
    ],
)
def test_fit_rejects_internally_valid_candidate_bound_to_wrong_request_before_policy_inference(
    zero_model, monkeypatch, field
):
    """A strict candidate is still not proof it belongs to this fit request."""
    original = api.train_ablation_model

    def wrong(*args, **kwargs):
        if field == "result_type":
            return {}
        if field == "variant":
            return original(*args, **{**kwargs, "variant": "mean-word.v1"})
        result = original(*args, **kwargs)
        if field == "config":
            return replace(result, config=replace(result.config, seed=result.config.seed + 1))
        if field in ("training_conversations", "validation_conversations"):
            # Keeping >=2 classes and support internally valid does not match the request.
            return replace(result, **{field: 2})
        if field in ("training_prefixes", "validation_prefixes"):
            return replace(result, **{field: 3})
        if field == "numeric_limits":
            return replace(
                result,
                parameters=replace(
                    result.parameters,
                    limits=replace(
                        result.parameters.limits,
                        max_workspace_bytes=result.parameters.limits.max_workspace_bytes - 1,
                    ),
                ),
            )
        if field == "pooling_limits":
            return replace(
                result,
                parameters=replace(
                    result.parameters,
                    pooling_limits=replace(
                        result.parameters.pooling_limits,
                        max_operations=result.parameters.pooling_limits.max_operations - 1,
                    ),
                ),
            )
        if field == "training_limits":
            return replace(
                result,
                training_limits=replace(
                    result.training_limits,
                    max_total_pooling_operations=result.training_limits.max_total_pooling_operations
                    - 1,
                ),
            )
        return replace(result, **{field: "0" * 64})

    monkeypatch.setattr(api, "train_ablation_model", wrong)
    monkeypatch.setattr(
        api,
        "infer_ablation_turns",
        lambda *_, **__: pytest.fail("mismatched candidate reached policy inference"),
    )
    previous = zero_model.state
    with pytest.raises(ValueError, match=r"candidate|training result|request|support"):
        zero_model.fit(partition("new-t"), partition("new-v"), policy_validation=partition("new-p"))
    assert zero_model.state is previous


@pytest.mark.parametrize("long_policy", ["head", "reject"])
def test_frozen_retention_policy_and_exact_token_coverage(monkeypatch, long_policy):
    monkeypatch.setattr(api, "train_ablation_model", zero_candidate)
    model = api.AblationEventForecaster(
        config=api.AblationForecastConfig(
            "order-erased.v1", max_turn_tokens=2, long_turn_policy=long_policy
        ),
        training_config=NeuralTrainingConfig(epochs=1),
    )
    model.fit(partition("t"), partition("v"), policy_validation=partition("p"))
    query = ObservedPrefix(
        "q", (ObservedTurn("0", BASE, "red bright unseen"), ObservedTurn("1", BASE, "blue missing"))
    )
    if long_policy == "reject":
        with pytest.raises(ValueError, match="token"):
            model.predict(query)
    else:
        prediction = model.predict(query)
        assert (
            prediction.raw_tokens,
            prediction.retained_tokens,
            prediction.known_tokens,
            prediction.truncated_turns,
        ) == (5, 4, 3, 1)
        assert prediction.to_dict()["retained_token_fraction"] == 0.8
        assert prediction.to_dict()["known_retained_token_fraction"] == 0.75
