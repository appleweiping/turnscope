"""Authored tiny training acceptance, not a real-dataset quality benchmark."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from turnscope import neural_forecast_train as training_api
from turnscope.models import Conversation, Utterance
from turnscope.neural_forecast_data import (
    SequenceLimits,
    prepare_sequence_forecasts,
)
from turnscope.neural_forecast_math import (
    NeuralArchitecture,
    NeuralNumericLimits,
    infer_encoded_turns,
    parameter_shapes,
)
from turnscope.neural_forecast_train import (
    NeuralTrainingConfig,
    forward_encoded_batch,
    make_torch_hierarchy,
    train_sequence_model,
)
from turnscope.neural_token_data import encode_observed_prefix, fit_sequence_vocabulary

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def sources(partition):
    examples = [
        (True, ["red bright", "color red", "a stronger reply", "future attack"]),
        (False, ["blue calm", "color blue", "quiet response", "censored ending"]),
        (True, ["red disagreement", "please explain", "future attack"]),
        (False, ["blue quiet", "please explain", "censored ending"]),
        (True, ["red agree", "a conflict now", "future attack"]),
    ]
    count = 5 if partition == "train" else 3
    return [
        Conversation(
            f"{partition}-{index}",
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
            metadata={"forecast_groups": [f"group-{partition}-{index}"]},
        )
        for index, (label, texts) in enumerate(examples[:count])
    ]


def inputs(*, layers=1, max_features=10000):
    train = prepare_sequence_forecasts(sources("train"))
    validation = prepare_sequence_forecasts(sources("val"))
    vocabulary = fit_sequence_vocabulary(train, max_features=max_features)
    architecture = NeuralArchitecture(
        vocabulary.size,
        embedding_dim=3,
        word_hidden=3,
        turn_hidden=4,
        word_layers=layers,
        turn_layers=layers,
    )
    return train, validation, vocabulary, architecture


@pytest.fixture(scope="module")
def torch_module():
    return pytest.importorskip("torch", reason="optional CPU training dependency is not installed")


def hashes(module):
    return {
        name: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
        for name, value in module.state_dict().items()
    }


def test_real_cpu_all_layers_update_and_final_partial_batch_is_included(torch_module):
    import numpy as np

    train, val, vocabulary, architecture = inputs(layers=2)
    before = train.digest, val.digest, vocabulary.digest
    config = NeuralTrainingConfig(
        epochs=3, patience=3, batch_conversations=2, learning_rate=0.03, seed=37
    )
    result = train_sequence_model(train, val, vocabulary, architecture, config=config)
    assert result.config == config
    assert result.training_conversations == 5 and result.validation_conversations == 3
    assert result.training_prefixes == 7 and result.validation_prefixes == 5
    assert len(result.history) == 3
    assert all(epoch.optimizer_steps == 3 for epoch in result.history)  # 2+2+1, not 2+2
    assert all(
        math.isfinite(epoch.training_loss)
        and math.isfinite(epoch.validation_loss)
        and epoch.maximum_unclipped_gradient_norm > 0
        for epoch in result.history
    )
    assert set(result.changed_parameter_names) == set(parameter_shapes(architecture))
    assert result.initial_parameter_sha256 == tuple(
        hashes(make_torch_hierarchy(torch_module, architecture, seed=config.seed)).items()
    )
    assert all(np.isfinite(values).all() for values in result.parameters.arrays().values())
    assert np.count_nonzero(result.parameters.arrays()["embedding.weight"][0]) == 0
    assert (
        result.selected_epoch == min(result.history, key=lambda epoch: epoch.validation_loss).epoch
    )
    assert result.maximum_estimated_workspace_bytes > 0
    assert result.estimated_affine_multiplications > 0
    assert result.torch_version == str(torch_module.__version__)
    assert result.numpy_version == np.__version__
    assert (train.digest, val.digest, vocabulary.digest) == before


def test_independent_per_conversation_loss_and_selected_endpoint_gradients(
    torch_module, monkeypatch
):
    positive = training_api._EncodedConversation(((3, 2), (4, 2), (3, 2)), (1, 2), True)
    negative = training_api._EncodedConversation(((4, 2), (3, 2)), (1,), False)
    a = torch_module.tensor([13.0, -1.0, 2.0], requires_grad=True)
    b = torch_module.tensor([-17.0, 0.5], requires_grad=True)
    monkeypatch.setattr(training_api, "forward_encoded_batch", lambda *_: (a, b))
    loss = training_api._batch_loss(torch_module, None, (positive, negative))
    # Equal conversation weight, not one third per eligible prefix. Unselected
    # first-turn logits 13 and -17 have no supervised loss or gradient.
    expected = (
        (math.log1p(math.exp(1)) + math.log1p(math.exp(-2))) / 2 + math.log1p(math.exp(0.5))
    ) / 2
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-7)
    loss.backward()
    assert a.grad.tolist() == pytest.approx(
        [0.0, (1 / (1 + math.exp(1)) - 1) / 4, (1 / (1 + math.exp(-2)) - 1) / 4]
    )
    assert b.grad.tolist() == pytest.approx([0.0, (1 / (1 + math.exp(-0.5))) / 2])


def test_early_stopping_restores_earliest_best_not_last_epoch(torch_module, monkeypatch):
    train, val, vocabulary, architecture = inputs()
    recorded = []
    losses = iter([0.8, 0.6, 0.6, 0.7])

    def controlled_validation(_torch, module, *_args):
        recorded.append(hashes(module))
        return next(losses)

    monkeypatch.setattr(training_api, "_validation_loss", controlled_validation)
    config = NeuralTrainingConfig(
        epochs=8, patience=2, batch_conversations=2, learning_rate=0.02, seed=11
    )
    result = train_sequence_model(train, val, vocabulary, architecture, config=config)
    assert result.selected_epoch == 2 and len(result.history) == 4
    assert [epoch.validation_loss for epoch in result.history] == [0.8, 0.6, 0.6, 0.7]
    selected = {
        tensor.name: hashlib.sha256(tensor.data).hexdigest() for tensor in result.parameters.tensors
    }
    assert selected == recorded[1]
    assert selected != recorded[-1]


def test_selected_torch_checkpoint_matches_frozen_numpy_and_causal_prefix(torch_module):
    train, val, vocabulary, architecture = inputs(layers=2)
    result = train_sequence_model(
        train,
        val,
        vocabulary,
        architecture,
        config=NeuralTrainingConfig(epochs=1, batch_conversations=2, seed=3),
    )
    module = make_torch_hierarchy(torch_module, architecture, seed=999)
    module.load_state_dict(
        {
            name: torch_module.tensor(array.copy())
            for name, array in result.parameters.arrays().items()
        },
        strict=True,
    )
    prefix = train.observations[0]
    encoded = encode_observed_prefix(prefix, vocabulary).turns
    record = training_api._EncodedConversation(encoded, (len(encoded) - 1,), True)
    with torch_module.no_grad():
        logits = forward_encoded_batch(torch_module, module, (record,))[0].tolist()
    inferred = infer_encoded_turns(result.parameters, encoded)
    assert inferred.logits == pytest.approx(logits, rel=2e-5, abs=1e-6)
    assert infer_encoded_turns(result.parameters, encoded[:2]).logits == pytest.approx(
        inferred.logits[:2], abs=1e-12
    )


def test_token_budget_batcher_covers_each_conversation_and_validation_weights_partial(
    torch_module, monkeypatch
):
    records = tuple(
        training_api._EncodedConversation(((3, 2), (4, 2)), (1,), bool(index % 2))
        for index in range(5)
    )
    config = NeuralTrainingConfig(batch_conversations=5, max_batch_token_positions=8)
    batches = training_api._batches(records, [4, 1, 3, 0, 2], config)
    assert batches == ((4, 1), (3, 0), (2,))
    assert sorted(index for batch in batches for index in batch) == list(range(5))
    returned = iter([1.0, 3.0, 10.0])
    monkeypatch.setattr(training_api, "_batch_loss", lambda *_: torch_module.tensor(next(returned)))

    class EvaluationOnly:
        def eval(self):
            return self

    actual = training_api._validation_loss(torch_module, EvaluationOnly(), records, batches)
    assert actual == pytest.approx((2 * 1 + 2 * 3 + 1 * 10) / 5)


def test_failure_after_private_update_preserves_all_caller_inputs(torch_module, monkeypatch):
    train, val, vocabulary, architecture = inputs()
    before = train.to_dict(), val.to_dict(), vocabulary.to_dict()
    original = training_api._batch_loss
    calls = 0

    def invalid_second_batch(*args):
        nonlocal calls
        calls += 1
        return (
            original(*args) if calls == 1 else torch_module.tensor(float("nan"), requires_grad=True)
        )

    monkeypatch.setattr(training_api, "_batch_loss", invalid_second_batch)
    with pytest.raises(ValueError, match="nonfinite loss"):
        train_sequence_model(
            train,
            val,
            vocabulary,
            architecture,
            config=NeuralTrainingConfig(epochs=1, batch_conversations=2),
        )
    assert calls == 2
    assert (train.to_dict(), val.to_dict(), vocabulary.to_dict()) == before


@pytest.mark.parametrize(
    "change",
    [
        {"epochs": True},
        {"patience": 0},
        {"batch_conversations": 129},
        {"max_batch_token_positions": 0},
        {"max_workspace_bytes": -1},
        {"max_total_affine_multiplications": 10**400},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"learning_rate": 10**400},
        {"gradient_clip": False},
        {"gradient_clip": 0},
        {"seed": -1},
        {"min_document_frequency": True},
        {"max_features": 0},
    ],
)
def test_strict_training_config(change):
    with pytest.raises(ValueError):
        NeuralTrainingConfig(**change)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"config": {}}, "typed configuration"),
        ({"numeric_limits": {}}, "typed configuration"),
        ({"data_limits": {}}, "typed configuration"),
        ({"config": NeuralTrainingConfig(max_total_affine_multiplications=1)}, "affine"),
        ({"config": NeuralTrainingConfig(max_workspace_bytes=1)}, "workspace"),
        ({"config": NeuralTrainingConfig(max_batch_token_positions=1)}, "batch_token"),
        ({"numeric_limits": NeuralNumericLimits(max_parameters=1)}, "max_parameters"),
        ({"numeric_limits": NeuralNumericLimits(max_token_positions=1)}, "token_positions"),
        ({"data_limits": SequenceLimits(max_source_turns=1)}, "source inventory"),
    ],
)
def test_admission_failures_precede_torch_import(options, match, monkeypatch):
    args = inputs()

    def forbidden():
        raise AssertionError("training dependency loaded before input admission")

    monkeypatch.setattr(training_api, "_torch", forbidden)
    before = tuple(value.digest for value in args[:3])
    with pytest.raises(ValueError, match=match):
        train_sequence_model(*args, **options)
    assert tuple(value.digest for value in args[:3]) == before


def test_foreign_vocabulary_groups_and_empty_classes_rejected_before_torch(monkeypatch):
    train, val, vocabulary, architecture = inputs()

    def forbidden():
        raise AssertionError("training dependency loaded before split/vocabulary admission")

    monkeypatch.setattr(training_api, "_torch", forbidden)
    with pytest.raises(ValueError, match="overlap"):
        train_sequence_model(train, train, vocabulary, architecture)
    # Matching token inventory is insufficient: document frequencies/counts must
    # have been fitted exclusively on the training observations.
    foreign = fit_sequence_vocabulary(val)
    with pytest.raises(ValueError, match="exclusively"):
        train_sequence_model(
            train, val, foreign, replace(architecture, vocabulary_size=foreign.size)
        )
    wrong_arch = replace(architecture, vocabulary_size=vocabulary.size + 1)
    with pytest.raises(ValueError, match="vocabulary size"):
        train_sequence_model(train, val, vocabulary, wrong_arch)
    one_class = prepare_sequence_forecasts([sources("other")[0]])
    with pytest.raises(ValueError, match="both classes"):
        train_sequence_model(train, one_class, vocabulary, architecture)
    with pytest.raises(ValueError, match="typed sequence"):
        train_sequence_model([], val, vocabulary, architecture)


def test_excluded_conversation_group_still_blocks_cross_partition_training(monkeypatch):
    training_sources = sources("train")
    excluded = Conversation(
        "excluded",
        [Utterance("0", "person", "already event", BASE, metadata={"event": True})],
        metadata={"forecast_groups": ["shared-hidden-group"]},
    )
    train = prepare_sequence_forecasts([*training_sources, excluded])
    validation_sources = sources("val")
    validation_sources[0].metadata["forecast_groups"].append("shared-hidden-group")
    validation = prepare_sequence_forecasts(validation_sources)
    vocabulary = fit_sequence_vocabulary(train)
    architecture = NeuralArchitecture(vocabulary.size, 3, 3, 4)

    def forbidden():
        raise AssertionError("Torch loaded despite excluded-group overlap")

    monkeypatch.setattr(training_api, "_torch", forbidden)
    with pytest.raises(ValueError, match="overlap"):
        train_sequence_model(train, validation, vocabulary, architecture)


def test_nondefault_vocabulary_selection_requires_matching_training_settings(torch_module):
    train, val, vocabulary, architecture = inputs(max_features=3)
    config = NeuralTrainingConfig(max_features=3, epochs=1, batch_conversations=3)
    result = train_sequence_model(train, val, vocabulary, architecture, config=config)
    assert result.config.max_features == 3 and result.vocabulary_digest == vocabulary.digest
    assert result.parameters.architecture.vocabulary_size == 6
