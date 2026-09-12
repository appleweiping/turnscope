"""Original small CPU numerical/training oracles, not CGA quality evidence."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from turnscope import neural_ablation_train as api
from turnscope.models import Conversation, Utterance
from turnscope.neural_ablation_math import (
    ABLATION_VARIANTS,
    AblationPoolingLimits,
    ablation_parameter_shapes,
    infer_ablation_turns,
)
from turnscope.neural_forecast_data import SequenceLimits, prepare_sequence_forecasts
from turnscope.neural_forecast_math import NeuralArchitecture, NeuralNumericLimits
from turnscope.neural_forecast_train import (
    NeuralTrainingConfig,
    _EncodedConversation,
    make_torch_hierarchy,
)
from turnscope.neural_token_data import encode_observed_prefix, fit_sequence_vocabulary


def sources(partition, count=5):
    patterns = [
        (True, ["red red bright", "red conflict", "long exchange here", "unobserved"]),
        (False, ["blue calm", "one calm answer", "unobserved"]),
        (True, ["red", "please explain red", "unobserved"]),
        (False, ["blue quiet", "", "response calm", "unobserved"]),
        (True, ["red conflict grows", "loud red", "unobserved"]),
    ]
    return [
        Conversation(
            f"{partition}-{index}",
            [
                Utterance(
                    str(position),
                    "person",
                    text,
                    datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=position),
                    metadata={"event": label and position == len(texts) - 1},
                )
                for position, text in enumerate(texts)
            ],
            metadata={"forecast_groups": [f"group-{partition}-{index}"]},
        )
        for index, (label, texts) in enumerate(patterns[:count])
    ]


def inputs(count=5):
    train = prepare_sequence_forecasts(sources("train", count))
    val = prepare_sequence_forecasts(sources("val", 3))
    vocabulary = fit_sequence_vocabulary(train)
    return train, val, vocabulary, NeuralArchitecture(vocabulary.size)


@pytest.fixture(scope="module")
def torch_module():
    torch = pytest.importorskip("torch", reason="optional local CPU training dependency")
    previous = torch.get_num_threads()
    # Bounded authored tests only; public runtime never changes this setting.
    torch.set_num_threads(1)
    try:
        yield torch
    finally:
        torch.set_num_threads(previous)


def hashes(module):
    return {
        name: hashlib.sha256(value.detach().numpy().tobytes()).hexdigest()
        for name, value in module.state_dict().items()
    }


def serial_logits(torch, values, turns, variant):
    """Independent unpadded reset-after recurrence, no production forward helpers."""

    def step(x, h, prefix):
        a = values[prefix + ".weight_ih_l0"] @ x + values[prefix + ".bias_ih_l0"]
        b = values[prefix + ".weight_hh_l0"] @ h + values[prefix + ".bias_hh_l0"]
        ar, az, an = a.chunk(3)
        br, bz, bn = b.chunk(3)
        r, z = torch.sigmoid(ar + br), torch.sigmoid(az + bz)
        return (1 - z) * torch.tanh(an + r * bn) + z * h

    vectors = []
    for turn in turns:
        sequence = values["embedding.weight"][list(turn)]
        if variant == "current-turn.v1":
            directions = []
            for backward in (False, True):
                h = torch.zeros(64, dtype=sequence.dtype)
                suffix = "_reverse" if backward else ""
                for x in reversed(sequence.unbind()) if backward else sequence:
                    a = (
                        values["word.weight_ih_l0" + suffix] @ x
                        + values["word.bias_ih_l0" + suffix]
                    )
                    b = (
                        values["word.weight_hh_l0" + suffix] @ h
                        + values["word.bias_hh_l0" + suffix]
                    )
                    ar, az, an = a.chunk(3)
                    br, bz, bn = b.chunk(3)
                    r, z = torch.sigmoid(ar + br), torch.sigmoid(az + bz)
                    h = (1 - z) * torch.tanh(an + r * bn) + z * h
                directions.append(h)
            vectors.append(torch.cat(directions))
        else:
            # Sum real EOS-inclusive rows, not embedding_bag or a masked batch.
            mean = sum(sequence.unbind()) / len(turn)
            vectors.append(torch.cat((mean, mean)) if variant == "mean-word.v1" else mean)
    h = torch.zeros(64, dtype=vectors[0].dtype)
    outputs = []
    for position, vector in enumerate(vectors):
        if variant == "current-turn.v1":
            a = values["turn.weight_ih_l0"] @ vector + values["turn.bias_ih_l0"]
            ar, az, an = a.chunk(3)
            br, bz, bn = values["turn.bias_hh_l0"].chunk(3)
            h = (1 - torch.sigmoid(az + bz)) * torch.tanh(an + torch.sigmoid(ar + br) * bn)
        elif variant == "mean-word.v1":
            h = step(vector, h, "turn")
        else:
            h = sum(vectors[: position + 1]) / (position + 1)
        hidden = torch.tanh(values["head.weight"] @ h + values["head.bias"])
        outputs.append((values["output.weight"] @ hidden + values["output.bias"]).squeeze(0))
    return torch.stack(outputs)


def serial_loss(torch, values, batch, variant):
    rows = []
    for record in batch:
        output = serial_logits(torch, values, record.turns, variant)
        selected = output[list(record.endpoints)]
        # Independent stable BCE expression and explicit per-conversation mean.
        rows.append(
            (torch.nn.functional.softplus(selected) - float(record.label) * selected).mean()
        )
    return torch.stack(rows).mean()


def authored_records():
    return (
        _EncodedConversation(((3, 4, 3, 2), (2,), (5, 3, 2)), (1, 2), True),
        _EncodedConversation(((4, 2), (3, 5, 4, 4, 2)), (1,), False),
        _EncodedConversation(((5, 2), (3, 3, 2), (4, 2), (5, 5, 2)), (1, 3), True),
    )


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
@pytest.mark.parametrize("seed", [17, 101, 202])
def test_exact_main_initial_subset_and_absent_inactive_inventory(torch_module, variant, seed):
    architecture = NeuralArchitecture(7)
    main = make_torch_hierarchy(torch_module, architecture, seed=seed)
    module, original = api.make_torch_ablation(torch_module, variant, architecture, seed=seed)
    expected = ablation_parameter_shapes(variant, architecture)
    assert dict(original) == hashes(main)
    assert list(hashes(module)) == list(expected)
    assert hashes(module) == {name: hashes(main)[name] for name in expected}
    assert all(parameter.requires_grad for parameter in module.parameters())
    assert (
        len(list(module.parameters()))
        == {"current-turn.v1": 16, "mean-word.v1": 9, "order-erased.v1": 5}[variant]
    )
    assert torch_module.count_nonzero(module.embedding.weight[0]) == 0


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_serial_no_padding_forward_and_all_active_gradients(torch_module, variant):
    torch = torch_module
    module, _ = api.make_torch_ablation(torch, variant, NeuralArchitecture(7), seed=17)
    values = {
        name: value.detach().clone().requires_grad_() for name, value in module.named_parameters()
    }
    batch = authored_records()
    actual = api.forward_ablation_batch(torch, module, batch, variant=variant)
    expected = tuple(serial_logits(torch, values, item.turns, variant) for item in batch)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
    actual_loss = api._batch_loss(torch, module, batch, variant)
    reference_loss = serial_loss(torch, values, batch, variant)
    torch.testing.assert_close(actual_loss, reference_loss, atol=1e-7, rtol=1e-6)
    actual_loss.backward()
    reference_loss.backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and values[name].grad is not None
        assert bool((parameter.grad != 0).any()), name
        torch.testing.assert_close(
            parameter.grad, values[name].grad, atol=2e-8, rtol=2e-4, msg=name
        )
    assert torch.count_nonzero(module.embedding.weight.grad[0]) == 0


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_actual_partial_batches_all_active_updates_and_numpy_selected_checkpoint(
    torch_module, variant
):
    train, val, vocabulary, architecture = inputs()
    before = (train.digest, val.digest, vocabulary.digest)
    config = NeuralTrainingConfig(
        epochs=2, patience=2, batch_conversations=2, learning_rate=0.01, seed=17
    )
    result = api.train_ablation_model(
        train, val, vocabulary, architecture, variant=variant, config=config
    )
    assert result.training_conversations == 5 and result.validation_conversations == 3
    assert result.training_prefixes == 7 and result.validation_prefixes == 4
    assert len(result.history) == 2 and all(row.optimizer_steps == 3 for row in result.history)
    assert set(result.changed_parameter_names) == set(
        ablation_parameter_shapes(variant, architecture)
    )
    assert result.initialization_workspace_bytes == 12 * architecture.parameter_count
    assert result.maximum_estimated_workspace_bytes >= 48 * result.parameters.parameter_count
    assert result.initialization_version == "canonical-main-subset-init.v1"
    assert "unobserved" not in vocabulary.tokens
    assert result.training_limits.to_dict() == {"max_total_pooling_operations": 100_000_000_000}
    module, _ = api.make_torch_ablation(torch_module, variant, architecture, seed=101)
    module.load_state_dict(
        {
            name: torch_module.tensor(value.copy())
            for name, value in result.parameters.arrays().items()
        },
        strict=True,
    )
    turns = encode_observed_prefix(train.observations[0], vocabulary).turns
    with torch_module.no_grad():
        serial = serial_logits(torch_module, dict(module.named_parameters()), turns, variant)
    inferred = infer_ablation_turns(result.parameters, turns)
    assert inferred.logits == pytest.approx(serial.tolist(), rel=2e-5, abs=2e-7)
    assert infer_ablation_turns(result.parameters, turns[:2]).logits == pytest.approx(
        inferred.logits[:2], abs=1e-12
    )
    assert not result.parameters.arrays()["embedding.weight"].flags.writeable
    assert before == (train.digest, val.digest, vocabulary.digest)


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_serial_two_step_clipped_adam_matches_public_training(torch_module, variant):
    torch = torch_module
    train, val, vocabulary, architecture = inputs(count=3)
    config = NeuralTrainingConfig(
        epochs=1, batch_conversations=2, seed=101, learning_rate=0.001, gradient_clip=0.03
    )
    module, _ = api.make_torch_ablation(torch, variant, architecture, seed=config.seed)
    params = {
        name: value.detach().clone().requires_grad_() for name, value in module.named_parameters()
    }
    records, _ = api._encode_partition(
        train,
        vocabulary,
        architecture,
        variant,
        NeuralNumericLimits(),
        SequenceLimits(),
        AblationPoolingLimits(),
    )
    order = list(range(3))
    random.Random(config.seed).shuffle(order)
    # Independent explicit 2+1 schedule, not the production batcher.
    batches = [tuple(records[index] for index in order[:2]), (records[order[2]],)]
    first = {name: torch.zeros_like(value) for name, value in params.items()}
    second = {name: torch.zeros_like(value) for name, value in params.items()}
    clipped = False
    for step, batch in enumerate(batches, start=1):
        for value in params.values():
            value.grad = None
        serial_loss(torch, params, batch, variant).backward()
        norm = torch.stack([value.grad.norm(2) for value in params.values()]).norm(2)
        factor = min(1.0, config.gradient_clip / (float(norm) + 1e-6))
        clipped |= factor < 1
        with torch.no_grad():
            for name, value in params.items():
                gradient = value.grad * factor
                first[name] = 0.9 * first[name] + 0.1 * gradient
                second[name] = 0.999 * second[name] + 0.001 * gradient.square()
                value -= (
                    config.learning_rate
                    * (first[name] / (1 - 0.9**step))
                    / ((second[name] / (1 - 0.999**step)).sqrt() + 1e-8)
                )
    result = api.train_ablation_model(
        train, val, vocabulary, architecture, variant=variant, config=config
    )
    assert clipped and result.history[0].optimizer_steps == 2
    for name, array in result.parameters.arrays().items():
        # Float32 packed/serial reductions and Adam operation order are not bitwise identical.
        torch.testing.assert_close(
            torch.tensor(array.copy()), params[name], atol=3e-6, rtol=2e-4, msg=name
        )


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_training_preserves_global_rng_dtype_threads_and_caller_state(torch_module, variant):
    import numpy as np

    torch = torch_module
    args = inputs(count=3)
    before = tuple(value.digest for value in args[:3])
    python_rng, numpy_rng, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    dtype, threads = torch.get_default_dtype(), torch.get_num_threads()
    deterministic = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    result = api.train_ablation_model(
        *args, variant=variant, config=NeuralTrainingConfig(epochs=1, seed=202)
    )
    assert result.torch_threads == threads
    assert random.getstate() == python_rng
    assert np.random.get_state()[0] == numpy_rng[0]
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert np.random.get_state()[2:] == numpy_rng[2:]
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert (torch.get_default_dtype(), torch.get_num_threads()) == (dtype, threads)
    assert (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    ) == deterministic
    assert tuple(value.digest for value in args[:3]) == before


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_causal_batch_outputs_and_required_invariances(torch_module, variant):
    torch = torch_module
    module, _ = api.make_torch_ablation(torch, variant, NeuralArchitecture(7), seed=17)
    initial = ((3, 4, 2), (5, 2), (4, 3, 5, 2))
    future_changed = (*initial[:2], (6, 6, 2))
    swapped = (initial[1], initial[0], initial[2])
    token_swapped = ((4, 3, 2), initial[1], (5, 3, 4, 2))
    rows = tuple(
        _EncodedConversation(turns, (1, 2), True)
        for turns in (initial, future_changed, swapped, token_swapped)
    )
    with torch.no_grad():
        output = api.forward_ablation_batch(torch, module, rows, variant=variant)
    torch.testing.assert_close(output[0][:2], output[1][:2], atol=1e-7, rtol=1e-6)
    if variant in ("current-turn.v1", "order-erased.v1"):
        torch.testing.assert_close(output[0][-1], output[2][-1], atol=1e-7, rtol=1e-6)
    else:
        assert abs(float(output[0][-1] - output[2][-1])) > 1e-7
    if variant != "current-turn.v1":
        torch.testing.assert_close(output[0], output[3], atol=1e-7, rtol=1e-6)
    assert abs(float(output[0][-1] - output[1][-1])) > 1e-7


def test_earliest_best_restored_not_last_private_weights(torch_module, monkeypatch):
    recorded, losses = [], iter([0.8, 0.6, 0.6, 0.7])

    def validation(_torch, module, *_args):
        recorded.append(hashes(module))
        return next(losses)

    monkeypatch.setattr(api, "_validation_loss", validation)
    result = api.train_ablation_model(
        *inputs(count=3),
        variant="order-erased.v1",
        config=NeuralTrainingConfig(epochs=8, patience=2, batch_conversations=2, seed=17),
    )
    final = {row.name: hashlib.sha256(row.data).hexdigest() for row in result.parameters.tensors}
    assert result.selected_epoch == 2 and len(result.history) == 4
    assert final == recorded[1] and final != recorded[-1]


@pytest.mark.parametrize(
    "options,match",
    [
        ({"config": {}}, "typed"),
        ({"numeric_limits": {}}, "typed"),
        ({"data_limits": {}}, "typed"),
        ({"pooling_limits": {}}, "typed"),
        ({"training_limits": {}}, "typed"),
        ({"numeric_limits": NeuralNumericLimits(max_parameters=1)}, "initializer"),
        ({"numeric_limits": NeuralNumericLimits(max_affine_multiplications=1)}, "affine"),
        ({"numeric_limits": NeuralNumericLimits(max_token_positions=1)}, "token_positions"),
        ({"data_limits": SequenceLimits(max_source_turns=1)}, "source inventory"),
        ({"pooling_limits": AblationPoolingLimits(max_operations=1)}, "pooling"),
        (
            {"training_limits": api.AblationTrainingLimits(max_total_pooling_operations=1)},
            "pooling",
        ),
        ({"config": NeuralTrainingConfig(max_total_affine_multiplications=1)}, "affine"),
        ({"config": NeuralTrainingConfig(max_workspace_bytes=1)}, "workspace"),
        ({"config": NeuralTrainingConfig(max_batch_token_positions=1)}, "batch_token"),
    ],
)
def test_all_admission_precedes_torch(options, match, monkeypatch):
    monkeypatch.setattr(api, "_torch", lambda: pytest.fail("Torch imported before admission"))
    with pytest.raises(ValueError, match=match):
        api.train_ablation_model(*inputs(), variant="mean-word.v1", **options)


@pytest.mark.parametrize("bad", [True, 0, -1, 10**400, 1.0, "1"])
def test_strict_pool_training_budget(bad):
    with pytest.raises(ValueError):
        api.AblationTrainingLimits(max_total_pooling_operations=bad)


def test_training_groups_vocab_architecture_and_empty_class_preflight(monkeypatch):
    monkeypatch.setattr(api, "_torch", lambda: pytest.fail("Torch imported before data admission"))
    train, val, vocab, arch = inputs()
    with pytest.raises(ValueError, match="overlap"):
        api.train_ablation_model(train, train, vocab, arch, variant="mean-word.v1")
    for variant in (None, True, "main", "mean-word"):
        with pytest.raises(ValueError, match="variant"):
            api.train_ablation_model(train, val, vocab, arch, variant=variant)
    with pytest.raises(ValueError, match="64-wide"):
        api.train_ablation_model(
            train, val, vocab, replace(arch, embedding_dim=3), variant="mean-word.v1"
        )
    with pytest.raises(ValueError, match="vocabulary size"):
        api.train_ablation_model(
            train,
            val,
            vocab,
            replace(arch, vocabulary_size=arch.vocabulary_size + 1),
            variant="mean-word.v1",
        )
    foreign = fit_sequence_vocabulary(val)
    with pytest.raises(ValueError, match="exclusively"):
        api.train_ablation_model(
            train, val, foreign, NeuralArchitecture(foreign.size), variant="mean-word.v1"
        )
    with pytest.raises(ValueError, match="both classes"):
        api.train_ablation_model(
            train,
            prepare_sequence_forecasts(sources("one", 1)),
            vocab,
            arch,
            variant="mean-word.v1",
        )


def test_pool_and_workspace_exact_bound_then_one_less_before_torch(torch_module, monkeypatch):
    args = inputs(count=3)
    config = NeuralTrainingConfig(epochs=1, batch_conversations=2)
    result = api.train_ablation_model(*args, variant="order-erased.v1", config=config)
    totals = [0, 0, 0]
    for data, factor in ((args[0], 3), (args[1], 1)):
        for observation in data.observations:
            turns = encode_observed_prefix(observation, args[2]).turns
            # Independent closed counts, not the runtime's work object.
            units, positions = len(turns), sum(map(len, turns))
            totals[0] += factor * units * 32 * 65
            totals[1] += factor * ((positions - units) + units - 1) * 64
            totals[2] += factor * 2 * units * 64
    assert (
        result.estimated_affine_multiplications,
        result.estimated_pooling_additions,
        result.estimated_pooling_scalings,
    ) == tuple(totals)
    exact = replace(
        config,
        max_workspace_bytes=result.maximum_estimated_workspace_bytes,
        max_total_affine_multiplications=totals[0],
    )
    api.train_ablation_model(
        *args,
        variant="order-erased.v1",
        config=exact,
        training_limits=api.AblationTrainingLimits(sum(totals[1:])),
    )
    monkeypatch.setattr(api, "_torch", lambda: pytest.fail("Torch imported despite quota failure"))
    with pytest.raises(ValueError, match="pooling"):
        api.train_ablation_model(
            *args,
            variant="order-erased.v1",
            config=exact,
            training_limits=api.AblationTrainingLimits(sum(totals[1:]) - 1),
        )
    with pytest.raises(ValueError, match="workspace"):
        api.train_ablation_model(
            *args,
            variant="order-erased.v1",
            config=replace(exact, max_workspace_bytes=exact.max_workspace_bytes - 1),
        )


def test_failure_after_private_step_does_not_mutate_callers(torch_module, monkeypatch):
    args = inputs()
    before = tuple(item.to_dict() for item in args[:3])
    original, calls = api._batch_loss, 0

    def second_nan(*params):
        nonlocal calls
        calls += 1
        return (
            original(*params)
            if calls == 1
            else torch_module.tensor(float("nan"), requires_grad=True)
        )

    monkeypatch.setattr(api, "_batch_loss", second_nan)
    with pytest.raises(ValueError, match="nonfinite"):
        api.train_ablation_model(
            *args,
            variant="mean-word.v1",
            config=NeuralTrainingConfig(epochs=1, batch_conversations=2),
        )
    assert calls == 2 and tuple(item.to_dict() for item in args[:3]) == before


def test_result_strict_identity_counts_and_selection(torch_module):
    result = api.train_ablation_model(
        *inputs(count=3), variant="order-erased.v1", config=NeuralTrainingConfig(epochs=1)
    )
    for changes in (
        {"variant": "mean-word.v1"},
        {"parameters": {}},
        {"config": {}},
        {"training_limits": {}},
        {"initialization_version": "other"},
        {"initial_parameter_sha256": ()},
        {
            "initial_parameter_sha256": (
                ("wrong-name", "0" * 64),
                *result.initial_parameter_sha256[1:],
            )
        },
        {
            "initial_parameter_sha256": (
                (result.initial_parameter_sha256[0][0], "0" * 64),
                *result.initial_parameter_sha256[1:],
            )
        },
        {"main_initial_parameter_sha256": ()},
        {"changed_parameter_names": ()},
        {"training_conversations": True},
        {"training_prefixes": 0},
        {"training_partition_digest": "z" * 64},
        {"history": []},
        {"history": (replace(result.history[0], epoch=True),)},
        {"history": (replace(result.history[0], optimizer_steps=True),)},
        {"selected_epoch": True},
        {"selected_epoch": 2},
        {"torch_threads": 0},
        {"numpy_version": ""},
        {"estimated_pooling_scalings": 10**400},
        {"initialization_workspace_bytes": 1},
        {
            "training_limits": api.AblationTrainingLimits(100),
            "estimated_pooling_additions": 100,
            "estimated_pooling_scalings": 100,
        },
        {"config": replace(result.config, epochs=2)},
        {"history": (replace(result.history[0], training_loss=float("nan")),)},
    ):
        with pytest.raises(ValueError):
            replace(result, **changes)


def test_hand_conversation_loss_gradient_and_partial_validation_weight(torch_module, monkeypatch):
    torch = torch_module
    batch = authored_records()[:2]
    positive = torch.tensor([13.0, -1.0, 2.0], requires_grad=True)
    negative = torch.tensor([-17.0, 0.5], requires_grad=True)
    monkeypatch.setattr(api, "forward_ablation_batch", lambda *_, **__: (positive, negative))
    actual = api._batch_loss(torch, None, batch, "order-erased.v1")
    expected = (
        (math.log1p(math.exp(1)) + math.log1p(math.exp(-2))) / 2 + math.log1p(math.exp(0.5))
    ) / 2
    assert float(actual.detach()) == pytest.approx(expected, abs=1e-7)
    actual.backward()
    assert positive.grad.tolist() == pytest.approx(
        [0.0, (1 / (1 + math.exp(1)) - 1) / 4, (1 / (1 + math.exp(-2)) - 1) / 4]
    )
    assert negative.grad.tolist() == pytest.approx([0.0, (1 / (1 + math.exp(-0.5))) / 2])
    module, _ = api.make_torch_ablation(torch, "order-erased.v1", NeuralArchitecture(7), seed=17)
    losses = iter([1.0, 4.0])
    monkeypatch.setattr(api, "_batch_loss", lambda *_: torch.tensor(next(losses)))
    assert (
        api._validation_loss(torch, module, authored_records(), ((0, 1), (2,)), "order-erased.v1")
        == 2.0
    )
    monkeypatch.setattr(api, "_batch_loss", lambda *_: torch.tensor(float("nan")))
    with pytest.raises(ValueError, match="nonfinite"):
        api._validation_loss(torch, module, authored_records(), ((0, 1), (2,)), "order-erased.v1")


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_nondefault_dtype_is_not_changed_and_candidate_remains_float32(torch_module, variant):
    torch = torch_module
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        result = api.train_ablation_model(
            *inputs(count=3), variant=variant, config=NeuralTrainingConfig(epochs=1, seed=17)
        )
        assert torch.get_default_dtype() == torch.float64
        assert all(array.dtype.str == "<f4" for array in result.parameters.arrays().values())
    finally:
        torch.set_default_dtype(previous)


def test_excluded_group_collision_is_still_rejected_before_torch(monkeypatch):
    monkeypatch.setattr(api, "_torch", lambda: pytest.fail("Torch imported before group admission"))
    training = sources("train")
    excluded = Conversation(
        "excluded",
        [
            Utterance(
                "0",
                "person",
                "already event",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                metadata={"event": True},
            )
        ],
        metadata={"forecast_groups": ["shared-hidden-group"]},
    )
    train = prepare_sequence_forecasts([*training, excluded])
    validation = sources("val", 3)
    validation[0].metadata["forecast_groups"].append("shared-hidden-group")
    val = prepare_sequence_forecasts(validation)
    vocabulary = fit_sequence_vocabulary(train)
    with pytest.raises(ValueError, match="overlap"):
        api.train_ablation_model(
            train, val, vocabulary, NeuralArchitecture(vocabulary.size), variant="order-erased.v1"
        )


def test_initial_magnitude_rejected_before_forward_or_optimizer(torch_module, monkeypatch):
    args = inputs(count=3)
    before = tuple(value.digest for value in args[:3])
    monkeypatch.setattr(
        api,
        "forward_ablation_batch",
        lambda *_, **__: pytest.fail("forward ran with inadmissible initial parameters"),
    )
    monkeypatch.setattr(
        torch_module.optim,
        "Adam",
        lambda *_, **__: pytest.fail("optimizer allocated before initial magnitude check"),
    )
    with pytest.raises(ValueError, match="initialized parameters"):
        api.train_ablation_model(
            *args,
            variant="order-erased.v1",
            config=NeuralTrainingConfig(epochs=1),
            numeric_limits=NeuralNumericLimits(max_parameter_magnitude=1e-8),
        )
    assert before == tuple(value.digest for value in args[:3])


def test_nonfinite_optimizer_update_is_rejected(torch_module, monkeypatch):
    original = torch_module.optim.Adam.step

    def invalid_step(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        with torch_module.no_grad():
            optimizer.param_groups[0]["params"][0][1, 0] = float("nan")
        return result

    monkeypatch.setattr(torch_module.optim.Adam, "step", invalid_step)
    args = inputs(count=3)
    before = tuple(value.digest for value in args[:3])
    with pytest.raises(ValueError, match="optimizer produced nonfinite"):
        api.train_ablation_model(
            *args, variant="order-erased.v1", config=NeuralTrainingConfig(epochs=1)
        )
    assert before == tuple(value.digest for value in args[:3])
