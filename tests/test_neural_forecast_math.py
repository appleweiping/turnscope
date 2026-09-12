"""Scalar arithmetic oracles for the neural primitive, without a trained model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from turnscope import neural_forecast_math as numeric
from turnscope.neural_forecast_math import (
    FrozenNeuralParameters,
    FrozenNeuralTensor,
    NeuralArchitecture,
    NeuralNumericLimits,
    infer_encoded_turns,
    parameter_shapes,
)


def architecture(**changes):
    return replace(NeuralArchitecture(6, 2, 2, 4), **changes)


def arrays_for(config, *, zero=False):
    """Small deterministic nonsymmetric parameters, not model-generated fixtures."""
    arrays = {}
    for name, shape in parameter_shapes(config).items():
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
        values = [((seed + 13 * index) % 43 - 21) / 31 for index in range(math.prod(shape))]
        arrays[name] = (
            np.zeros(shape, dtype="<f4") if zero else np.array(values, dtype="<f4").reshape(shape)
        )
    return arrays


def scalar_sigmoid(value):
    return 1 / (1 + math.exp(-value))


def scalar_step(x, h, weights):
    """Independent Python loops; no production helper or NumPy matrix multiply."""
    wi, wh, bi, bh = weights
    hidden = len(h)
    a = [
        math.fsum(w * v for w, v in zip(row, x, strict=True)) + bias
        for row, bias in zip(wi, bi, strict=True)
    ]
    b = [
        math.fsum(w * v for w, v in zip(row, h, strict=True)) + bias
        for row, bias in zip(wh, bh, strict=True)
    ]
    output = []
    for j in range(hidden):
        r = scalar_sigmoid(a[j] + b[j])
        z = scalar_sigmoid(a[j + hidden] + b[j + hidden])
        candidate = math.tanh(a[j + 2 * hidden] + r * b[j + 2 * hidden])
        output.append((1 - z) * candidate + z * h[j])
    return output


def scalar_direction(sequence, arrays, level, layer, hidden, backward=False):
    suffix = f"_l{layer}" + ("_reverse" if backward else "")
    weights = [
        arrays[f"{level}.{kind}{suffix}"].tolist()
        for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
    ]
    state = [0.0] * hidden
    result = [None] * len(sequence)
    for index in reversed(range(len(sequence))) if backward else range(len(sequence)):
        state = scalar_step(sequence[index], state, weights)
        result[index] = state
    return result, state


def scalar_hierarchy(config, arrays, turns):
    vectors = []
    for turn in turns:
        sequence = [arrays["embedding.weight"][token].tolist() for token in turn]
        for layer in range(config.word_layers):
            forward, last_f = scalar_direction(sequence, arrays, "word", layer, config.word_hidden)
            reverse, last_r = scalar_direction(
                sequence, arrays, "word", layer, config.word_hidden, True
            )
            sequence = [a + b for a, b in zip(forward, reverse, strict=True)]
        vectors.append(last_f + last_r)
    states = vectors
    for layer in range(config.turn_layers):
        states, _ = scalar_direction(states, arrays, "turn", layer, config.turn_hidden)
    logits = []
    for state in states:
        hidden = [
            math.tanh(
                math.fsum(float(w) * h for w, h in zip(row, state, strict=True)) + float(bias)
            )
            for row, bias in zip(arrays["head.weight"], arrays["head.bias"], strict=True)
        ]
        logits.append(
            math.fsum(float(w) * h for w, h in zip(arrays["output.weight"][0], hidden, strict=True))
            + float(arrays["output.bias"][0])
        )
    return vectors, states, logits


def test_shape_inventory_and_independent_parameter_count():
    config = architecture()
    assert config.parameter_count == 12 + 72 + 120 + 13 == 217
    assert config.head_hidden == 2
    assert len(parameter_shapes(config)) == 17
    double = architecture(word_layers=2, turn_layers=2)
    shapes = parameter_shapes(double)
    assert len(shapes) == 29
    assert shapes["word.weight_ih_l1_reverse"] == (6, 4)
    assert shapes["turn.weight_ih_l1"] == (12, 4)
    assert config.to_dict()["vocabulary_size"] == 6
    with pytest.raises(FrozenInstanceError):
        config.word_hidden = 3
    with pytest.raises(ValueError, match="architecture"):
        parameter_shapes(config.to_dict())


@pytest.mark.parametrize(
    "name,value",
    [
        ("vocabulary_size", 2),
        ("embedding_dim", True),
        ("word_hidden", 257),
        ("turn_hidden", 1),
        ("word_layers", 0),
        ("turn_layers", 3),
    ],
)
def test_architecture_rejects_invalid_dimensions(name, value):
    with pytest.raises(ValueError, match=name):
        architecture(**{name: value})


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_parameters", True),
        ("max_observed_turns", 257),
        ("max_turn_tokens", 0),
        ("max_token_positions", 262401),
        ("max_affine_multiplications", 10**100),
        ("max_workspace_bytes", 0),
        ("max_parameter_magnitude", float("nan")),
        ("max_parameter_magnitude", float("inf")),
        ("max_parameter_magnitude", False),
        ("max_parameter_magnitude", -1),
    ],
)
def test_limits_validate_finite_closed_hard_caps(name, value):
    with pytest.raises(ValueError, match=name):
        NeuralNumericLimits(**{name: value})


def test_gru_half_gates_and_gate_order_hand_oracle():
    x = np.array([0.0])
    previous = np.array([0.25])
    # r=z=1/2; n=tanh(0.4+0.5*0.8), h=0.5*n+0.125.
    weights = (
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        np.array([0.0, 0.0, 0.4]),
        np.array([0.0, 0.0, 0.8]),
    )
    actual = numeric._gru_step(np, x, previous, weights)
    assert actual[0] == pytest.approx(0.5 * math.tanh(0.8) + 0.125, abs=1e-15)
    # Distinct reset and update logits makes a swapped gate order observably wrong.
    other = (weights[0], weights[1], np.array([math.log(3), -math.log(3), 0.4]), weights[3])
    actual = numeric._gru_step(np, x, previous, other)[0]
    correct = 0.75 * math.tanh(0.4 + 0.75 * 0.8) + 0.25 * 0.25
    swapped = 0.25 * math.tanh(0.4 + 0.25 * 0.8) + 0.75 * 0.25
    assert actual == pytest.approx(correct, abs=1e-15)
    assert abs(actual - swapped) > 0.1


def test_reset_after_is_not_reset_before_nondiagonal_matrix():
    previous = np.array([0.2, -0.4])
    wi = np.zeros((6, 1))
    wh = np.zeros((6, 2))
    wh[4:] = [[0.7, -0.3], [0.6, 0.2]]
    bi = np.array([math.log(3), -math.log(3), 0.0, 0.0, 0.1, -0.2])
    bh = np.array([0.0, 0.0, 0.0, 0.0, 0.5, -0.1])
    actual = numeric._gru_step(np, np.zeros(1), previous, (wi, wh, bi, bh))
    expected = [
        0.5 * math.tanh(0.1 + 0.75 * (0.7 * 0.2 - 0.3 * -0.4 + 0.5)) + 0.1,
        0.5 * math.tanh(-0.2 + 0.25 * (0.6 * 0.2 + 0.2 * -0.4 - 0.1)) - 0.2,
    ]
    wrong = (
        0.5 * np.tanh(bi[4:] + wh[4:] @ (np.array([0.75, 0.25]) * previous) + bh[4:])
        + 0.5 * previous
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
    assert np.max(np.abs(actual - wrong)) > 0.02


@pytest.mark.parametrize("word_layers,turn_layers", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_full_hierarchy_matches_independent_scalar_oracle(word_layers, turn_layers):
    config = architecture(word_layers=word_layers, turn_layers=turn_layers)
    arrays = arrays_for(config)
    turns = ((3, 4, 1, 2), (2,), (5, 3, 2))
    expected_vectors, expected_states, expected_logits = scalar_hierarchy(config, arrays, turns)
    result = infer_encoded_turns(FrozenNeuralParameters.from_arrays(config, arrays), turns)
    np.testing.assert_allclose(result.turn_vectors, expected_vectors, rtol=0, atol=2e-15)
    np.testing.assert_allclose(result.turn_states, expected_states, rtol=0, atol=2e-15)
    np.testing.assert_allclose(result.logits, expected_logits, rtol=0, atol=2e-15)
    np.testing.assert_allclose(
        result.probabilities, [scalar_sigmoid(x) for x in expected_logits], rtol=0, atol=2e-15
    )


def test_word_and_turn_order_matter_but_later_turns_do_not_change_prefix():
    config = architecture(word_layers=2, turn_layers=2)
    parameters = FrozenNeuralParameters.from_arrays(config, arrays_for(config))
    turns = ((3, 4, 2), (5, 1, 2), (4, 3, 2))
    full = infer_encoded_turns(parameters, turns)
    for endpoint in (1, 2):
        partial = infer_encoded_turns(parameters, turns[:endpoint])
        np.testing.assert_allclose(partial.logits, full.logits[:endpoint], rtol=0, atol=2e-15)
    reordered_words = infer_encoded_turns(parameters, ((4, 3, 2), *turns[1:]))
    reordered_turns = infer_encoded_turns(parameters, tuple(reversed(turns)))
    assert abs(full.logits[0] - reordered_words.logits[0]) > 1e-8
    assert abs(full.logits[-1] - reordered_turns.logits[-1]) > 1e-8


def test_zero_model_extreme_bias_and_stable_sigmoid():
    config = architecture()
    arrays = arrays_for(config, zero=True)
    for bias, expected in [(0.0, 0.5), (1e6, 1.0), (-1e6, 0.0)]:
        arrays["output.bias"][0] = bias
        result = infer_encoded_turns(FrozenNeuralParameters.from_arrays(config, arrays), ((2,),))
        assert result.logits == (bias,)
        assert result.probabilities == (expected,)
        assert result.turn_states == ((0.0, 0.0, 0.0, 0.0),)
    with np.errstate(all="raise"):
        values = numeric._sigmoid(np, np.array([-700.0, 0.0, 700.0]))
    assert values[0] > 0 and values[1] == 0.5 and values[2] == 1


def test_immutable_owned_bytes_digest_and_input_independence():
    config = architecture()
    arrays = arrays_for(config)
    frozen = FrozenNeuralParameters.from_arrays(config, arrays)
    original = frozen.digest
    arrays["embedding.weight"][:] = 999
    assert frozen.digest == original
    viewed = frozen.arrays()
    with pytest.raises(TypeError):
        viewed["extra"] = np.zeros(1)
    with pytest.raises(ValueError):
        viewed["embedding.weight"].setflags(write=True)
    with pytest.raises(ValueError):
        viewed["embedding.weight"].base.setflags(write=True)
    with pytest.raises(ValueError):
        viewed["embedding.weight"][0, 0] = 1
    assert FrozenNeuralParameters(config, frozen.tensors).digest == original
    changed = arrays_for(config)
    changed["output.bias"][0] += 0.125
    assert FrozenNeuralParameters.from_arrays(config, changed).digest != original


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "float64",
        "big_endian",
        "object",
        "shape",
        "strides",
        "nan",
        "inf",
        "magnitude",
        "array_subclass",
    ],
)
def test_reject_malformed_arrays(kind):
    config = architecture()
    arrays = arrays_for(config)
    name = "embedding.weight"
    if kind == "missing":
        arrays.pop(name)
    elif kind == "extra":
        arrays["extra"] = np.zeros(1, dtype="<f4")
    elif kind in ("float64", "big_endian", "object"):
        arrays[name] = arrays[name].astype(
            {"float64": "f8", "big_endian": ">f4", "object": "O"}[kind]
        )
    elif kind == "shape":
        arrays[name] = arrays[name].ravel()
    elif kind == "strides":
        arrays[name] = np.asfortranarray(arrays[name])
    elif kind == "array_subclass":

        class CustomArray(np.ndarray):
            pass

        arrays[name] = arrays[name].view(CustomArray)
    else:
        arrays[name][0, 0] = {"nan": float("nan"), "inf": float("inf"), "magnitude": 1e7}[kind]
    with pytest.raises(ValueError):
        FrozenNeuralParameters.from_arrays(config, arrays)


@pytest.mark.parametrize(
    "turns",
    [
        (),
        [],
        ((2,), []),
        ((),),
        ((0, 2),),
        ((True, 2),),
        ((1,),),
        ((2, 3, 2),),
        ((6, 2),),
        ((1.0, 2),),
    ],
)
def test_reject_invalid_token_structure(turns):
    config = architecture()
    parameters = FrozenNeuralParameters.from_arrays(config, arrays_for(config))
    with pytest.raises(ValueError):
        infer_encoded_turns(parameters, turns)


def test_exact_work_counts_and_per_call_limits():
    config = architecture()
    parameters = FrozenNeuralParameters.from_arrays(config, arrays_for(config))
    turns = ((3, 2), (2,))
    result = infer_encoded_turns(parameters, turns)
    assert result.work.turns == 2 and result.work.token_positions == 3
    assert (
        result.work.affine_multiplications
        == 3 * 2 * 3 * 2 * (2 + 2) + 2 * 3 * 4 * (4 + 4) + 2 * 2 * (4 + 1)
        == 356
    )
    base = NeuralNumericLimits(
        max_parameters=217,
        max_observed_turns=2,
        max_turn_tokens=2,
        max_token_positions=3,
        max_affine_multiplications=356,
        max_workspace_bytes=result.work.estimated_workspace_bytes,
    )
    assert infer_encoded_turns(parameters, turns, limits=base) == result
    for field in (
        "max_parameters",
        "max_observed_turns",
        "max_turn_tokens",
        "max_token_positions",
        "max_affine_multiplications",
        "max_workspace_bytes",
    ):
        with pytest.raises(ValueError, match=field if field != "max_observed_turns" else "bounded"):
            infer_encoded_turns(
                parameters, turns, limits=replace(base, **{field: getattr(base, field) - 1})
            )
    assert (
        infer_encoded_turns(
            parameters, turns, limits=replace(base, max_parameter_magnitude=1)
        ).logits
        == result.logits
    )
    with pytest.raises(ValueError, match="magnitude"):
        infer_encoded_turns(parameters, turns, limits=replace(base, max_parameter_magnitude=0.1))


def test_tensor_and_parameter_constructor_closed_inventory():
    for name, shape, data in [
        ("", (1,), bytes(4)),
        ("x", [1], bytes(4)),
        ("x", (True,), bytes(4)),
        ("x", (100003, 100003), b""),
        ("x", (1,), bytearray(4)),
    ]:
        with pytest.raises(ValueError):
            FrozenNeuralTensor(name, shape, data)
    config = architecture()
    parameters = FrozenNeuralParameters.from_arrays(config, arrays_for(config))
    for members in (
        parameters.tensors[:-1],
        list(parameters.tensors),
        tuple(reversed(parameters.tensors)),
        (None, *parameters.tensors[1:]),
    ):
        with pytest.raises(ValueError):
            FrozenNeuralParameters(config, members)
    with pytest.raises(ValueError, match="limits"):
        FrozenNeuralParameters(config, parameters.tensors, {})
    with pytest.raises(ValueError, match="max_parameters"):
        FrozenNeuralParameters(config, parameters.tensors, NeuralNumericLimits(max_parameters=216))
    with pytest.raises(ValueError, match="limits"):
        FrozenNeuralParameters.from_arrays(config, {}, limits={})
    with pytest.raises(ValueError, match="max_parameters"):
        FrozenNeuralParameters.from_arrays(
            config, {}, limits=NeuralNumericLimits(max_parameters=216)
        )
    with pytest.raises(ValueError, match="FrozenNeuralParameters"):
        infer_encoded_turns({}, ((2,),))
    with pytest.raises(ValueError, match="numeric limits"):
        infer_encoded_turns(parameters, ((2,),), limits={})
