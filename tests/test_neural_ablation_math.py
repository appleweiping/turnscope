"""Independent equations and admission tests, not evidence of trained CGA quality."""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from turnscope import neural_ablation_math as ablation
from turnscope.neural_ablation_math import (
    ABLATION_VARIANTS,
    AblationPoolingLimits,
    FrozenAblationParameters,
    ablation_numerical_version,
    ablation_parameter_shapes,
    admit_ablation_turns,
    infer_ablation_turns,
)
from turnscope.neural_forecast_math import (
    FrozenNeuralParameters,
    FrozenNeuralTensor,
    NeuralArchitecture,
    infer_encoded_turns,
    parameter_shapes,
)

ARCHITECTURE = NeuralArchitecture(7)
TURNS = ((3, 1, 2), (4, 4, 3, 2), (2,), (5, 3, 6, 2))


def arrays_for(variant, *, zero=False):
    arrays = {}
    for name, shape in ablation_parameter_shapes(variant, ARCHITECTURE).items():
        offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
        values = np.arange(math.prod(shape), dtype=np.int64)
        values = ((values * 19 + offset) % 103 - 51) / 1400
        arrays[name] = (
            np.zeros(shape, dtype="<f4") if zero else np.array(values, dtype="<f4").reshape(shape)
        )
    arrays["embedding.weight"][0] = 0
    return arrays


def frozen(variant, **kwargs):
    return FrozenAblationParameters.from_arrays(
        variant, ARCHITECTURE, arrays_for(variant), **kwargs
    )


def sigmoid(value):
    return 1 / (1 + math.exp(-value))


def linear(matrix, vector, bias):
    return [
        math.fsum(float(weight) * value for weight, value in zip(row, vector, strict=True))
        + float(offset)
        for row, offset in zip(matrix, bias, strict=True)
    ]


def scalar_gru(sequence, arrays, level, *, reverse=False, reset_each=False):
    """Plain-Python scalar reset-after equations; no production recurrence helper."""
    suffix = "_l0_reverse" if reverse else "_l0"
    states = []
    previous = [0.0] * 64
    for vector in reversed(sequence) if reverse else sequence:
        incoming = linear(
            arrays[f"{level}.weight_ih{suffix}"], vector, arrays[f"{level}.bias_ih{suffix}"]
        )
        recurrent = (
            [float(value) for value in arrays[f"{level}.bias_hh{suffix}"]]
            if reset_each
            else linear(
                arrays[f"{level}.weight_hh{suffix}"], previous, arrays[f"{level}.bias_hh{suffix}"]
            )
        )
        current = []
        for index in range(64):
            reset = sigmoid(incoming[index] + recurrent[index])
            update = sigmoid(incoming[index + 64] + recurrent[index + 64])
            candidate = math.tanh(incoming[index + 128] + reset * recurrent[index + 128])
            current.append(
                (1 - update) * candidate + update * (0 if reset_each else previous[index])
            )
        states.append(current)
        previous = current
    return states


def scalar_ablation(variant, arrays, turns):
    vectors = []
    for turn in turns:
        embeddings = [
            [float(value) for value in arrays["embedding.weight"][token]] for token in turn
        ]
        if variant == "current-turn.v1":
            forward = scalar_gru(embeddings, arrays, "word")[-1]
            backward = scalar_gru(embeddings, arrays, "word", reverse=True)[-1]
            vectors.append(forward + backward)
        else:
            mean = [math.fsum(row[index] for row in embeddings) / len(turn) for index in range(64)]
            vectors.append(mean + mean if variant == "mean-word.v1" else mean)
    if variant == "current-turn.v1":
        states = scalar_gru(vectors, arrays, "turn", reset_each=True)
    elif variant == "mean-word.v1":
        states = scalar_gru(vectors, arrays, "turn")
    else:
        states = [
            [math.fsum(row[index] for row in vectors[: end + 1]) / (end + 1) for index in range(64)]
            for end in range(len(vectors))
        ]
    logits = []
    for state in states:
        hidden = [
            math.tanh(value) for value in linear(arrays["head.weight"], state, arrays["head.bias"])
        ]
        logits.append(linear(arrays["output.weight"], hidden, arrays["output.bias"])[0])
    return vectors, states, logits


@pytest.mark.parametrize(
    ("variant", "members", "constant"),
    [("current-turn.v1", 16, 76993), ("mean-word.v1", 9, 39361), ("order-erased.v1", 5, 2113)],
)
def test_inventory_parameter_formula_and_inactive_members(variant, members, constant):
    for vocabulary in (3, 7, 10003):
        architecture = NeuralArchitecture(vocabulary)
        shapes = ablation_parameter_shapes(variant, architecture)
        assert len(shapes) == members
        assert sum(math.prod(shape) for shape in shapes.values()) == 64 * vocabulary + constant
        if variant == "current-turn.v1":
            assert "turn.weight_hh_l0" not in shapes
            assert "turn.bias_hh_l0" in shapes
        elif variant == "mean-word.v1":
            assert not any(name.startswith("word.") for name in shapes)
        else:
            assert not any(name.startswith(("word.", "turn.")) for name in shapes)


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_independent_scalar_equations_every_state_and_probability(variant):
    arrays = arrays_for(variant)
    expected_vectors, expected_states, expected_logits = scalar_ablation(variant, arrays, TURNS)
    result = infer_ablation_turns(
        FrozenAblationParameters.from_arrays(variant, ARCHITECTURE, arrays), TURNS
    )
    np.testing.assert_allclose(result.turn_vectors, expected_vectors, rtol=0, atol=2e-16)
    np.testing.assert_allclose(result.turn_states, expected_states, rtol=0, atol=2e-16)
    np.testing.assert_allclose(result.logits, expected_logits, rtol=0, atol=2e-16)
    np.testing.assert_allclose(
        result.probabilities, [sigmoid(value) for value in expected_logits], rtol=0, atol=2e-16
    )


def test_current_only_matches_main_gru_from_zero_for_each_turn_not_full_history():
    arrays = arrays_for("current-turn.v1")
    main_arrays = {
        name: arrays[name] if name in arrays else np.full(shape, 0.25, dtype="<f4")
        for name, shape in parameter_shapes(ARCHITECTURE).items()
    }
    main = FrozenNeuralParameters.from_arrays(ARCHITECTURE, main_arrays)
    actual = infer_ablation_turns(frozen("current-turn.v1"), TURNS)
    expected = [infer_encoded_turns(main, (turn,)).logits[0] for turn in TURNS]
    np.testing.assert_allclose(actual.logits, expected, rtol=0, atol=2e-16)
    assert abs(infer_encoded_turns(main, TURNS).logits[-1] - actual.logits[-1]) > 1e-6


def test_zero_state_gate_preserves_reset_gated_recurrent_candidate_bias():
    arrays = arrays_for("current-turn.v1", zero=True)
    arrays["turn.bias_ih_l0"][:64] = -0.5
    arrays["turn.bias_hh_l0"][:64] = 0.25
    arrays["turn.bias_ih_l0"][64:128] = 0.75
    arrays["turn.bias_hh_l0"][64:128] = -0.125
    arrays["turn.bias_ih_l0"][128:] = 0.5
    arrays["turn.bias_hh_l0"][128:] = 0.25
    expected = (1 - sigmoid(0.625)) * math.tanh(0.5 + sigmoid(-0.25) * 0.25)
    model = FrozenAblationParameters.from_arrays("current-turn.v1", ARCHITECTURE, arrays)
    np.testing.assert_allclose(
        infer_ablation_turns(model, TURNS).turn_states, expected, rtol=0, atol=2e-16
    )
    arrays["turn.bias_hh_l0"][128:] = 0
    changed = FrozenAblationParameters.from_arrays("current-turn.v1", ARCHITECTURE, arrays)
    assert infer_ablation_turns(changed, TURNS).turn_states[0][0] != expected


@pytest.mark.parametrize("variant", ("mean-word.v1", "order-erased.v1"))
def test_pool_includes_eos_unk_and_duplicates_but_never_pad_or_raw_lengths(variant):
    arrays = arrays_for(variant, zero=True)
    arrays["embedding.weight"][1] = 2
    arrays["embedding.weight"][2] = 4
    arrays["embedding.weight"][3] = 10
    model = FrozenAblationParameters.from_arrays(variant, ARCHITECTURE, arrays)
    result = infer_ablation_turns(model, ((1, 3, 3, 2), (2,)))
    width = 128 if variant == "mean-word.v1" else 64
    assert result.turn_vectors == ((6.5,) * width, (4.0,) * width)
    if variant == "order-erased.v1":
        assert result.turn_states == ((6.5,) * 64, (5.25,) * 64)
        assert result.turn_states[-1][0] != (2 + 10 + 10 + 4 + 4) / 5


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_later_turns_cannot_change_earlier_outputs_and_prefix_recomputation_agrees(variant):
    model = frozen(variant)
    complete = infer_ablation_turns(model, TURNS)
    for end in range(1, len(TURNS) + 1):
        prefix = infer_ablation_turns(model, TURNS[:end])
        np.testing.assert_allclose(prefix.logits, complete.logits[:end], rtol=0, atol=1e-15)
    changed = infer_ablation_turns(model, (*TURNS[:2], (6, 6, 6, 2)))
    np.testing.assert_allclose(changed.logits[:2], complete.logits[:2], rtol=0, atol=1e-15)


def test_current_only_ignores_earlier_content_but_not_current_content():
    model = frozen("current-turn.v1")
    original = infer_ablation_turns(model, TURNS)
    assert infer_ablation_turns(model, ((6, 2), *TURNS[1:])).logits[-1] == original.logits[-1]
    assert (
        abs(infer_ablation_turns(model, (*TURNS[:-1], (4, 2))).logits[-1] - original.logits[-1])
        > 1e-8
    )


@pytest.mark.parametrize("variant", ("mean-word.v1", "order-erased.v1"))
def test_retained_within_turn_permutation_invariance(variant):
    model = frozen(variant)
    reordered = tuple((*reversed(turn[:-1]), 2) for turn in TURNS)
    np.testing.assert_allclose(
        infer_ablation_turns(model, reordered).logits,
        infer_ablation_turns(model, TURNS).logits,
        rtol=0,
        atol=1e-15,
    )


def test_mean_word_retains_history_order_while_order_erased_does_not():
    turns = (*TURNS[:2], TURNS[3])
    reordered = (turns[1], turns[0], turns[2])
    mean = frozen("mean-word.v1")
    erased = frozen("order-erased.v1")
    assert (
        abs(
            infer_ablation_turns(mean, turns).logits[-1]
            - infer_ablation_turns(mean, reordered).logits[-1]
        )
        > 1e-8
    )
    assert infer_ablation_turns(erased, turns).logits[-1] == pytest.approx(
        infer_ablation_turns(erased, reordered).logits[-1], abs=1e-15, rel=0
    )
    repeat = (turns[0],) * 4
    assert len(set(infer_ablation_turns(erased, repeat).logits)) == 1


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_owned_snapshot_and_variant_bound_digest(variant):
    arrays = arrays_for(variant)
    model = FrozenAblationParameters.from_arrays(variant, ARCHITECTURE, arrays)
    before = model.digest
    arrays["output.bias"][0] += 1
    assert model.digest == before
    with pytest.raises(ValueError):
        model.arrays()["output.bias"].setflags(write=True)
    with pytest.raises(TypeError):
        model.arrays()["unknown"] = np.zeros(1)
    with pytest.raises(FrozenInstanceError):
        model.variant = "main"
    assert FrozenAblationParameters.from_arrays(variant, ARCHITECTURE, arrays).digest != before
    assert ablation_numerical_version(variant)


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_exact_work_formulas_and_caps_before_numerical_allocation(variant, monkeypatch):
    model = frozen(variant)
    turns, positions = len(TURNS), sum(map(len, TURNS))
    affine = turns * 32 * 65
    additions = scalings = 0
    if variant == "current-turn.v1":
        affine += positions * 6 * 64 * 128 + turns * 3 * 64 * 128
    else:
        additions = (positions - turns) * 64
        scalings = turns * 64
        if variant == "mean-word.v1":
            affine += turns * 3 * 64 * 192
        else:
            additions += (turns - 1) * 64
            scalings *= 2
    work = admit_ablation_turns(variant, ARCHITECTURE, TURNS)
    assert (
        work.turns,
        work.token_positions,
        work.affine_multiplications,
        work.pooling_additions,
        work.pooling_scalings,
    ) == (turns, positions, affine, additions, scalings)
    exact = replace(
        model.limits,
        max_parameters=model.parameter_count,
        max_affine_multiplications=affine,
        max_workspace_bytes=work.estimated_workspace_bytes,
        max_token_positions=positions,
        max_observed_turns=turns,
        max_turn_tokens=4,
    )
    assert infer_ablation_turns(model, TURNS, limits=exact).work == work

    def forbidden():
        pytest.fail("admission must reject before NumPy allocation")

    monkeypatch.setattr(ablation, "_numpy", forbidden)
    for changes in (
        {"max_parameters": model.parameter_count - 1},
        {"max_affine_multiplications": affine - 1},
        {"max_workspace_bytes": work.estimated_workspace_bytes - 1},
        {"max_token_positions": positions - 1},
        {"max_observed_turns": turns - 1},
        {"max_turn_tokens": 3},
    ):
        with pytest.raises(ValueError):
            infer_ablation_turns(model, TURNS, limits=replace(exact, **changes))
    if additions + scalings:
        assert (
            admit_ablation_turns(
                variant,
                ARCHITECTURE,
                TURNS,
                pooling_limits=AblationPoolingLimits(additions + scalings),
            )
            == work
        )
        with pytest.raises(ValueError, match="max_pooling_operations"):
            infer_ablation_turns(
                model, TURNS, pooling_limits=AblationPoolingLimits(additions + scalings - 1)
            )


@pytest.mark.parametrize(
    "turns",
    [
        (),
        [],
        ((2,), []),
        ((True, 2),),
        ((0, 2),),
        ((7, 2),),
        ((3,),),
        ((2, 3, 2),),
        ((),),
        ((3.0, 2),),
    ],
)
def test_bad_encoded_inputs_reject_array_free(turns, monkeypatch):
    monkeypatch.setattr(ablation, "_numpy", lambda: pytest.fail("unexpected allocation"))
    with pytest.raises(ValueError):
        admit_ablation_turns("order-erased.v1", ARCHITECTURE, turns)


@pytest.mark.parametrize("variant", [None, True, "main", "mean-word.v2", [], 1])
def test_unknown_or_nonstring_variants_reject(variant):
    with pytest.raises(ValueError):
        ablation_parameter_shapes(variant, ARCHITECTURE)


@pytest.mark.parametrize(
    "changes",
    [
        {"word_layers": 2},
        {"turn_layers": 2},
        {"embedding_dim": 32},
        {"word_hidden": 32},
        {"turn_hidden": 32},
    ],
)
def test_unsupported_architectures_are_not_silently_projected(changes):
    with pytest.raises(ValueError, match="fixed 64-wide"):
        ablation_parameter_shapes("mean-word.v1", replace(ARCHITECTURE, **changes))


@pytest.mark.parametrize("value", [True, 0, -1, 1.0, 1_000_000_001])
def test_pooling_cap_is_a_bounded_strict_integer(value):
    with pytest.raises(ValueError):
        AblationPoolingLimits(value)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "dtype",
        "shape",
        "strided",
        "nan",
        "infinity",
        "large",
        "pad",
        "negative-zero-pad",
    ],
)
def test_parameter_boundaries(mutation):
    variant = "order-erased.v1"
    arrays = arrays_for(variant)
    if mutation == "extra":
        arrays["turn.weight_hh_l0"] = np.zeros((192, 64), dtype="<f4")
    elif mutation == "missing":
        arrays.pop("output.bias")
    elif mutation == "dtype":
        arrays["head.bias"] = arrays["head.bias"].astype(np.float64)
    elif mutation == "shape":
        arrays["head.bias"] = np.zeros(31, dtype="<f4")
    elif mutation == "strided":
        arrays["head.bias"] = np.zeros(64, dtype="<f4")[::2]
    elif mutation in ("nan", "infinity", "large"):
        arrays["head.bias"][0] = {"nan": np.nan, "infinity": np.inf, "large": 1_000_001}[mutation]
    else:
        arrays["embedding.weight"][0, 0] = 1 if mutation == "pad" else -0.0
    with pytest.raises(ValueError):
        FrozenAblationParameters.from_arrays(variant, ARCHITECTURE, arrays)


def test_constructor_rejects_cross_mode_order_and_shape_without_numpy(monkeypatch):
    model = frozen("mean-word.v1")
    monkeypatch.setattr(ablation, "_numpy", lambda: pytest.fail("unexpected allocation"))
    for changes in (
        {"variant": "order-erased.v1"},
        {"tensors": tuple(reversed(model.tensors))},
        {"tensors": model.tensors[:-1]},
        {"limits": None},
        {"pooling_limits": None},
        {"tensors": list(model.tensors)},
        {"tensors": (FrozenNeuralTensor("x", (1,), b"\0" * 4), *model.tensors[1:])},
    ):
        with pytest.raises(ValueError):
            replace(model, **changes)


def test_mapping_values_are_snapshotted_once():
    arrays = arrays_for("order-erased.v1")

    class Changing(Mapping):
        def __init__(self):
            self.reads = {}

        def __iter__(self):
            return iter(arrays)

        def __len__(self):
            return len(arrays)

        def __getitem__(self, key):
            self.reads[key] = self.reads.get(key, 0) + 1
            return arrays[key] if self.reads[key] == 1 else np.zeros(1)

    changing = Changing()
    model = FrozenAblationParameters.from_arrays("order-erased.v1", ARCHITECTURE, changing)
    assert set(changing.reads.values()) == {1}
    assert model.digest == frozen("order-erased.v1").digest


def test_stricter_magnitude_scans_actual_values_not_only_saved_bound():
    model = frozen("order-erased.v1")
    assert infer_ablation_turns(
        model, TURNS, limits=replace(model.limits, max_parameter_magnitude=0.04)
    )
    with pytest.raises(ValueError, match="max_parameter_magnitude"):
        infer_ablation_turns(
            model, TURNS, limits=replace(model.limits, max_parameter_magnitude=0.001)
        )


def test_lazy_import_and_inference_without_torch():
    source = Path(ablation.__file__).resolve().parents[1]
    script = f"""
import importlib.abc, sys
sys.path.insert(0, {str(source)!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('numpy', 'torch'):
            raise AssertionError(fullname)
block = Block()
sys.meta_path.insert(0, block)
from turnscope.neural_ablation_math import ablation_parameter_shapes
from turnscope.neural_forecast_math import NeuralArchitecture
assert len(ablation_parameter_shapes('order-erased.v1', NeuralArchitecture(3))) == 5
sys.meta_path.remove(block)
import numpy as np
sys.modules['torch'] = None
from turnscope.neural_ablation_math import FrozenAblationParameters, infer_ablation_turns
architecture = NeuralArchitecture(3)
shapes = ablation_parameter_shapes('order-erased.v1', architecture)
arrays = {{name: np.zeros(shape, dtype='<f4') for name, shape in shapes.items()}}
model = FrozenAblationParameters.from_arrays('order-erased.v1', architecture, arrays)
assert infer_ablation_turns(model, ((2,),)).probabilities == (0.5,)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
