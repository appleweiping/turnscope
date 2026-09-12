"""Independent sparse-coordinate derivatives and hostile-admission review oracles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest

from turnscope import neural_ablation_math as subject
from turnscope.neural_forecast_math import NeuralArchitecture

ARCH = NeuralArchitecture(6)
MODES = ("current-turn.v1", "mean-word.v1", "order-erased.v1")
TURNS = ((3, 3, 2), (1, 4, 2), (2,))


def sparse_arrays(mode):
    arrays = {
        name: np.zeros(shape, dtype="<f4")
        for name, shape in subject.ablation_parameter_shapes(mode, ARCH).items()
    }
    arrays["embedding.weight"][1:, :2] = np.array(
        [[0.25, -0.125], [0.5, 0.25], [-0.25, 0.375], [0.125, -0.5], [0.75, -0.25]],
        dtype="<f4",
    )
    for name, value in arrays.items():
        if name.startswith(("word.", "turn.")):
            if "weight" in name:
                value[[0, 64, 128], 0] = [0.25, -0.375, 0.5]
                value[[0, 64, 128], 1] = [-0.125, 0.25, 0.375]
                if value.shape[1] == 128:
                    value[[0, 64, 128], 64] = [0.375, 0.125, -0.25]
            else:
                value[[0, 64, 128]] = [0.125, -0.25, 0.375]
    arrays["head.weight"][0, :2] = [0.75, -0.25]
    arrays["head.bias"][0] = -0.125
    arrays["output.weight"][0, 0] = 0.5
    arrays["output.bias"][0] = 0.25
    return arrays


def production(mode, arrays, turns=TURNS):
    frozen = subject.FrozenAblationParameters.from_arrays(mode, ARCH, arrays)
    return subject.infer_ablation_turns(frozen, turns)


def torch_oracle(mode, arrays, turns=TURNS):
    """Use native GRUCell, not the implementation's reset-after recurrence helper."""
    torch = pytest.importorskip("torch")
    tensors = {
        name: torch.tensor(value.astype(np.float64), dtype=torch.float64, requires_grad=True)
        for name, value in arrays.items()
    }

    def cell(vector, hidden, level, reverse=False):
        suffix = "_l0_reverse" if reverse else "_l0"
        # Functional native GRUCell gives gradients with respect to our leaf tensors.
        return torch._VF.gru_cell(
            vector.unsqueeze(0),
            hidden.unsqueeze(0),
            tensors[f"{level}.weight_ih{suffix}"],
            tensors[f"{level}.weight_hh{suffix}"],
            tensors[f"{level}.bias_ih{suffix}"],
            tensors[f"{level}.bias_hh{suffix}"],
        )[0]

    if mode == "current-turn.v1":
        tensors["turn.weight_hh_l0"] = torch.full(
            (192, 64), 0.375, dtype=torch.float64, requires_grad=True
        )
    previous = torch.zeros(64, dtype=torch.float64)
    total = torch.zeros(64, dtype=torch.float64)
    logits = []
    for count, turn in enumerate(turns, 1):
        sequence = tensors["embedding.weight"][list(turn)]
        if mode == "current-turn.v1":
            forward = backward = torch.zeros(64, dtype=torch.float64)
            for vector in sequence:
                forward = cell(vector, forward, "word")
            for vector in reversed(sequence):
                backward = cell(vector, backward, "word", True)
            context = cell(
                torch.cat((forward, backward)), torch.zeros(64, dtype=torch.float64), "turn"
            )
        elif mode == "mean-word.v1":
            mean = sequence.mean(dim=0)
            context = cell(torch.cat((mean, mean)), previous, "turn")
            previous = context
        else:
            total = total + sequence.mean(dim=0)
            context = total / count
        hidden = torch.tanh(tensors["head.weight"] @ context + tensors["head.bias"])
        logits.append((tensors["output.weight"] @ hidden + tensors["output.bias"])[0])
    return tensors, torch.stack(logits)


@pytest.mark.parametrize("mode", MODES)
def test_every_active_block_native_autograd_matches_quantized_finite_difference(mode):
    arrays = sparse_arrays(mode)
    tensors, logits = torch_oracle(mode, arrays)
    weights = np.array([0.25, 0.5, 1.0])
    torch = pytest.importorskip("torch")
    loss = (logits * torch.tensor(weights)).sum()
    loss.backward()
    np.testing.assert_allclose(
        production(mode, arrays).logits, logits.detach().numpy(), rtol=0, atol=2e-15
    )
    for name, value in arrays.items():
        gradient = tensors[name].grad.detach().numpy()
        index = np.unravel_index(np.abs(gradient).argmax(), gradient.shape)
        expected = gradient[index]
        assert abs(expected) > 1e-10, f"authored fixture must exercise active block {name}"
        plus, minus = value.copy(), value.copy()
        plus[index] += 2**-10
        minus[index] -= 2**-10
        positive = weights @ production(mode, {**arrays, name: plus}).logits
        negative = weights @ production(mode, {**arrays, name: minus}).logits
        # Divide by actual stored float32 perturbation, not an ideal decimal delta.
        observed = (positive - negative) / (float(plus[index]) - float(minus[index]))
        assert observed == pytest.approx(expected, rel=2e-5, abs=2e-8), name
    if mode == "current-turn.v1":
        assert np.count_nonzero(tensors["turn.weight_hh_l0"].grad.numpy()) == 0
        recurrent_bias_gradient = tensors["turn.bias_hh_l0"].grad.numpy()
        assert all(abs(recurrent_bias_gradient[i]) > 1e-8 for i in (0, 64, 128))
        assert "turn.weight_hh_l0" not in arrays


@pytest.mark.parametrize("mode", ("mean-word.v1", "order-erased.v1"))
def test_mean_occurrence_eos_and_prefix_gradient_coefficients(mode):
    arrays = sparse_arrays(mode)
    turns = ((3, 3, 2), (1, 2), (2,))
    base = production(mode, arrays, turns)
    for token, expected_means, expected_prefix in (
        (3, (2 / 3, 0, 0), (2 / 3, 1 / 3, 2 / 9)),
        (1, (0, 1 / 2, 0), (0, 1 / 4, 1 / 6)),
        (2, (1 / 3, 1 / 2, 1), (1 / 3, 5 / 12, 11 / 18)),
        (5, (0, 0, 0), (0, 0, 0)),
    ):
        changed = {name: value.copy() for name, value in arrays.items()}
        changed["embedding.weight"][token, 0] += 0.125
        after = production(mode, changed, turns)
        actual = (np.array(after.turn_vectors)[:, 0] - np.array(base.turn_vectors)[:, 0]) / 0.125
        np.testing.assert_allclose(actual, expected_means, rtol=0, atol=1e-14)
        if mode == "mean-word.v1":
            np.testing.assert_array_equal(
                np.array(after.turn_vectors)[:, :64], np.array(after.turn_vectors)[:, 64:]
            )
        else:
            actual = (np.array(after.turn_states)[:, 0] - np.array(base.turn_states)[:, 0]) / 0.125
            np.testing.assert_allclose(actual, expected_prefix, rtol=0, atol=1e-14)


@pytest.mark.parametrize("mode", MODES)
def test_future_only_embedding_perturbation_cannot_change_prefix_logit_or_derivative(mode):
    arrays = sparse_arrays(mode)
    turns = ((3, 2), (4, 2), (5, 5, 2))
    original = production(mode, arrays, turns)
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["embedding.weight"][5, 0] += 0.5
    later = production(mode, changed, turns)
    np.testing.assert_allclose(later.logits[:2], original.logits[:2], rtol=0, atol=1e-15)
    assert abs(later.logits[-1] - original.logits[-1]) > 1e-7
    for end in (1, 2):
        np.testing.assert_allclose(
            production(mode, arrays, turns[:end]).logits, original.logits[:end], rtol=0, atol=1e-15
        )


@pytest.mark.parametrize("mutation", ("dtype", "shape", "strided"))
def test_late_malformed_member_rejects_before_any_immutable_tensor_copy(mutation, monkeypatch):
    arrays = sparse_arrays("order-erased.v1")
    if mutation == "strided":
        arrays["output.weight"] = np.zeros((1, 64), dtype="<f4")[:, ::2]
    else:
        arrays["output.bias"] = {
            "dtype": np.zeros(1, dtype=np.float64),
            "shape": np.zeros(2, dtype="<f4"),
        }[mutation]

    def forbidden(*_args, **_kwargs):
        pytest.fail("copied parameter bytes before complete structure validation")

    monkeypatch.setattr(subject, "FrozenNeuralTensor", forbidden)
    with pytest.raises(ValueError):
        subject.FrozenAblationParameters.from_arrays("order-erased.v1", ARCH, arrays)


def test_mapping_cannot_hide_extra_inactive_parameter_by_overriding_keys():
    arrays = sparse_arrays("order-erased.v1")

    class FalseKeys(Mapping):
        def __iter__(self):
            return iter((*arrays, "turn.weight_hh_l0"))

        def __len__(self):
            return len(arrays) + 1

        def __getitem__(self, key):
            return arrays[key]

        def keys(self):
            return arrays.keys()

    with pytest.raises(ValueError, match=r"inventory|names|mapping"):
        subject.FrozenAblationParameters.from_arrays("order-erased.v1", ARCH, FalseKeys())


@pytest.mark.parametrize("mutation", ("shape", "dtype"))
@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array.*:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:Setting the dtype on a NumPy array.*:DeprecationWarning")
def test_late_mapping_callback_invalidating_earlier_array_rejects_before_copy(
    mutation, monkeypatch
):
    arrays = sparse_arrays("order-erased.v1")

    class Mutating(Mapping):
        def __iter__(self):
            return iter(arrays)

        def __len__(self):
            return len(arrays)

        def __getitem__(self, key):
            if key == "output.bias":
                if mutation == "shape":
                    arrays["embedding.weight"].shape = (64, 6)
                else:
                    arrays["embedding.weight"].dtype = np.float64
            return arrays[key]

    monkeypatch.setattr(
        subject, "FrozenNeuralTensor", lambda *_a, **_k: pytest.fail("stale validated array copied")
    )
    with pytest.raises(ValueError, match="C-contiguous"):
        subject.FrozenAblationParameters.from_arrays("order-erased.v1", ARCH, Mutating())


def test_late_callback_rebinding_mapping_cannot_swap_the_once_fetched_array():
    arrays = sparse_arrays("order-erased.v1")
    first_embedding = arrays["embedding.weight"].copy()
    reads = {}

    class Rebinding(Mapping):
        def __iter__(self):
            return iter(arrays)

        def __len__(self):
            return len(arrays)

        def __getitem__(self, key):
            reads[key] = reads.get(key, 0) + 1
            assert reads[key] == 1, "mapping values must be fetched once, not reread for copying"
            if key == "output.bias":
                arrays["embedding.weight"] = np.zeros_like(first_embedding)
            return arrays[key]

    frozen = subject.FrozenAblationParameters.from_arrays("order-erased.v1", ARCH, Rebinding())
    assert len(reads) == 5
    np.testing.assert_array_equal(frozen.arrays()["embedding.weight"], first_embedding)
    assert not np.array_equal(frozen.arrays()["embedding.weight"], arrays["embedding.weight"])


@pytest.mark.parametrize("mode", MODES)
def test_invalid_structure_or_budget_does_not_import_numpy(mode, monkeypatch):
    shapes = subject.ablation_parameter_shapes(mode, ARCH)
    count = sum(np.prod(shape) for shape in shapes.values())

    class NeverRead(Mapping):
        def __iter__(self):
            pytest.fail("mapping inspected before max_parameters admission")

        def __len__(self):
            pytest.fail("mapping inspected before max_parameters admission")

        def __getitem__(self, key):
            pytest.fail("mapping value read before max_parameters admission")

    monkeypatch.setattr(subject, "_numpy", lambda: pytest.fail("NumPy reached before admission"))
    limits = replace(subject.NeuralNumericLimits(), max_parameters=int(count) - 1)
    with pytest.raises(ValueError, match="max_parameters"):
        subject.FrozenAblationParameters.from_arrays(mode, ARCH, NeverRead(), limits=limits)
    with pytest.raises(ValueError):
        subject.admit_ablation_turns(mode, ARCH, ((3, 2), (0, 2)))
