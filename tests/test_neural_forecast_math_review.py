"""Independent immutable-input, admission-order and causal GRU checks.

These tests require NumPy, never Torch. The optional installed-Torch comparison
used during review is a separate local oracle, not a suite dependency.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest

from turnscope import neural_forecast_math as model


def arrays_for(architecture):
    """Authored unequal gate/direction weights, exactly reproducible in float32."""
    return {
        name: np.ascontiguousarray(
            (((np.arange(np.prod(shape), dtype=np.int64) + 17 * index) % 29) - 14)
            .reshape(shape)
            .astype("<f4")
            / np.float32(19.0)
        )
        for index, (name, shape) in enumerate(model.parameter_shapes(architecture).items())
    }


def tiny_parameters():
    architecture = model.NeuralArchitecture(7, 3, 2, 3)
    return model.FrozenNeuralParameters.from_arrays(architecture, arrays_for(architecture))


def test_validated_mapping_members_are_snapshotted_without_second_lookup():
    architecture = model.NeuralArchitecture(3, 1, 1, 2)
    shapes = model.parameter_shapes(architecture)

    class ChangingMapping(Mapping):
        def __init__(self):
            self.reads = dict.fromkeys(shapes, 0)

        def __iter__(self):
            return iter(shapes)

        def __len__(self):
            return len(shapes)

        def __getitem__(self, key):
            self.reads[key] += 1
            if self.reads[key] == 1:
                return np.zeros(shapes[key], dtype="<f4")
            # Same byte length, wrong declared type; the bit pattern is float32 1.
            # This demonstrates the validation bypass without a large allocation.
            return np.full(shapes[key], 1065353216, dtype="<i4")

    supplied = ChangingMapping()
    parameters = model.FrozenNeuralParameters.from_arrays(architecture, supplied)
    assert set(supplied.reads.values()) == {1}
    for values in parameters.arrays().values():
        assert np.count_nonzero(values) == 0


def test_huge_integer_magnitude_is_rejected_as_a_controlled_configuration_error():
    with pytest.raises(ValueError, match="max_parameter_magnitude"):
        model.NeuralNumericLimits(max_parameter_magnitude=10**400)


def test_workspace_admission_precedes_stricter_numeric_parameter_scan(monkeypatch):
    parameters = tiny_parameters()
    limits = replace(parameters.limits, max_workspace_bytes=1, max_parameter_magnitude=0.5)

    def forbidden_numpy():
        pytest.fail("numerical arrays were accessed before the impossible workspace was rejected")

    monkeypatch.setattr(model, "_numpy", forbidden_numpy)
    with pytest.raises(ValueError, match="max_workspace_bytes"):
        model.infer_encoded_turns(parameters, ((3, 2),), limits=limits)


def test_parameter_count_admission_precedes_mapping_reads_or_numpy(monkeypatch):
    architecture = model.NeuralArchitecture(7, 3, 2, 3)

    class Unreadable(Mapping):
        def __iter__(self):
            pytest.fail("mapping was touched before parameter admission")

        def __len__(self):
            pytest.fail("mapping was touched before parameter admission")

        def __getitem__(self, key):
            pytest.fail("mapping was touched before parameter admission")

    def forbidden_numpy():
        pytest.fail("NumPy was touched before parameter admission")

    monkeypatch.setattr(model, "_numpy", forbidden_numpy)
    with pytest.raises(ValueError, match="max_parameters"):
        model.FrozenNeuralParameters.from_arrays(
            architecture, Unreadable(), limits=model.NeuralNumericLimits(max_parameters=1)
        )


def test_returned_array_views_cannot_mutate_frozen_owners_or_share_caller_storage():
    architecture = model.NeuralArchitecture(7, 3, 2, 3)
    supplied = arrays_for(architecture)
    parameters = model.FrozenNeuralParameters.from_arrays(architecture, supplied)
    digest = parameters.digest
    expected = parameters.tensors[0].data
    supplied["embedding.weight"].fill(42)
    exposed = parameters.arrays()
    value = exposed["embedding.weight"]
    assert value.tobytes() == expected
    assert not value.flags.writeable
    assert value.data.readonly
    with pytest.raises(ValueError):
        value.setflags(write=True)
    with pytest.raises(ValueError):
        value.base.setflags(write=True)
    with pytest.raises(TypeError):
        exposed["embedding.weight"] = supplied["embedding.weight"]
    assert parameters.digest == digest
    assert parameters.arrays()["embedding.weight"].tobytes() == expected


@pytest.mark.parametrize("word_layers", (1, 2))
@pytest.mark.parametrize("turn_layers", (1, 2))
def test_later_turn_content_cannot_change_an_observed_prefix(word_layers, turn_layers):
    architecture = model.NeuralArchitecture(7, 3, 2, 3, word_layers, turn_layers)
    parameters = model.FrozenNeuralParameters.from_arrays(architecture, arrays_for(architecture))
    turns = ((3, 4, 2), (5, 2), (6, 3, 4, 2))
    complete = model.infer_encoded_turns(parameters, turns)
    for length in (1, 2):
        prefix = model.infer_encoded_turns(parameters, turns[:length])
        assert prefix.turn_vectors == complete.turn_vectors[:length]
        assert prefix.turn_states == complete.turn_states[:length]
        np.testing.assert_allclose(prefix.logits, complete.logits[:length], rtol=0, atol=1e-15)
        np.testing.assert_allclose(
            prefix.probabilities, complete.probabilities[:length], rtol=0, atol=1e-15
        )


def test_module_import_is_torch_free_and_does_not_eagerly_load_numpy():
    source = """
import importlib.abc
import sys
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy'}:
            raise AssertionError('unexpected eager numerical backend import: ' + fullname)
sys.meta_path.insert(0, NoBackend())
import turnscope.neural_forecast_math as module
assert module.NeuralArchitecture(3, 1, 1, 2).parameter_count > 0
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", source],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stderr
