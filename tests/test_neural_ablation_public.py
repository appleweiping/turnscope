"""Public controls remain optional-dependency-free until numerical work begins."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import turnscope
from turnscope import AblationEventForecaster, AblationForecastConfig, NeuralArtifactLimits
from turnscope import neural_ablation_artifact as archive


def test_public_import_and_help_do_not_load_optional_numerical_dependencies():
    command = """
import importlib.abc
import sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch'}:
            raise AssertionError('optional dependency imported during public import')
sys.meta_path.insert(0, Forbidden())
import turnscope
from turnscope.cli import main
assert turnscope.AblationForecastConfig('order-erased.v1').variant == 'order-erased.v1'
assert all(name in turnscope.__all__ for name in (
    'AblationEventForecaster', 'AblationForecastConfig', 'AblationForecastPrediction',
    'AblationForecastState', 'AblationPoolingLimits', 'AblationTrainingLimits',
    'AblationTrainingResult', 'load_ablation_forecaster', 'save_ablation_forecaster'))
try:
    main(['--help'])
except SystemExit as outcome:
    assert outcome.code == 0
assert not any(name.split('.')[0] in {'numpy', 'torch'} for name in sys.modules)
"""
    # Preserve the current installed/editable package resolution; installed-wheel
    # isolation is separately tested by the shipped -I demonstration.
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", command],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("overwrite", [False, True])
def test_public_save_forwards_explicit_publication_policy(monkeypatch, tmp_path, overwrite):
    model = AblationEventForecaster(config=AblationForecastConfig("current-turn.v1"))
    sentinel = object()
    save = Mock(return_value=sentinel)
    monkeypatch.setattr(archive, "save_ablation_forecaster", save)
    target = tmp_path / "model.tsa"
    assert model.save(target, overwrite=overwrite) is sentinel
    save.assert_called_once_with(model, target, overwrite=overwrite)


def test_public_load_passes_restrictive_limits_without_widening(monkeypatch, tmp_path):
    sentinel = object()
    load = Mock(return_value=sentinel)
    monkeypatch.setattr(archive, "load_ablation_forecaster", load)
    limits = NeuralArtifactLimits(max_file_bytes=1024)
    target = tmp_path / "model.tsa"
    assert AblationEventForecaster.load(target, limits=limits) is sentinel
    load.assert_called_once_with(target, limits=limits)


def test_public_wrapper_uses_distinct_real_archive_and_exclusive_default(tmp_path):
    from test_neural_ablation_artifact import authored

    model = authored("order-erased.v1")
    target = tmp_path / "private-model.tsa"
    publication = model.save(target)
    assert Path(publication.path) == target
    assert AblationEventForecaster.load(target).digest == model.digest
    with pytest.raises(FileExistsError):
        model.save(target)
    with pytest.raises(turnscope.NeuralArtifactError):
        turnscope.load_neural_forecaster(target)
    replaced = model.save(target, overwrite=True)
    assert replaced.sha256 == publication.sha256
    with pytest.raises(turnscope.NeuralArtifactError):
        AblationEventForecaster.load(target, limits=NeuralArtifactLimits(max_file_bytes=1024))
