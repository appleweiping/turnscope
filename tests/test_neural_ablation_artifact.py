"""Authored archive oracles; optional CPU fits are not CGA quality evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
import subprocess
import sys
import zipfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from turnscope import neural_ablation_artifact as api
from turnscope import neural_forecast_artifact as base
from turnscope.models import Conversation, Utterance
from turnscope.neural_ablation import (
    AblationEventForecaster,
    AblationForecastConfig,
    AblationForecastState,
)
from turnscope.neural_ablation_math import (
    ABLATION_VARIANTS,
    FrozenAblationParameters,
    ablation_parameter_shapes,
    infer_ablation_turns,
)
from turnscope.neural_ablation_train import AblationTrainingLimits, AblationTrainingResult
from turnscope.neural_forecast_data import SequenceLimits, SequencePolicy
from turnscope.neural_forecast_math import FrozenNeuralTensor, parameter_shapes
from turnscope.neural_forecast_train import NeuralEpoch, NeuralTrainingConfig
from turnscope.neural_token_data import SequenceVocabulary


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def authored(variant="order-erased.v1", *, history=None):
    config = AblationForecastConfig(variant, max_turn_tokens=4)
    vocabulary = SequenceVocabulary(("alpha", "beta"), (2, 1), 2, 4, "head")
    architecture = config.architecture(vocabulary.size)
    tensors = []
    for name, shape in ablation_parameter_shapes(variant, architecture).items():
        raw = struct.pack("<f", 0.001) * math.prod(shape)
        if name == "embedding.weight":
            raw = bytes(64 * 4) + raw[64 * 4 :]
        tensors.append(FrozenNeuralTensor(name, shape, raw))
    parameters = FrozenAblationParameters(variant, architecture, tuple(tensors))
    initial = tuple((tensor.name, sha(tensor.data)) for tensor in tensors)
    main_initial = tuple(
        (name, dict(initial).get(name, sha(name))) for name in parameter_shapes(architecture)
    )
    history = history or (NeuralEpoch(1, 0.7, 0.7, 1, 0.1),)
    training = NeuralTrainingConfig(epochs=len(history), patience=2)
    result = AblationTrainingResult(
        variant=variant,
        parameters=parameters,
        config=training,
        training_limits=AblationTrainingLimits(),
        history=history,
        selected_epoch=min(history, key=lambda item: item.validation_loss).epoch,
        initial_parameter_sha256=initial,
        main_initial_parameter_sha256=main_initial,
        changed_parameter_names=(),
        training_partition_digest=sha("train"),
        validation_partition_digest=sha("validation"),
        vocabulary_digest=vocabulary.digest,
        training_conversations=2,
        validation_conversations=2,
        training_prefixes=2,
        validation_prefixes=2,
        maximum_estimated_workspace_bytes=100_000_000,
        initialization_workspace_bytes=12 * architecture.parameter_count,
        estimated_affine_multiplications=10_000_000 * len(history),
        estimated_pooling_additions=0 if variant == "current-turn.v1" else 100_000 * len(history),
        estimated_pooling_scalings=0 if variant == "current-turn.v1" else 100_000 * len(history),
        torch_version="authored-not-trained",
        numpy_version="authored-not-trained",
        torch_threads=1,
    )
    groups = [frozenset((sha(name + "a"), sha(name + "b"))) for name in ("t", "v", "p")]
    state = AblationForecastState(
        config,
        SequencePolicy(),
        SequenceLimits(),
        vocabulary,
        result,
        0.6,
        0.75,
        *groups,
        sha("policy"),
        2,
        2,
    )
    model = AblationEventForecaster(config=config, training_config=training)
    model._state = state
    return model


def unpack(raw):
    members = [
        (name, bytes(data))
        for name, data in base._members(raw, base.NeuralArtifactLimits()).items()
    ]
    return json.loads(members[0][1]), members


def repack(manifest, members, *, model_hash=True):
    limits = base.NeuralArtifactLimits()
    if model_hash:
        declaration = {
            key: value for key, value in manifest["state"].items() if key != "vocabulary"
        }
        manifest["model_digest"] = sha(base._canonical(declaration, limits))
    manifest.pop("sha256", None)
    manifest["sha256"] = sha(base._canonical(manifest, limits))
    members[0] = ("manifest.json", base._canonical(manifest, limits))
    return base._archive(members, limits)


@pytest.fixture
def saved(tmp_path):
    source = authored()
    path = tmp_path / "model.tsa"
    api.save_ablation_forecaster(source, path)
    return path, source


@pytest.mark.parametrize("variant,count", tuple(zip(ABLATION_VARIANTS, (16, 9, 5), strict=True)))
def test_deterministic_exact_state_and_inference_roundtrip(tmp_path, variant, count):
    model = authored(variant)
    path = tmp_path / "model.tsa"
    result = api.save_ablation_forecaster(model, path)
    restored = api.load_ablation_forecaster(path)
    assert restored.state == model.state
    assert restored.digest == model.digest
    assert restored.training_summary == model.training_summary
    assert restored.state.vocabulary.to_dict() == model.state.vocabulary.to_dict()
    assert infer_ablation_turns(
        restored.state.parameters, ((3, 2), (4, 3, 2))
    ) == infer_ablation_turns(model.state.parameters, ((3, 2), (4, 3, 2)))
    second = tmp_path / "second.tsa"
    api.save_ablation_forecaster(restored, second)
    assert path.read_bytes() == second.read_bytes()
    assert result == api.NeuralSaveResult(str(path), sha(path.read_bytes()), path.stat().st_size)
    with zipfile.ZipFile(path) as archive:
        assert len(archive.infolist()) == count + 1
        assert all(
            info.compress_type == zipfile.ZIP_STORED and stat.S_ISREG(info.external_attr >> 16)
            for info in archive.infolist()
        )
    for array in restored.state.parameters.arrays().values():
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_main_loader_rejects_every_control_even_with_valid_checksums(tmp_path, variant):
    path = tmp_path / "control.tsa"
    api.save_ablation_forecaster(authored(variant), path)
    with pytest.raises(base.NeuralArtifactError, match="unsupported"):
        base.load_neural_forecaster(path)


def test_new_loader_rejects_actual_main_artifact(tmp_path):
    from turnscope.neural_forecast import (
        HierarchicalEventForecaster,
        NeuralForecastConfig,
        NeuralForecastState,
    )
    from turnscope.neural_forecast_math import FrozenNeuralParameters
    from turnscope.neural_forecast_train import NeuralTrainingResult

    reference = authored()
    old = reference.state
    config = NeuralForecastConfig(max_turn_tokens=4)
    architecture = config.architecture(old.vocabulary.size)
    tensors = tuple(
        FrozenNeuralTensor(name, shape, bytes(4 * math.prod(shape)))
        for name, shape in parameter_shapes(architecture).items()
    )
    parameters = FrozenNeuralParameters(architecture, tensors)
    result = NeuralTrainingResult(
        parameters,
        old.training.config,
        old.training.history,
        1,
        tuple((item.name, sha(item.data)) for item in tensors),
        (),
        sha("t"),
        sha("v"),
        old.vocabulary.digest,
        2,
        2,
        2,
        2,
        100_000_000,
        10_000_000,
        "authored",
        "authored",
        1,
    )
    state = NeuralForecastState(
        config,
        old.policy,
        old.data_limits,
        old.vocabulary,
        result,
        0.6,
        0.75,
        old.training_groups,
        old.validation_groups,
        old.policy_validation_groups,
        sha("p"),
        2,
        2,
        False,
    )
    model = HierarchicalEventForecaster(config=config)
    model._state = state
    path = tmp_path / "main.tsn"
    base.save_neural_forecaster(model, path)
    with pytest.raises(api.NeuralArtifactError, match="unsupported"):
        api.load_ablation_forecaster(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(format="other"),
        lambda m: m.update(numerical_version="current-turn.reset-after-zero-rzn.v1"),
        lambda m: m.update(training_version="main"),
        lambda m: m.update(initialization_version="main"),
        lambda m: m["state"]["summary"].update(variant="mean-word.v1"),
        lambda m: m["state"]["summary"]["config"].update(variant="mean-word.v1"),
        lambda m: m["state"]["summary"]["reference_architecture"].update(word_layers=2),
        lambda m: m["state"]["summary"].update(parameter_count=True),
        lambda m: m["state"]["summary"].update(threshold=True),
        lambda m: m["state"]["summary"].update(threshold=1.1),
        lambda m: m["state"]["summary"].update(policy_validation_balanced_accuracy=0.49),
        lambda m: m["state"]["summary"].update(probability_calibration_claimed=True),
        lambda m: m["state"]["summary"].update(probability_floor=0.0),
        lambda m: m["state"]["summary"].update(policy_reuses_model_validation=True),
        lambda m: m["state"]["summary"].update(policy_reuses_model_validation=0),
        lambda m: m["state"].update(policy_validation_groups=m["state"]["model_validation_groups"]),
        lambda m: m["state"]["summary"].update(
            policy_validation_partition_digest=m["state"]["summary"][
                "model_validation_partition_digest"
            ]
        ),
        lambda m: m["state"]["summary"].update(policy_validation_conversations=True),
        lambda m: m["state"]["summary"].update(training_prefixes=1),
        lambda m: m["state"]["vocabulary"].update(tokenizer_version="wrong-ucd"),
        lambda m: m["state"]["summary"]["config"].update(max_turn_tokens=3),
        lambda m: m["state"]["summary"]["training_config"].update(min_document_frequency=2),
        lambda m: m["state"]["summary"]["training_config"].update(max_features=1),
        lambda m: m["state"]["summary"]["history"][0].update(epoch=True),
        lambda m: m["state"]["summary"]["history"][0].update(optimizer_steps=0),
        lambda m: m["state"]["summary"].update(changed_parameter_names=["head.bias"]),
        lambda m: m["state"]["summary"]["main_initial_parameter_sha256"].update(
            {"embedding.weight": sha("different")}
        ),
        lambda m: m["state"]["summary"].update(initialization_workspace_bytes=1),
        lambda m: m["state"]["summary"].update(maximum_estimated_workspace_bytes=1),
        lambda m: m["state"]["summary"].update(estimated_affine_multiplications=1),
        lambda m: m["state"]["summary"].update(estimated_pooling_scalings=0),
        lambda m: m["state"]["summary"]["config"].update(max_inference_affine_multiplications=1),
        lambda m: m["state"]["summary"]["config"].update(max_inference_pooling_operations=1),
        lambda m: m["state"]["summary"].update(unknown="not permitted"),
    ],
)
def test_rechecksummed_semantic_mismatches_reject(saved, change):
    path, _ = saved
    manifest, members = unpack(path.read_bytes())
    change(manifest)
    path.write_bytes(repack(manifest, members))
    with pytest.raises(api.NeuralArtifactError):
        api.load_ablation_forecaster(path)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(shape=[True, 32]),
        lambda d: d.update(dtype="<f8"),
        lambda d: d.update(name="word.weight_hh_l0"),
        lambda d: d.update(data_bytes=True),
        lambda d: d.update(data_sha256="0" * 64),
        lambda d: d.update(member_sha256="0" * 64),
    ],
)
def test_descriptor_checks_precede_tensor_copies(saved, monkeypatch, change):
    path, _ = saved
    manifest, members = unpack(path.read_bytes())
    change(manifest["tensors"][-1])
    path.write_bytes(repack(manifest, members))
    sentinel = Mock(side_effect=AssertionError("tensor allocation happened before all admission"))
    monkeypatch.setattr(api, "FrozenNeuralTensor", sentinel)
    with pytest.raises(api.NeuralArtifactError):
        api.load_ablation_forecaster(path)
    sentinel.assert_not_called()


@pytest.mark.parametrize("kind", ["extra", "missing", "reorder", "header", "nan", "negative-pad"])
def test_inventory_header_and_rehashed_data_rejected(saved, kind):
    path, _ = saved
    manifest, members = unpack(path.read_bytes())
    if kind == "extra":
        members.append(("tensors/turn.weight_hh_l0.npy", b"inactive"))
    elif kind == "missing":
        members.pop()
    elif kind == "reorder":
        members[1], members[2] = members[2], members[1]
    else:
        name, raw = members[1]
        header = base._npy_header(tuple(manifest["tensors"][0]["shape"]))
        if kind == "header":
            raw = raw.replace(b"False", b"True ", 1)
        else:
            value = float("nan") if kind == "nan" else -0.0
            raw = raw[: len(header)] + struct.pack("<f", value) + raw[len(header) + 4 :]
        members[1] = name, raw
        manifest["tensors"][0]["member_sha256"] = sha(raw)
        manifest["tensors"][0]["data_sha256"] = sha(raw[len(header) :])
    path.write_bytes(repack(manifest, members))
    with pytest.raises(api.NeuralArtifactError):
        api.load_ablation_forecaster(path)


@pytest.mark.parametrize(
    "limits",
    [
        base.NeuralArtifactLimits(max_file_bytes=1),
        base.NeuralArtifactLimits(max_manifest_bytes=1),
        base.NeuralArtifactLimits(max_manifest_nodes=1),
        base.NeuralArtifactLimits(max_manifest_depth=1),
        base.NeuralArtifactLimits(max_parameters=1),
        base.NeuralArtifactLimits(max_parameter_magnitude=0.0001),
    ],
)
def test_load_limits_reject_without_changing_saved_policy(saved, limits):
    path, _ = saved
    before = path.read_bytes()
    with pytest.raises(api.NeuralArtifactError):
        api.load_ablation_forecaster(path, limits=limits)
    assert path.read_bytes() == before


def test_more_restrictive_sufficient_limits_do_not_change_identity(saved):
    path, source = saved
    model = api.load_ablation_forecaster(
        path,
        limits=base.NeuralArtifactLimits(
            max_parameters=source.state.parameters.parameter_count,
            max_parameter_magnitude=0.5,
            max_file_bytes=path.stat().st_size,
        ),
    )
    assert model.digest == source.digest
    assert model.state.parameters.limits == source.state.parameters.limits


def test_valid_earliest_best_history_and_changed_inventory_roundtrip(tmp_path):
    history = (
        NeuralEpoch(1, 0.8, 0.8, 1, 1.0),
        NeuralEpoch(2, 0.7, 0.6, 1, 1.0),
        NeuralEpoch(3, 0.6, 0.6, 1, 1.0),
    )
    source = authored(history=history)
    result = source.state.training
    initial = tuple(
        (name, sha("original") if name == "head.bias" else digest)
        for name, digest in result.initial_parameter_sha256
    )
    main = tuple(
        (name, dict(initial).get(name, digest))
        for name, digest in result.main_initial_parameter_sha256
    )
    source._state = replace(
        source.state,
        training=replace(
            result,
            initial_parameter_sha256=initial,
            main_initial_parameter_sha256=main,
            changed_parameter_names=("head.bias",),
        ),
    )
    path = tmp_path / "history.tsa"
    api.save_ablation_forecaster(source, path)
    loaded = api.load_ablation_forecaster(path)
    assert loaded.state.training.selected_epoch == 2
    assert loaded.state.training.changed_parameter_names == ("head.bias",)
    manifest, members = unpack(path.read_bytes())
    manifest["state"]["summary"]["selected_epoch"] = 3
    path.write_bytes(repack(manifest, members))
    with pytest.raises(api.NeuralArtifactError, match="earliest"):
        api.load_ablation_forecaster(path)


def blocked_load(path, digest):
    script = """
import importlib.abc, sys, pickle
import numpy
def forbidden(*args, **kwargs):
    raise AssertionError('generic deserialization forbidden')
numpy.load = pickle.loads = pickle.load = forbidden
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch must not be imported')
sys.meta_path.insert(0, Forbidden())
from turnscope.neural_ablation_artifact import load_ablation_forecaster
from turnscope.neural_ablation_math import infer_ablation_turns
model = load_ablation_forecaster(sys.argv[1])
assert model.digest == sys.argv[2]
assert len(infer_ablation_turns(model.state.parameters, ((2,), (2,))).probabilities) == 2
assert 'torch' not in sys.modules
print(model.digest)
"""
    completed = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script, str(path), digest],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.stdout.strip() == digest


def test_fresh_child_load_forbids_torch_and_generic_deserializers(saved):
    path, model = saved
    blocked_load(path, model.digest)


def tiny_sources(split):
    result = []
    for index, label in enumerate((True, False)):
        texts = [
            "red bright" if label else "blue calm",
            "red answer" if label else "blue answer",
            "future-only",
        ]
        turns = [
            Utterance(
                str(i),
                "person",
                text,
                datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=i),
                metadata={"event": label and i == 2},
            )
            for i, text in enumerate(texts)
        ]
        result.append(
            Conversation(
                f"{split}-{index}", turns, metadata={"forecast_groups": [f"{split}-{index}"]}
            )
        )
    return result


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_actual_cpu_fit_save_reopen_then_torch_forbidden_child(tmp_path, variant):
    torch = pytest.importorskip(
        "torch", reason="actual CPU fit requires optional Torch, not counted when skipped"
    )
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model = AblationEventForecaster(
            config=AblationForecastConfig(variant, max_turn_tokens=4),
            training_config=NeuralTrainingConfig(epochs=1, batch_conversations=2, seed=17),
        )
        model.fit(
            tiny_sources("train"),
            tiny_sources("validation"),
            policy_validation=tiny_sources("policy"),
        )
        assert model.state.training.changed_parameter_names
        path = tmp_path / "trained.tsa"
        api.save_ablation_forecaster(model, path)
        restored = api.load_ablation_forecaster(path)
        assert restored.state == model.state
        assert restored.evaluate(tiny_sources("test")) == model.evaluate(tiny_sources("test"))
        blocked_load(path, model.digest)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize(
    "change",
    [
        lambda model: object.__setattr__(model.state, "threshold", True),
        lambda model: object.__setattr__(model.state, "_identity", "0" * 64),
        lambda model: object.__setattr__(model.state, "training_groups", ["not immutable"]),
        lambda model: object.__setattr__(
            model.state.training, "history", [NeuralEpoch(1, 1, 1, 1, 1)]
        ),
        lambda model: object.__setattr__(
            model.state.training,
            "initial_parameter_sha256",
            tuple(reversed(model.state.training.initial_parameter_sha256)),
        ),
        lambda model: object.__setattr__(model.state.training, "vocabulary_digest", "0" * 64),
        lambda model: object.__setattr__(
            model.state.parameters, "tensors", model.state.parameters.tensors[:-1]
        ),
    ],
)
def test_inconsistent_manually_mutated_source_rejected_before_tempfile(
    tmp_path, monkeypatch, change
):
    model = authored()
    change(model)
    create = Mock(side_effect=AssertionError("no publication preparation allowed"))
    monkeypatch.setattr(api.tempfile, "NamedTemporaryFile", create)
    with pytest.raises(api.NeuralArtifactError):
        api.save_ablation_forecaster(model, tmp_path / "invalid.tsa")
    create.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_exclusive_output_and_explicit_overwrite_preserve_hardlink_sibling(saved):
    path, source = saved
    sibling = path.parent / "sibling.tsa"
    os.link(path, sibling)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        api.save_ablation_forecaster(source, path)
    source._state = replace(source.state, threshold=0.75)
    result = api.save_ablation_forecaster(source, path, overwrite=True)
    assert result.cleanup_warning is None
    assert sibling.read_bytes() == original
    assert not os.path.samefile(sibling, path)
    assert api.load_ablation_forecaster(path).state.threshold == 0.75


@pytest.mark.parametrize("overwrite", [False, True])
def test_output_appears_during_validation_not_clobbered(tmp_path, monkeypatch, overwrite):
    path = tmp_path / "race.tsa"
    encode = api._encode

    def concurrent(model, limits):
        raw = encode(model, limits)
        path.write_bytes(b"other writer")
        return raw

    monkeypatch.setattr(api, "_encode", concurrent)
    with pytest.raises((api.NeuralArtifactError, FileExistsError)):
        api.save_ablation_forecaster(authored(), path, overwrite=overwrite)
    assert path.read_bytes() == b"other writer"
    assert list(tmp_path.iterdir()) == [path]


def test_existing_output_changes_during_overwrite_is_detected(saved, monkeypatch):
    path, source = saved
    encode = api._encode

    def concurrent(model, limits):
        raw = encode(model, limits)
        path.write_bytes(b"another writer replaces this output")
        return raw

    monkeypatch.setattr(api, "_encode", concurrent)
    with pytest.raises(api.NeuralArtifactError, match="changed"):
        api.save_ablation_forecaster(source, path, overwrite=True)
    assert path.read_bytes() == b"another writer replaces this output"


def test_postpublication_cleanup_failure_returns_success_warning(tmp_path, monkeypatch):
    unlink = Path.unlink

    def cannot_unlink(path, *args, **kwargs):
        if path.name.startswith(".turnscope-ablation-"):
            raise OSError("private platform detail")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", cannot_unlink)
    path = tmp_path / "done.tsa"
    source = authored()
    result = api.save_ablation_forecaster(source, path)
    assert (
        result.cleanup_warning
        == "Artifact was published; a private temporary file could not be removed."
    )
    assert api.load_ablation_forecaster(path).digest == source.digest
    assert len(list(tmp_path.glob(".turnscope-ablation-*"))) == 1


def test_prepublication_failure_retains_original_error_when_cleanup_also_fails(
    tmp_path, monkeypatch
):
    def failed(*args, **kwargs):
        raise OSError("original sync failure")

    monkeypatch.setattr(api.os, "fsync", failed)
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=OSError("cleanup failure")))
    target = tmp_path / "never-published.tsa"
    with pytest.raises(OSError, match="original sync failure"):
        api.save_ablation_forecaster(authored(), target)
    assert not target.exists()


def test_short_temporary_write_rejected_and_cleaned(tmp_path, monkeypatch):
    original = api.tempfile.NamedTemporaryFile

    class Short:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            self.name = self.inner.name
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def write(self, raw):
            return self.inner.write(raw[:-1])

    monkeypatch.setattr(api.tempfile, "NamedTemporaryFile", Short)
    with pytest.raises(OSError, match="completely written"):
        api.save_ablation_forecaster(authored(), tmp_path / "short.tsa")
    assert list(tmp_path.iterdir()) == []


def test_load_rejects_replaced_file_between_lstat_and_open(saved, monkeypatch):
    path, _ = saved
    real_open = base.os.open
    replacement = path.with_name("replacement")
    replacement.write_bytes(path.read_bytes())

    def raced_open(name, flags, *args, **kwargs):
        if Path(name) == path:
            os.replace(replacement, path)
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(base.os, "open", raced_open)
    with pytest.raises(api.NeuralArtifactError, match="changed"):
        api.load_ablation_forecaster(path)


def test_symlink_and_directory_boundaries(saved):
    path, source = saved
    with pytest.raises(api.NeuralArtifactError, match="regular"):
        api.load_ablation_forecaster(path.parent)
    with pytest.raises(api.NeuralArtifactError, match="regular"):
        api.save_ablation_forecaster(source, path.parent, overwrite=True)
    link = path.parent / "symbolic.tsa"
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("host does not grant symlink creation")
    with pytest.raises(api.NeuralArtifactError, match="symlink"):
        api.load_ablation_forecaster(link)
    with pytest.raises(api.NeuralArtifactError, match="symlink"):
        api.save_ablation_forecaster(source, link, overwrite=True)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"format":1,"format":2}',
        b'{"format":NaN}',
        b'{"format":1e9999}',
        b'{"format":9223372036854775808}',
        b'{"format":"\xff"}',
        b'{"format":"\\ud800"}',
        b'{"format": 1}',
        b"[" * 18 + b"0" + b"]" * 18,
    ],
)
def test_noncanonical_or_unbounded_json_rejected_before_restore(tmp_path, monkeypatch, raw):
    path = tmp_path / "json.tsa"
    path.write_bytes(base._archive([("manifest.json", raw)], base.NeuralArtifactLimits()))
    restore = Mock(side_effect=AssertionError("not a parsed closed manifest"))
    monkeypatch.setattr(api, "_restore", restore)
    with pytest.raises(api.NeuralArtifactError):
        api.load_ablation_forecaster(path)
    restore.assert_not_called()


@pytest.mark.parametrize("mutation", ["duplicate", "compressed", "trailing", "crc"])
def test_unsafe_zip_container_rejected_through_shared_reader(saved, mutation):
    path, _ = saved
    raw = path.read_bytes()
    if mutation == "duplicate":
        _, members = unpack(raw)
        raw = base._archive([*members, members[-1]], base.NeuralArtifactLimits())
    elif mutation == "compressed":
        import io

        _, members = unpack(raw)
        destination = io.BytesIO()
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in members:
                archive.writestr(name, data)
        raw = destination.getvalue()
    elif mutation == "trailing":
        raw += b"unindexed"
    else:
        raw = raw[:100] + bytes((raw[100] ^ 1,)) + raw[101:]
    path.write_bytes(raw)
    with pytest.raises(api.NeuralArtifactError):
        api.load_ablation_forecaster(path)


def test_load_type_and_save_boolean_contract(saved, tmp_path):
    path, model = saved
    with pytest.raises(api.NeuralArtifactError):
        api.load_ablation_forecaster(path, limits={})
    with pytest.raises(api.NeuralArtifactError):
        api.save_ablation_forecaster(model, tmp_path / "other.tsa", overwrite=1)
    with pytest.raises(api.NeuralArtifactError):
        api.save_ablation_forecaster(object(), tmp_path / "other.tsa")
    with pytest.raises(api.NeuralArtifactError):
        api.save_ablation_forecaster(
            AblationEventForecaster(config=model.state.config), tmp_path / "other.tsa"
        )


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_full_canonical_initializer_must_fit_saved_numeric_parameter_limit(tmp_path, variant):
    source = authored(variant)
    path = tmp_path / "initializer.tsa"
    api.save_ablation_forecaster(source, path)
    manifest, members = unpack(path.read_bytes())
    manifest["state"]["summary"]["numeric_limits"]["max_parameters"] = (
        source.state.parameters.parameter_count
    )
    path.write_bytes(repack(manifest, members))
    with pytest.raises(api.NeuralArtifactError, match="initializer"):
        api.load_ablation_forecaster(path)


def test_supported_large_affine_configuration_is_not_artificially_capped(tmp_path):
    source = authored()
    settings = replace(source.state.training.config, max_total_affine_multiplications=10**14)
    source._state = replace(
        source.state,
        training=replace(
            source.state.training, config=settings, estimated_affine_multiplications=10**13 + 1
        ),
    )
    path = tmp_path / "declared-large-budget.tsa"
    api.save_ablation_forecaster(source, path)
    loaded = api.load_ablation_forecaster(path)
    assert loaded.state.training.estimated_affine_multiplications == 10**13 + 1


@pytest.mark.parametrize(
    "field,claimed",
    [
        ("estimated_affine_multiplications", 33_280),
        ("estimated_pooling_additions", 512),
        ("estimated_pooling_scalings", 2048),
    ],
)
def test_minimum_work_counts_extra_eligible_prefix_turns(saved, field, claimed):
    path, _ = saved
    manifest, members = unpack(path.read_bytes())
    summary = manifest["state"]["summary"]
    summary.update(training_prefixes=4, model_validation_prefixes=4)
    # Two conversations, four eligible prefixes => at least six real turns in
    # each split. Old conv*min_turns only charged four and admitted these values.
    summary[field] = claimed
    path.write_bytes(repack(manifest, members))
    with pytest.raises(api.NeuralArtifactError, match="minimum support"):
        api.load_ablation_forecaster(path)


def test_policy_budget_counts_extra_eligible_prefix_turns(saved):
    path, _ = saved
    manifest, members = unpack(path.read_bytes())
    summary = manifest["state"]["summary"]
    summary["policy_validation_prefixes"] = 4
    summary["config"]["max_inference_affine_multiplications"] = 4 * 2080
    path.write_bytes(repack(manifest, members))
    with pytest.raises(api.NeuralArtifactError, match="policy validation support"):
        api.load_ablation_forecaster(path)
