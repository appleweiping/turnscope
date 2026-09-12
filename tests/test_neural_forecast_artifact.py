"""Authored inference data: no training, checkpoint download, or Torch dependency."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import stat
import struct
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from turnscope import neural_forecast_artifact as artifact
from turnscope.neural_forecast import (
    HierarchicalEventForecaster,
    NeuralForecastConfig,
    NeuralForecastState,
)
from turnscope.neural_forecast_data import SequenceLimits, SequencePolicy
from turnscope.neural_forecast_math import (
    FrozenNeuralParameters,
    FrozenNeuralTensor,
    NeuralNumericLimits,
    infer_encoded_turns,
    parameter_shapes,
)
from turnscope.neural_forecast_train import NeuralEpoch, NeuralTrainingConfig, NeuralTrainingResult
from turnscope.neural_token_data import SequenceVocabulary


def sha(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def model(*, word_layers=1, turn_layers=1, reuse=True, histories=None):
    config = NeuralForecastConfig(
        embedding_dim=2,
        word_hidden=2,
        turn_hidden=3,
        word_layers=word_layers,
        turn_layers=turn_layers,
        max_turn_tokens=4,
    )
    vocabulary = SequenceVocabulary(("alpha", "beta"), (2, 1), 2, 4, "head")
    architecture = config.architecture(vocabulary.size)
    tensors = tuple(
        FrozenNeuralTensor(name, shape, struct.pack("<f", 0.125) * math.prod(shape))
        for name, shape in parameter_shapes(architecture).items()
    )
    parameters = FrozenNeuralParameters(architecture, tensors, NeuralNumericLimits())
    initial = tuple((tensor.name, sha(tensor.data)) for tensor in tensors)
    records = histories or (NeuralEpoch(1, 0.7, 0.7, 1, 0.1),)
    training_config = NeuralTrainingConfig(epochs=len(records), patience=2)
    result = NeuralTrainingResult(
        parameters=parameters,
        config=training_config,
        history=records,
        selected_epoch=min(records, key=lambda record: record.validation_loss).epoch,
        initial_parameter_sha256=initial,
        changed_parameter_names=(),
        training_partition_digest=sha("training"),
        validation_partition_digest=sha("validation"),
        vocabulary_digest=vocabulary.digest,
        training_conversations=2,
        validation_conversations=2,
        training_prefixes=3,
        validation_prefixes=2,
        maximum_estimated_workspace_bytes=48 * architecture.parameter_count + 4096,
        estimated_affine_multiplications=10000 * len(records),
        torch_version="authored-fixture",
        numpy_version="authored-fixture",
        torch_threads=1,
    )
    train_groups = frozenset((sha("training-a"), sha("training-b")))
    val_groups = frozenset((sha("validation-a"), sha("validation-b")))
    policy_groups = val_groups if reuse else frozenset((sha("policy-a"), sha("policy-b")))
    state = NeuralForecastState(
        config,
        SequencePolicy(),
        SequenceLimits(),
        vocabulary,
        result,
        0.6,
        0.75,
        train_groups,
        val_groups,
        policy_groups,
        sha("validation" if reuse else "policy"),
        2,
        2,
        reuse,
    )
    value = HierarchicalEventForecaster(config=config, training_config=training_config)
    value._state = state
    return value


def unpack(raw):
    members = [
        (name, bytes(value))
        for name, value in artifact._members(raw, artifact.NeuralArtifactLimits()).items()
    ]
    manifest = json.loads(members[0][1])
    return manifest, members


def repack(manifest, members, *, rehash=True):
    limits = artifact.NeuralArtifactLimits()
    if rehash:
        # Re-sign the declared model identity as well, except a test that
        # deliberately substitutes that field. Semantic probes must not pass
        # merely because they forgot to recompute an ordinary checksum.
        original = json.loads(members[0][1])
        if manifest["model_digest"] == original["model_digest"]:
            declaration = {
                key: value for key, value in manifest["state"].items() if key != "vocabulary"
            }
            manifest["model_digest"] = sha(artifact._canonical(declaration, limits))
        manifest.pop("sha256", None)
        manifest["sha256"] = sha(artifact._canonical(manifest, limits))
    members[0] = ("manifest.json", artifact._canonical(manifest, limits))
    return artifact._archive(members, limits)


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / "model.tsn"
    source = model()
    artifact.save_neural_forecaster(source, path)
    return path, source


@pytest.mark.parametrize("word_layers,turn_layers", [(1, 1), (1, 2), (2, 1), (2, 2)])
@pytest.mark.parametrize("reuse", [False, True])
def test_roundtrip_exact_full_state_and_inference(tmp_path, word_layers, turn_layers, reuse):
    source = model(word_layers=word_layers, turn_layers=turn_layers, reuse=reuse)
    target = tmp_path / "model.tsn"
    result = artifact.save_neural_forecaster(source, target)
    loaded = artifact.load_neural_forecaster(target)
    assert loaded.digest == source.digest
    assert loaded.state.summary() == source.state.summary()
    assert loaded.state.vocabulary.to_dict() == source.state.vocabulary.to_dict()
    assert loaded.state.training_groups == source.state.training_groups
    assert loaded.state.policy_validation_groups == source.state.policy_validation_groups
    assert loaded.state.parameters.limits == source.state.parameters.limits
    turns = ((3, 2), (4, 3, 2))
    assert infer_encoded_turns(loaded.state.parameters, turns) == infer_encoded_turns(
        source.state.parameters, turns
    )
    assert result == artifact.NeuralSaveResult(
        str(target), sha(target.read_bytes()), target.stat().st_size
    )
    again = tmp_path / "again.tsn"
    artifact.save_neural_forecaster(loaded, again)
    assert again.read_bytes() == target.read_bytes()
    with zipfile.ZipFile(target) as archive:
        assert len(archive.infolist()) == len(source.state.parameters.tensors) + 1
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert stat.S_ISREG(info.external_attr >> 16)


def test_load_in_fresh_process_forbids_torch_and_generic_deserializers(saved):
    path, source = saved
    script = """
import importlib.abc, sys, pickle
import numpy
def forbidden(*args, **kwargs):
    raise AssertionError('generic deserializer is forbidden')
numpy.load = pickle.load = pickle.loads = forbidden
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Torch import is forbidden')
sys.meta_path.insert(0, Forbidden())
from turnscope.neural_forecast_artifact import load_neural_forecaster
value = load_neural_forecaster(sys.argv[1])
assert 'torch' not in sys.modules
print(value.digest)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout.strip() == source.digest


def test_public_model_save_load_wrappers_and_exports(tmp_path):
    import turnscope

    source = model()
    target = tmp_path / "public.tsn"
    result = source.save(target)
    assert isinstance(result, turnscope.NeuralSaveResult)
    loaded = turnscope.HierarchicalEventForecaster.load(target)
    assert loaded.digest == source.digest
    assert turnscope.load_neural_forecaster(target).digest == source.digest
    assert turnscope.save_neural_forecaster is artifact.save_neural_forecaster
    with pytest.raises(FileExistsError):
        source.save(target)
    assert source.save(target, overwrite=True).sha256 == result.sha256
    with pytest.raises(turnscope.NeuralArtifactError):
        turnscope.HierarchicalEventForecaster.load(
            target, limits=turnscope.NeuralArtifactLimits(max_file_bytes=1)
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(format="other"),
        lambda m: m.update(numerical_version="other"),
        lambda m: m.update(training_version="other"),
        lambda m: m.update(initialization_version="other"),
        lambda m: m.update(model_digest="0" * 64),
        lambda m: m["state"]["summary"].update(format="other"),
        lambda m: m["state"]["summary"].update(probability_floor=0),
        lambda m: m["state"]["summary"].update(probability_calibration_claimed=True),
        lambda m: m["state"]["summary"].update(threshold_rule="probability > threshold"),
        lambda m: m["state"]["summary"].update(threshold_objective="test selection"),
        lambda m: m["state"]["summary"].update(threshold=True),
        lambda m: m["state"]["summary"].update(threshold=-0.1),
        lambda m: m["state"]["summary"].update(policy_validation_balanced_accuracy=0.49),
        lambda m: m["state"]["summary"].update(parameter_digest="0" * 64),
        lambda m: m["state"]["summary"].update(vocabulary_digest="0" * 64),
        lambda m: m["state"]["summary"].update(vocabulary_size=6),
        lambda m: m["state"]["summary"].update(parameter_count=1),
        lambda m: m["state"]["summary"]["config"].update(extra=1),
        lambda m: m["state"]["summary"]["config"].update(word_layers=True),
        lambda m: m["state"]["summary"]["config"].update(max_turn_tokens=8),
        lambda m: m["state"]["summary"]["config"].update(max_inference_affine_multiplications=1),
        lambda m: m["state"]["summary"]["data_limits"].update(max_source_turns=1),
        lambda m: m["state"]["summary"]["numeric_limits"].update(max_token_positions=1),
        lambda m: m["state"]["summary"]["numeric_limits"].update(max_workspace_bytes=1),
        lambda m: m["state"]["summary"]["training_config"].update(min_document_frequency=2),
        lambda m: m["state"]["summary"]["training_config"].update(max_features=1),
        lambda m: m["state"]["vocabulary"].update(tokenizer_version="foreign-UCD"),
        lambda m: m["state"]["vocabulary"].update(documents=3),
        lambda m: m["state"]["summary"].update(training_conversations=True),
        lambda m: m["state"]["summary"].update(training_prefixes=1),
        lambda m: m["state"]["summary"].update(training_partition_digest="bad"),
        lambda m: m["state"].update(training_groups=m["state"]["model_validation_groups"]),
        lambda m: m["state"].update(policy_validation_groups=m["state"]["training_groups"]),
        lambda m: m["state"]["summary"].update(policy_reuses_model_validation=1),
        lambda m: m["state"]["summary"].update(policy_reuses_model_validation=False),
        lambda m: m["state"]["summary"].update(policy_validation_prefixes=3),
        lambda m: m["state"]["summary"].update(policy_validation_partition_digest="0" * 64),
        lambda m: m["state"]["summary"]["history"][0].update(epoch=2),
        lambda m: m["state"]["summary"]["history"][0].update(training_loss=-1),
        lambda m: m["state"]["summary"]["history"][0].update(optimizer_steps=0),
        lambda m: m["state"]["summary"]["history"][0].update(optimizer_steps=3),
        lambda m: m["state"]["summary"]["history"][0].update(maximum_unclipped_gradient_norm=-1),
        lambda m: m["state"]["summary"].update(selected_epoch=True),
        lambda m: m["state"]["summary"].update(changed_parameter_names=["embedding.weight"]),
        lambda m: m["state"]["summary"]["initial_parameter_sha256"].update(extra="0" * 64),
        lambda m: m["state"]["summary"].update(maximum_estimated_workspace_bytes=1),
        lambda m: m["state"]["summary"].update(estimated_affine_multiplications=0),
        lambda m: m["state"]["summary"].update(estimated_affine_multiplications=1),
        lambda m: m["state"]["summary"].update(torch_threads=True),
        lambda m: m["state"]["summary"].update(torch_version="bad\nversion"),
        lambda m: m["tensors"][0].update(shape=[True, 2]),
        lambda m: m["tensors"][0].update(dtype="object"),
        lambda m: m["tensors"][0].update(name="unknown"),
        lambda m: m["tensors"][0].update(member="../outside.npy"),
        lambda m: m["tensors"][0].update(data_bytes=4),
        lambda m: m["tensors"][0].update(data_sha256="0" * 64),
        lambda m: m["tensors"][0].update(member_sha256="0" * 64),
    ],
)
def test_rehashed_but_inconsistent_artifact_is_rejected(saved, change):
    path, _ = saved
    manifest, members = unpack(path.read_bytes())
    change(manifest)
    path.write_bytes(repack(manifest, members))
    with pytest.raises(artifact.NeuralArtifactError):
        artifact.load_neural_forecaster(path)


def test_earliest_best_and_stopping_are_validated_not_just_rehashed(tmp_path):
    source = model(histories=(NeuralEpoch(1, 0.8, 0.6, 1, 0.1), NeuralEpoch(2, 0.7, 0.6, 1, 0.1)))
    path = tmp_path / "model"
    artifact.save_neural_forecaster(source, path)
    manifest, members = unpack(path.read_bytes())
    for change in (
        lambda s: s.update(selected_epoch=2),
        lambda s: s["training_config"].update(epochs=4),
        lambda s: s.update(estimated_affine_multiplications=19999),
        lambda s: s["history"].append({**s["history"][1], "epoch": 3}),
    ):
        modified = copy.deepcopy(manifest)
        change(modified["state"]["summary"])
        path.write_bytes(repack(modified, members[:]))
        with pytest.raises(artifact.NeuralArtifactError):
            artifact.load_neural_forecaster(path)


def test_real_early_stop_and_changed_hash_inventory_roundtrip(tmp_path):
    source = model(histories=tuple(NeuralEpoch(i, 0.8, 0.6, 1, 0.1) for i in (1, 2, 3)))
    training = source.state.training
    training = replace(
        training,
        config=replace(training.config, epochs=8),
        estimated_affine_multiplications=80000,
        initial_parameter_sha256=(
            ("embedding.weight", "0" * 64),
            *training.initial_parameter_sha256[1:],
        ),
        changed_parameter_names=("embedding.weight",),
    )
    source._state = replace(source.state, training=training)
    path = tmp_path / "model"
    artifact.save_neural_forecaster(source, path)
    assert artifact.load_neural_forecaster(path).digest == source.digest
    manifest, members = unpack(path.read_bytes())
    manifest["state"]["summary"]["history"].append(
        {**manifest["state"]["summary"]["history"][-1], "epoch": 4}
    )
    path.write_bytes(repack(manifest, members))
    with pytest.raises(artifact.NeuralArtifactError, match="continues"):
        artifact.load_neural_forecaster(path)


@pytest.mark.parametrize("replacement", [b"not a zip", b"PK\x05\x06" + b"\0" * 18])
def test_nonarchive_rejected(tmp_path, replacement):
    path = tmp_path / "invalid"
    path.write_bytes(replacement)
    with pytest.raises(artifact.NeuralArtifactError):
        artifact.load_neural_forecaster(path)


@pytest.mark.parametrize(
    "mode",
    [
        "compression",
        "duplicate",
        "symlink",
        "encrypted",
        "extra",
        "comment",
        "prefix",
        "trailer",
        "overlap",
        "crc",
        "name",
        "count",
    ],
)
def test_unsafe_zip_metadata_and_structure_rejected(saved, mode):
    path, _ = saved
    raw = path.read_bytes()
    _, members = unpack(raw)
    if mode in ("compression", "duplicate", "symlink", "extra", "comment"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, data in members:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = (
                    (stat.S_IFLNK if mode == "symlink" else stat.S_IFREG) | 0o600
                ) << 16
                info.compress_type = (
                    zipfile.ZIP_DEFLATED if mode == "compression" else zipfile.ZIP_STORED
                )
                if mode == "extra":
                    info.extra = b"\x01\x00\x00\x00"
                archive.writestr(info, data)
            if mode == "duplicate":
                with pytest.warns(UserWarning):
                    archive.writestr("manifest.json", members[0][1])
            if mode == "comment":
                archive.comment = b"ignored data"
        raw = buffer.getvalue()
    elif mode == "prefix":
        raw = b"hidden prefix" + raw
    elif mode == "trailer":
        raw += b"hidden trailer"
    else:
        value = bytearray(raw)
        directory = artifact._END.unpack_from(raw, len(raw) - artifact._END.size)[6]
        if mode == "encrypted":
            struct.pack_into("<H", value, 6, 1)
            struct.pack_into("<H", value, directory + 8, 1)
        elif mode == "overlap":
            second = directory + artifact._CENTRAL.size + len("manifest.json")
            struct.pack_into("<L", value, second + 42, 0)
        elif mode == "crc":
            value[artifact._LOCAL.size + len("manifest.json")] ^= 1
        elif mode == "name":
            value[directory + artifact._CENTRAL.size] = ord("/")
        else:
            struct.pack_into("<H", value, len(value) - artifact._END.size + 8, 65535)
        raw = bytes(value)
    path.write_bytes(raw)
    with pytest.raises(artifact.NeuralArtifactError):
        artifact.load_neural_forecaster(path)


@pytest.mark.parametrize("change", ["header", "nan", "magnitude", "trailing", "object"])
def test_rehashed_tensor_still_requires_exact_header_and_finite_float32(saved, change):
    path, _ = saved
    manifest, members = unpack(path.read_bytes())
    name, raw = members[1]
    header_length = len(artifact._npy_header(tuple(manifest["tensors"][0]["shape"])))
    value = bytearray(raw)
    if change == "header":
        value[8] ^= 1
    elif change == "object":
        value = bytearray(raw.replace(b"<f4", b"|O4"))
    elif change == "nan":
        value[header_length : header_length + 4] = struct.pack("<f", float("nan"))
    elif change == "magnitude":
        value[header_length : header_length + 4] = struct.pack("<f", 1e7)
    else:
        value += b"extra"
    members[1] = name, bytes(value)
    manifest["tensors"][0]["member_sha256"] = sha(bytes(value))
    manifest["tensors"][0]["data_sha256"] = sha(bytes(value[header_length:]))
    path.write_bytes(repack(manifest, members))
    with pytest.raises(artifact.NeuralArtifactError):
        artifact.load_neural_forecaster(path)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":1e999}',
        b'{"a":9223372036854775808}',
        b'{"a":"\\ud800"}',
        b'{"a":' + b"1" * 129 + b"}",
        b'{"a":}',
        b"\xff",
    ],
)
def test_bad_json_never_bypasses_closed_parse(raw):
    with pytest.raises(artifact.NeuralArtifactError):
        artifact._parse_manifest(raw, artifact.NeuralArtifactLimits())


def test_manifest_size_depth_and_pending_nodes_are_admitted_before_decoder_or_encoder(monkeypatch):
    loads, dumps = (
        Mock(side_effect=AssertionError("decoder called")),
        Mock(side_effect=AssertionError("encoder called")),
    )
    monkeypatch.setattr(artifact.json, "loads", loads)
    monkeypatch.setattr(artifact.json, "dumps", dumps)
    for raw, limits in (
        (b"[0,0,0]", artifact.NeuralArtifactLimits(max_manifest_nodes=3)),
        (b"[[[0]]]", artifact.NeuralArtifactLimits(max_manifest_depth=2)),
        (b" " * 20, artifact.NeuralArtifactLimits(max_manifest_bytes=10)),
    ):
        with pytest.raises(artifact.NeuralArtifactError):
            artifact._parse_manifest(raw, limits)
    for value, limits in (
        ([[0, 0], [0, 0]], artifact.NeuralArtifactLimits(max_manifest_nodes=5)),
        ({"private": "\0" * 4}, artifact.NeuralArtifactLimits(max_manifest_bytes=20)),
        ("é" * 11, artifact.NeuralArtifactLimits(max_manifest_bytes=20)),
        ([[[0]]], artifact.NeuralArtifactLimits(max_manifest_depth=2)),
        ({1: "value"}, artifact.NeuralArtifactLimits()),
        (object(), artifact.NeuralArtifactLimits()),
    ):
        with pytest.raises(artifact.NeuralArtifactError):
            artifact._canonical(value, limits)
    loads.assert_not_called()
    dumps.assert_not_called()


def test_tightened_limits_reject_instead_of_modifying_saved_policy(saved, monkeypatch):
    path, source = saved
    original = path.read_bytes()
    for limits in (
        artifact.NeuralArtifactLimits(max_file_bytes=10),
        artifact.NeuralArtifactLimits(max_manifest_bytes=10),
        artifact.NeuralArtifactLimits(max_parameters=1),
        artifact.NeuralArtifactLimits(max_parameter_magnitude=0.1),
    ):
        with pytest.raises(artifact.NeuralArtifactError):
            artifact.load_neural_forecaster(path, limits=limits)
    assert path.read_bytes() == original
    restored = artifact.load_neural_forecaster(
        path, limits=artifact.NeuralArtifactLimits(max_parameter_magnitude=0.125)
    )
    assert restored.state.parameters.limits.max_parameter_magnitude == 1e6
    assert restored.digest == source.digest
    forbidden = Mock(side_effect=AssertionError("parameter allocation must follow admission"))
    monkeypatch.setattr(artifact, "FrozenNeuralParameters", forbidden)
    with pytest.raises(artifact.NeuralArtifactError, match="parameter budget"):
        artifact.load_neural_forecaster(
            path, limits=artifact.NeuralArtifactLimits(max_parameters=1)
        )
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_file_bytes", True),
        ("max_file_bytes", 64 * 1024 * 1024 + 1),
        ("max_manifest_bytes", 0),
        ("max_manifest_nodes", 200001),
        ("max_manifest_depth", 17),
        ("max_parameters", 16000001),
        ("max_parameter_magnitude", 0),
        ("max_parameter_magnitude", True),
        ("max_parameter_magnitude", 10**400),
    ],
)
def test_limits_typed_and_bounded(name, value):
    with pytest.raises(artifact.NeuralArtifactError):
        artifact.NeuralArtifactLimits(**{name: value})


def test_save_rejects_inconsistent_manually_constructed_state_before_publication(tmp_path):
    path = tmp_path / "model"
    for transform in (
        lambda s: replace(s, threshold=True),
        lambda s: replace(s, training_groups=s.validation_groups),
        lambda s: replace(s, training=replace(s.training, vocabulary_digest="0" * 64)),
        lambda s: replace(
            s, training=replace(s.training, changed_parameter_names=["embedding.weight"])
        ),
        lambda s: replace(
            s,
            training=replace(
                s.training,
                initial_parameter_sha256=s.training.initial_parameter_sha256
                + s.training.initial_parameter_sha256[:1],
            ),
        ),
        lambda s: replace(s, training_groups=tuple(s.training_groups)),
    ):
        source = model()
        source._state = transform(source.state)
        with pytest.raises(artifact.NeuralArtifactError):
            artifact.save_neural_forecaster(source, path)
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []


def test_save_default_exclusive_explicit_atomic_replace_and_conflict(saved, monkeypatch):
    path, source = saved
    old = path.read_bytes()
    with pytest.raises(FileExistsError):
        artifact.save_neural_forecaster(source, path)
    assert path.read_bytes() == old
    source._state = replace(source.state, threshold=0.7)
    result = artifact.save_neural_forecaster(source, path, overwrite=True)
    assert result.cleanup_warning is None
    assert artifact.load_neural_forecaster(path).digest == source.digest
    replacement = path.read_bytes()

    def conflict(*args):
        raise OSError("replace refused")

    monkeypatch.setattr(artifact.os, "replace", conflict)
    with pytest.raises(OSError, match="replace refused"):
        artifact.save_neural_forecaster(source, path, overwrite=True)
    assert path.read_bytes() == replacement
    assert not list(path.parent.glob(".turnscope-neural-*"))


def test_postpublication_cleanup_warning_is_success_and_owned_temp_retained(tmp_path, monkeypatch):
    path = tmp_path / "model"
    unlink = Path.unlink

    def fail(path, *args, **kwargs):
        if path.name.startswith(".turnscope-neural-"):
            raise PermissionError("private path SECRET")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    result = artifact.save_neural_forecaster(model(), path)
    assert result.cleanup_warning and "SECRET" not in result.cleanup_warning
    assert artifact.load_neural_forecaster(path).digest == model().digest
    assert len(list(tmp_path.glob(".turnscope-neural-*"))) == 1


def test_output_race_does_not_overwrite_another_file(tmp_path, monkeypatch):
    path = tmp_path / "model"
    link = os.link

    def raced(source, destination):
        Path(destination).write_bytes(b"other owner's file")
        return link(source, destination)

    monkeypatch.setattr(artifact.os, "link", raced)
    with pytest.raises(FileExistsError):
        artifact.save_neural_forecaster(model(), path)
    assert path.read_bytes() == b"other owner's file"
    assert not list(tmp_path.glob(".turnscope-neural-*"))


def test_read_and_write_refuse_symlinks_nonregular_files(saved):
    path, source = saved
    with pytest.raises(artifact.NeuralArtifactError):
        artifact.load_neural_forecaster(path.parent)
    with pytest.raises(artifact.NeuralArtifactError):
        artifact.save_neural_forecaster(source, path.parent, overwrite=True)
    alias = path.with_name("alias")
    try:
        alias.symlink_to(path)
    except OSError:
        pytest.skip("creating symlinks requires privileges on this Windows host")
    for function in (
        lambda: artifact.load_neural_forecaster(alias),
        lambda: artifact.save_neural_forecaster(source, alias, overwrite=True),
    ):
        with pytest.raises(artifact.NeuralArtifactError):
            function()


def test_open_identity_change_is_detected(saved, monkeypatch):
    path, _ = saved
    actual = artifact.os.fstat

    def altered(fd):
        info = actual(fd)
        values = list(info)
        values[1] += 1
        return os.stat_result(values)

    monkeypatch.setattr(artifact.os, "fstat", altered)
    with pytest.raises(artifact.NeuralArtifactError, match="changed"):
        artifact.load_neural_forecaster(path)


def test_manifest_and_archive_integrity_are_distinct(saved):
    path, source = saved
    manifest, members = unpack(path.read_bytes())
    assert manifest["sha256"] != source.digest
    manifest["sha256"] = "0" * 64
    path.write_bytes(repack(manifest, members, rehash=False))
    with pytest.raises(artifact.NeuralArtifactError, match="integrity"):
        artifact.load_neural_forecaster(path)


def test_source_tensor_budget_is_checked_before_any_payload_hash(monkeypatch):
    source = model()
    forbidden = Mock(side_effect=AssertionError("payload hashing happened before admission"))
    monkeypatch.setattr(artifact, "_sha", forbidden)
    with pytest.raises(artifact.NeuralArtifactError, match="tensor members"):
        artifact._encode(source, artifact.NeuralArtifactLimits(max_file_bytes=100))
    forbidden.assert_not_called()


def test_complete_manifest_cap_includes_own_sha_and_zip_overhead(saved):
    path, source = saved
    _, members = unpack(path.read_bytes())
    manifest_length = len(members[0][1])
    with pytest.raises(artifact.NeuralArtifactError, match="manifest"):
        artifact.load_neural_forecaster(
            path, limits=artifact.NeuralArtifactLimits(max_manifest_bytes=manifest_length - 1)
        )
    loaded = artifact.load_neural_forecaster(
        path, limits=artifact.NeuralArtifactLimits(max_manifest_bytes=manifest_length)
    )
    assert loaded.digest == source.digest
    with pytest.raises(artifact.NeuralArtifactError, match="file byte"):
        artifact._archive(
            members, artifact.NeuralArtifactLimits(max_file_bytes=len(path.read_bytes()) - 1)
        )


def test_exact_canonical_json_byte_count_for_escaping_and_unicode():
    value = {"é": ['\0\b\f\n\r\t\\"😀', -3, 1.5, True, False, None]}
    expected = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert (
        artifact._canonical(value, artifact.NeuralArtifactLimits(max_manifest_bytes=len(expected)))
        == expected
    )
    with pytest.raises(artifact.NeuralArtifactError, match="byte budget"):
        artifact._canonical(
            value, artifact.NeuralArtifactLimits(max_manifest_bytes=len(expected) - 1)
        )


def test_bad_model_type_unfitted_model_and_hidden_state_mismatch_are_controlled(tmp_path):
    source = model()
    object.__setattr__(source.state, "_identity", "0" * 64)
    for value in (object(), HierarchicalEventForecaster(), source):
        with pytest.raises(artifact.NeuralArtifactError):
            artifact.save_neural_forecaster(value, tmp_path / "out")
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(artifact.NeuralArtifactError):
        artifact.save_neural_forecaster(model(), tmp_path / "out", overwrite=1)
    with pytest.raises(artifact.NeuralArtifactError):
        artifact.load_neural_forecaster(tmp_path / "missing", limits={})


def test_flush_failure_does_not_publish_and_replacement_identity_race_is_rejected(
    saved, monkeypatch
):
    path, source = saved
    original = path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(artifact.os, "fsync", Mock(side_effect=OSError("disk flush failed")))
        with pytest.raises(OSError, match="disk flush"):
            artifact.save_neural_forecaster(source, path, overwrite=True)
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".turnscope-neural-*"))
    fsync = artifact.os.fsync

    def change_target(fd):
        fsync(fd)
        path.write_bytes(b"external update")

    monkeypatch.setattr(artifact.os, "fsync", change_target)
    with pytest.raises(artifact.NeuralArtifactError, match="changed"):
        artifact.save_neural_forecaster(source, path, overwrite=True)
    assert path.read_bytes() == b"external update"
    assert not list(path.parent.glob(".turnscope-neural-*"))
