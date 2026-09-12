"""Neural command safety plus a genuine tiny optional CPU/artifact roundtrip."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from turnscope import neural_forecast_artifact as artifact
from turnscope import neural_forecast_cli as command
from turnscope.cli import main


def write_partition(path, name, *, observations_only=False):
    records = []
    for index, positive in enumerate((True, False, True)):
        texts = ["red warm", "red answer"] if positive else ["blue calm", "blue answer"]
        if not observations_only:
            texts.append("private future attack" if positive else "private censored end")
        records.append(
            {
                "id": f"private-id-{name}-{index}",
                "utterances": [
                    {
                        "id": str(turn),
                        "role": "person",
                        "text": text,
                        "timestamp": f"2026-01-01T00:00:{turn:02d}+00:00",
                        "metadata": {"event": positive and not observations_only and turn == 2},
                    }
                    for turn, text in enumerate(texts)
                ],
                "metadata": {"forecast_groups": [f"group:{name}:{index}"]},
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


@pytest.fixture
def input_paths(tmp_path):
    training = write_partition(tmp_path / "training.jsonl", "training")
    validation = write_partition(tmp_path / "validation.jsonl", "validation")
    policy = write_partition(tmp_path / "policy.jsonl", "policy")
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "config": {"embedding_dim": 3, "word_hidden": 2, "turn_hidden": 4},
                "training_config": {"epochs": 1, "batch_conversations": 2, "seed": 3},
            }
        ),
        encoding="utf-8",
    )
    return training, validation, policy, settings


def test_help_does_not_import_optional_numerical_dependencies_or_emit_ansi():
    script = r"""
import importlib.abc, sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"numpy", "torch"}:
            raise AssertionError("help imported an optional numerical dependency")
sys.meta_path.insert(0, Forbidden())
from turnscope.cli import main
main(["neural-forecast", "fit", "--help"])
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHON_COLORS"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env=environment,
    )
    assert "--policy-validation" in result.stdout and "--settings" in result.stdout
    assert "\x1b" not in result.stdout + result.stderr


@pytest.mark.parametrize(
    "alias", ["same_inputs", "hardlink_inputs", "model_input", "report_input", "report_model"]
)
def test_fit_path_aliases_rejected_before_settings_or_training(
    input_paths, tmp_path, monkeypatch, capsys, alias
):
    training, validation, _, _ = input_paths
    model, report = tmp_path / "model.bundle", tmp_path / "report.json"
    if alias == "same_inputs":
        validation = training
    elif alias == "hardlink_inputs":
        validation = tmp_path / "hardlink-input"
        os.link(training, validation)
    elif alias == "model_input":
        model = training
    elif alias == "report_input":
        report = validation
    else:
        report = model
    before = training.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("work before alias checks")

    monkeypatch.setattr(command, "_settings", forbidden)
    assert (
        main(
            [
                "neural-forecast",
                "fit",
                str(training),
                str(validation),
                str(model),
                "--output",
                str(report),
            ]
        )
        == 2
    )
    assert "error" in capsys.readouterr().err
    assert training.read_bytes() == before


@pytest.mark.parametrize("action", ["predict", "evaluate", "inspect"])
def test_inference_output_aliases_rejected_before_bundle_loading(
    tmp_path, monkeypatch, capsys, action
):
    model = tmp_path / "model"
    model.write_bytes(b"not an artifact; must never be opened")

    def forbidden(*args, **kwargs):
        raise AssertionError("loaded model before alias checks")

    monkeypatch.setattr(artifact, "load_neural_forecaster", forbidden)
    arguments = ["neural-forecast", action, str(model)]
    if action != "inspect":
        arguments.append(str(tmp_path / "input"))
    assert main([*arguments, "--output", str(model)]) == 2
    assert "differ" in capsys.readouterr().err


@pytest.mark.parametrize(
    "settings",
    [
        {"config": {"private-secret-field": True}},
        {"config": {"word_hidden": True}},
        {"training_config": {"learning_rate": "private-secret-value"}},
        {"config": {"max_turn_tokens": 129}},
        {"policy": {"event_field": "x", "skip_field": "x"}},
    ],
)
def test_bad_settings_fail_without_training_or_private_echo(
    input_paths, tmp_path, monkeypatch, capsys, settings
):
    training, validation, _, path = input_paths
    path.write_text(json.dumps(settings), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("training ran despite invalid settings")

    monkeypatch.setattr(command.HierarchicalEventForecaster, "fit", forbidden)
    model = tmp_path / "model"
    assert (
        main(
            [
                "neural-forecast",
                "fit",
                str(training),
                str(validation),
                str(model),
                "--settings",
                str(path),
            ]
        )
        == 2
    )
    diagnostic = capsys.readouterr().err
    assert "private-secret" not in diagnostic and not model.exists()


def test_malformed_training_data_fails_before_model_fit(input_paths, tmp_path, monkeypatch, capsys):
    training, validation, _, settings = input_paths
    training.write_bytes(b'{"private-secret":"bad","private-secret":"duplicate"}\n')

    def forbidden(*args, **kwargs):
        raise AssertionError("model fit invoked after malformed source")

    monkeypatch.setattr(command.HierarchicalEventForecaster, "fit", forbidden)
    assert (
        main(
            [
                "neural-forecast",
                "fit",
                str(training),
                str(validation),
                str(tmp_path / "model"),
                "--settings",
                str(settings),
            ]
        )
        == 2
    )
    assert "private-secret" not in capsys.readouterr().err


def test_missing_optional_training_dependency_is_controlled(
    input_paths, tmp_path, monkeypatch, capsys
):
    training, validation, _, settings = input_paths

    def missing(*args, **kwargs):
        raise ImportError("CPU training dependency unavailable")

    monkeypatch.setattr(command.HierarchicalEventForecaster, "fit", missing)
    assert (
        main(
            [
                "neural-forecast",
                "fit",
                str(training),
                str(validation),
                str(tmp_path / "model"),
                "--settings",
                str(settings),
            ]
        )
        == 2
    )
    assert "CPU training dependency unavailable" in capsys.readouterr().err


def test_post_work_stdout_failure_returns_one_not_parse_error(monkeypatch):
    import argparse

    monkeypatch.setattr(command, "_inference", lambda args: {"complete": True})
    stream = io.StringIO()
    stream.close()
    monkeypatch.setattr(command.sys, "stdout", stream)
    assert (
        command.run_neural_forecast_command(
            argparse.Namespace(neural_action="inspect", output=None)
        )
        == 1
    )


@pytest.fixture(scope="module")
def real_cli_bundle(tmp_path_factory):
    pytest.importorskip("torch", reason="genuine CLI fit needs optional CPU training")
    directory = tmp_path_factory.mktemp("neural-cli-real")
    training = write_partition(directory / "training.jsonl", "real-training")
    validation = write_partition(directory / "validation.jsonl", "real-validation")
    policy = write_partition(directory / "policy.jsonl", "real-policy")
    settings = directory / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "config": {"embedding_dim": 3, "word_hidden": 2, "turn_hidden": 4},
                "training_config": {"epochs": 1, "batch_conversations": 2, "seed": 7},
            }
        ),
        encoding="utf-8",
    )
    bundle, report = directory / "model.bundle", directory / "fit.json"
    assert (
        main(
            [
                "neural-forecast",
                "fit",
                str(training),
                str(validation),
                str(bundle),
                "--settings",
                str(settings),
                "--policy-validation",
                str(policy),
                "--output",
                str(report),
            ]
        )
        == 0
    )
    return directory, bundle, json.loads(report.read_text(encoding="utf-8"))


def test_real_fit_saved_artifact_and_inspect_match_frozen_identity(real_cli_bundle):
    directory, bundle, training = real_cli_bundle
    loaded = artifact.load_neural_forecaster(bundle)
    assert training["completed"] is True and training["model_digest"] == loaded.digest
    assert not training["training"]["policy_reuses_model_validation"]
    assert training["training"]["changed_parameter_names"]
    assert "private future" not in json.dumps(training)
    assert "private-id" not in json.dumps(training)
    output = directory / "inspect.json"
    assert main(["neural-forecast", "inspect", str(bundle), "--output", str(output)]) == 0
    inspected = json.loads(output.read_text(encoding="utf-8"))
    assert inspected["model_digest"] == loaded.digest
    assert inspected["training"] == loaded.training_summary


def test_real_predict_default_privacy_and_explicit_identifier_optin(real_cli_bundle):
    directory, bundle, _ = real_cli_bundle
    observed = write_partition(directory / "observed.jsonl", "observed", observations_only=True)
    private, identified = directory / "predictions.json", directory / "identified.json"
    assert (
        main(["neural-forecast", "predict", str(bundle), str(observed), "--output", str(private)])
        == 0
    )
    report = json.loads(private.read_text(encoding="utf-8"))
    assert report["identifiers_included"] is False and report["text_included"] is False
    assert "private-id" not in json.dumps(report) and "red warm" not in json.dumps(report)
    assert [row["index"] for row in report["predictions"]] == [0, 1, 2]
    assert (
        main(
            [
                "neural-forecast",
                "predict",
                str(bundle),
                str(observed),
                "--include-identifiers",
                "--output",
                str(identified),
            ]
        )
        == 0
    )
    rows = json.loads(identified.read_text(encoding="utf-8"))["predictions"]
    assert all(row["conversation_id"].startswith("private-id-observed-") for row in rows)
    assert all("text" not in row for row in rows)


def test_real_evaluate_preserves_test_policy_and_rejects_fitting_group_overlap(
    real_cli_bundle, capsys
):
    directory, bundle, _ = real_cli_bundle
    heldout = write_partition(directory / "heldout.jsonl", "heldout")
    output = directory / "evaluation.json"
    assert (
        main(["neural-forecast", "evaluate", str(bundle), str(heldout), "--output", str(output)])
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["metrics"]["conversations"] == 3 and report["metrics"]["prefixes"] == 3
    assert report["whole_repository_parity_claimed"] is False
    denied = directory / "denied.json"
    assert (
        main(
            [
                "neural-forecast",
                "evaluate",
                str(bundle),
                str(directory / "training.jsonl"),
                "--output",
                str(denied),
            ]
        )
        == 2
    )
    assert "overlap" in capsys.readouterr().err and not denied.exists()


def test_real_loaded_cli_inference_in_fresh_subprocess_cannot_import_torch(real_cli_bundle):
    directory, bundle, _ = real_cli_bundle
    observed = write_partition(directory / "child-observed.jsonl", "child", observations_only=True)
    script = r"""
import importlib.abc, sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("frozen CLI inference imported Torch")
sys.meta_path.insert(0, Forbidden())
from turnscope.cli import main
sys.exit(main(["neural-forecast", "predict", sys.argv[1], sys.argv[2]]))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, str(bundle), str(observed)],
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
        env=environment,
    )
    assert len(json.loads(result.stdout)["predictions"]) == 3


def test_real_report_cleanup_warning_does_not_erase_published_result(
    real_cli_bundle, monkeypatch, capsys
):
    directory, bundle, _ = real_cli_bundle
    original = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if path.name.startswith(".turnscope-neural-report-"):
            raise OSError("private cleanup details")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    output = directory / "cleanup-report.json"
    assert main(["neural-forecast", "inspect", str(bundle), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["model_digest"]
    warning = capsys.readouterr().err
    assert "report published" in warning and "private cleanup details" not in warning


def _cached_candidate_for_delivery_fault(bundle, monkeypatch):
    """Reuse actually trained weights solely to isolate post-fit delivery faults."""
    model = artifact.load_neural_forecaster(bundle)
    monkeypatch.setattr(command, "_settings", lambda *_: model)
    monkeypatch.setattr(model, "fit", lambda *args, **kwargs: model)
    return model


def test_real_model_survives_report_destination_race(real_cli_bundle, monkeypatch, capsys):
    directory, bundle, _ = real_cli_bundle
    candidate = _cached_candidate_for_delivery_fault(bundle, monkeypatch)
    destination, report = (
        directory / "saved-before-report-race.bundle",
        directory / "raced-report.json",
    )
    original_save = artifact.save_neural_forecaster

    def raced_save(model, path):
        publication = original_save(model, path)
        report.write_bytes(b"other report writer")
        return publication

    monkeypatch.setattr(artifact, "save_neural_forecaster", raced_save)
    assert (
        main(
            [
                "neural-forecast",
                "fit",
                str(directory / "training.jsonl"),
                str(directory / "validation.jsonl"),
                str(destination),
                "--policy-validation",
                str(directory / "policy.jsonl"),
                "--output",
                str(report),
            ]
        )
        == 2
    )
    assert report.read_bytes() == b"other report writer"
    assert artifact.load_neural_forecaster(destination).digest == candidate.digest
    assert "new file" in capsys.readouterr().err


def test_real_model_survives_closed_report_stdout(real_cli_bundle, monkeypatch):
    directory, bundle, _ = real_cli_bundle
    candidate = _cached_candidate_for_delivery_fault(bundle, monkeypatch)
    destination = directory / "saved-before-closed-stdout.bundle"
    stream = io.StringIO()
    stream.close()
    monkeypatch.setattr(command.sys, "stdout", stream)
    assert (
        main(
            [
                "neural-forecast",
                "fit",
                str(directory / "training.jsonl"),
                str(directory / "validation.jsonl"),
                str(destination),
                "--policy-validation",
                str(directory / "policy.jsonl"),
            ]
        )
        == 1
    )
    assert artifact.load_neural_forecaster(destination).digest == candidate.digest
