"""Closed command boundaries and separately counted actual CPU process flows."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from turnscope import neural_ablation_artifact as artifact
from turnscope import neural_ablation_cli as command
from turnscope import neural_cli_io as shared
from turnscope.neural_ablation_math import ABLATION_VARIANTS
from turnscope.neural_forecast_data import SequenceLimits, SequencePolicy


def parser():
    value = argparse.ArgumentParser(prog="neural-ablation")
    command.configure_neural_ablation_parser(value)
    return value


def invoke(arguments):
    return command.run_neural_ablation_command(parser().parse_args(list(map(str, arguments))))


def partition(path, name, *, observed=False):
    records = []
    for index, label in enumerate((True, False, True)):
        texts = ["red café 中文" if label else "blue calm", "red answer" if label else "blue reply"]
        if not observed:
            texts.append("PRIVATE-FUTURE-MARKER")
        records.append(
            {
                "id": f"PRIVATE-ID-{name}-{index}",
                "utterances": [
                    {
                        "id": str(position),
                        "role": "person",
                        "text": text,
                        "timestamp": f"2026-01-01T00:00:{position:02d}+00:00",
                        "metadata": {"event": label and not observed and position == 2},
                    }
                    for position, text in enumerate(texts)
                ],
                "metadata": {"forecast_groups": [f"PRIVATE-GROUP-{name}-{index}"]},
            }
        )
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8"
    )
    return path


@pytest.fixture
def sources(tmp_path):
    result = {
        name: partition(tmp_path / (name + ".jsonl"), name)
        for name in ("training", "validation", "policy")
    }
    settings = tmp_path / "settings.json"
    settings.write_text(
        json.dumps(
            {
                "config": {"max_turn_tokens": 4},
                "training_config": {"epochs": 1, "batch_conversations": 2, "seed": 17},
            }
        ),
        encoding="utf-8",
    )
    return {**result, "settings": settings, "model": tmp_path / "model.tsa"}


def fit_arguments(paths, variant="order-erased.v1"):
    return [
        "fit",
        paths["training"],
        paths["validation"],
        paths["model"],
        "--policy-validation",
        paths["policy"],
        "--variant",
        variant,
        "--settings",
        paths["settings"],
    ]


@pytest.mark.parametrize("missing", ["--variant", "--policy-validation"])
def test_fit_has_no_implicit_mode_or_two_partition_fallback(sources, missing):
    arguments = fit_arguments(sources)
    position = arguments.index(missing)
    del arguments[position : position + 2]
    with pytest.raises(SystemExit) as caught:
        parser().parse_args(list(map(str, arguments)))
    assert caught.value.code == 2


def test_unknown_mode_rejected_by_parser(sources):
    with pytest.raises(SystemExit) as caught:
        parser().parse_args(list(map(str, fit_arguments(sources, "main-or-guessed"))))
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "alias",
    [
        "same-input",
        "hardlink-input",
        "policy-alias",
        "settings-alias",
        "model-input",
        "report-input",
        "report-model",
        "existing-model",
        "existing-report",
    ],
)
def test_paths_admitted_before_settings_reads_or_training(sources, monkeypatch, alias):
    report = sources["model"].with_suffix(".json")
    original = sources["training"].read_bytes()
    if alias == "same-input":
        sources["validation"] = sources["training"]
    elif alias == "hardlink-input":
        link = sources["training"].with_name("training-link")
        os.link(sources["training"], link)
        sources["validation"] = link
    elif alias == "policy-alias":
        sources["policy"] = sources["validation"]
    elif alias == "settings-alias":
        sources["settings"] = sources["training"]
    elif alias == "model-input":
        sources["model"] = sources["policy"]
    elif alias == "report-input":
        report = sources["training"]
    elif alias == "report-model":
        report = sources["model"]
    elif alias == "existing-model":
        sources["model"].write_bytes(b"existing")
    else:
        report.write_bytes(b"existing")
    forbidden = Mock(side_effect=AssertionError("no work before path validation"))
    monkeypatch.setattr(command, "_settings", forbidden)
    assert invoke([*fit_arguments(sources), "--output", report]) == 2
    forbidden.assert_not_called()
    assert sources["training"].read_bytes() == original


@pytest.mark.parametrize(
    "settings",
    [
        {"config": {"variant": "order-erased.v1"}},
        {"config": {"variant": "mean-word.v1"}},
        {"config": {"PRIVATE-UNKNOWN": 1}},
        {"config": {"max_turn_tokens": True}},
        {"config": {"embedding_dim": 3}},
        {"pooling_limits": {"max_operations": 10}},
        {"training_limits": {"max_total_pooling_operations": 10}},
        {"training_config": {"learning_rate": "PRIVATE-BAD-VALUE"}},
        {"numeric_limits": {"max_parameters": False}},
    ],
)
def test_settings_closed_and_mode_only_from_option(sources, monkeypatch, capsys, settings):
    sources["settings"].write_text(json.dumps(settings), encoding="utf-8")
    fit = Mock(side_effect=AssertionError("invalid settings invoked training"))
    monkeypatch.setattr(command.AblationEventForecaster, "fit", fit)
    assert invoke(fit_arguments(sources)) == 2
    fit.assert_not_called()
    captured = capsys.readouterr()
    assert "PRIVATE" not in captured.err and captured.out == ""
    assert not sources["model"].exists()


def test_fixed_pooling_defaults_and_explicit_variant_without_settings():
    from turnscope.neural_ablation_math import AblationPoolingLimits
    from turnscope.neural_ablation_train import AblationTrainingLimits

    model = command._settings(None, "mean-word.v1")
    assert model._config.variant == "mean-word.v1"
    assert model._pooling_limits == AblationPoolingLimits()
    assert model._training_limits == AblationTrainingLimits()


@pytest.mark.parametrize(
    "bad",
    [b'{"PRIVATE":1,"PRIVATE":2}\n', b'{"id":NaN}\n', b"\xff\n", b"\n", b'{"utterances":{}}\n'],
)
def test_malformed_jsonl_fails_before_fit_with_static_private_diagnostic(
    sources, monkeypatch, capsys, bad
):
    sources["policy"].write_bytes(bad)
    fit = Mock(side_effect=AssertionError("malformed data reached training"))
    monkeypatch.setattr(command.AblationEventForecaster, "fit", fit)
    assert invoke(fit_arguments(sources)) == 2
    fit.assert_not_called()
    error = capsys.readouterr().err
    assert "PRIVATE" not in error and str(sources["policy"]) not in error
    assert not sources["model"].exists()


def test_budget_rejects_before_conversation_decode(sources, monkeypatch):
    monkeypatch.setattr(shared, "MAX_FILE_BYTES", 1)
    decoder = Mock(side_effect=AssertionError("decoder ran after file admission failure"))
    monkeypatch.setattr(shared, "conversation_from_dict", decoder)
    assert invoke(fit_arguments(sources)) == 2
    decoder.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        ImportError("PRIVATE dependency"),
        RuntimeError("PRIVATE native"),
        ValueError("PRIVATE corpus"),
        TypeError("PRIVATE typed"),
        KeyError("PRIVATE key"),
        FloatingPointError("PRIVATE numeric"),
    ],
)
def test_execution_errors_do_not_echo_content_or_paths(sources, monkeypatch, capsys, error):
    monkeypatch.setattr(command.AblationEventForecaster, "fit", Mock(side_effect=error))
    assert invoke(fit_arguments(sources)) == 2
    assert "PRIVATE" not in capsys.readouterr().err
    assert not sources["model"].exists()


@pytest.mark.parametrize("action", ["inspect", "predict", "evaluate"])
def test_inference_alias_preflight_precedes_loading(tmp_path, monkeypatch, action):
    model = tmp_path / "private-model"
    model.write_bytes(b"not an artifact")
    arguments = [action, model]
    if action != "inspect":
        arguments.append(tmp_path / "private-source")
    load = Mock(side_effect=AssertionError("loaded before output alias validation"))
    monkeypatch.setattr(artifact, "load_ablation_forecaster", load)
    assert invoke([*arguments, "--output", model]) == 2
    load.assert_not_called()


@pytest.mark.parametrize("kind", ["closed", "absent", "short", "exception"])
def test_postfit_output_failures_preserve_published_model_and_return_one(
    tmp_path, monkeypatch, capsys, kind
):
    target = tmp_path / "published.tsa"

    def fit(args):
        target.write_bytes(b"authored stand-in for completed publication")
        return {"completed": True}

    monkeypatch.setattr(command, "_fit", fit)

    class Short(io.StringIO):
        def write(self, text):
            return len(text) - 1

    class Broken(io.StringIO):
        def write(self, text):
            raise OSError("PRIVATE stream failure")

    stream = io.StringIO()
    if kind == "closed":
        stream.close()
    elif kind == "absent":
        stream = None
    elif kind == "short":
        stream = Short()
    else:
        stream = Broken()
    monkeypatch.setattr(command.sys, "stdout", stream)
    args = argparse.Namespace(neural_ablation_action="fit", output=None)
    assert command.run_neural_ablation_command(args) == 1
    assert target.exists()
    error = capsys.readouterr().err
    assert "was published" in error and "PRIVATE" not in error


@pytest.mark.parametrize("closed_stderr", [False, True])
def test_closed_or_absent_stderr_cannot_mask_failure(monkeypatch, closed_stderr):
    stream = io.StringIO() if closed_stderr else None
    if stream is not None:
        stream.close()
    monkeypatch.setattr(command.sys, "stderr", stream)
    assert (
        command.run_neural_ablation_command(argparse.Namespace(neural_ablation_action="unknown"))
        == 2
    )


def test_report_cleanup_warning_is_success_and_private(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(command, "_inference", lambda args: {"complete": True})
    monkeypatch.setattr(
        command, "publish_neural_report", lambda payload, output: "PRIVATE implementation warning"
    )
    assert invoke(["inspect", tmp_path / "stand-in"]) == 0
    captured = capsys.readouterr()
    assert "cleanup" in captured.err and "PRIVATE" not in captured.err


def fake_model():
    """Transport/format oracle only; real model execution is tested separately."""
    value = SimpleNamespace(
        digest="a" * 64,
        training_summary={"declared_fixture": True},
        state=SimpleNamespace(
            config=SimpleNamespace(variant="mean-word.v1"),
            policy=SequencePolicy(),
            data_limits=SequenceLimits(),
        ),
    )
    value.fit = Mock(return_value=value)
    value.transform = Mock(
        side_effect=lambda inputs: tuple(
            SimpleNamespace(to_dict=lambda: {"probability": 0.75}) for _ in inputs
        )
    )
    value.evaluate = Mock(
        return_value={
            "format": "turnscope.neural-ablation-report.v1",
            "metrics": {"authored_transport_only": True},
        }
    )
    return value


@pytest.mark.parametrize("with_settings", [False, True])
def test_successful_fit_orchestration_uses_all_three_decoded_partitions(
    sources, monkeypatch, capsys, with_settings
):
    model = fake_model()
    monkeypatch.setattr(command, "_settings", lambda path, variant: model)

    def publish(value, path):
        assert value is model
        path.write_bytes(b"authored-transport-only")
        return artifact.NeuralSaveResult(str(path), "b" * 64, path.stat().st_size)

    monkeypatch.setattr(artifact, "save_ablation_forecaster", publish)
    arguments = fit_arguments(sources, "mean-word.v1")
    if not with_settings:
        position = arguments.index("--settings")
        del arguments[position : position + 2]
    assert invoke(arguments) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert set(report["sources"]) == {"training", "model_validation", "policy_validation"}
    args, kwargs = model.fit.call_args
    assert [item.id for item in args[0]] == [f"PRIVATE-ID-training-{i}" for i in range(3)]
    assert [item.id for item in args[1]] == [f"PRIVATE-ID-validation-{i}" for i in range(3)]
    assert [item.id for item in kwargs["policy_validation"]] == [
        f"PRIVATE-ID-policy-{i}" for i in range(3)
    ]
    assert report["source_execution_provenance_claimed"] is False
    assert "PRIVATE" not in captured.out and captured.err == ""


@pytest.mark.parametrize(
    "action,identifiers",
    [("inspect", False), ("predict", False), ("predict", True), ("evaluate", False)],
)
def test_successful_inference_formats_and_identifier_opt_in(
    tmp_path, monkeypatch, capsys, action, identifiers
):
    model = fake_model()
    monkeypatch.setattr(artifact, "load_ablation_forecaster", lambda path: model)
    bundle = tmp_path / "stand-in.tsa"
    source = partition(tmp_path / "source.jsonl", "unseen", observed=action == "predict")
    args = [action, bundle]
    if action != "inspect":
        args.append(source)
    if identifiers:
        args.append("--include-identifiers")
    assert invoke(args) == 0
    output = capsys.readouterr().out
    result = json.loads(output)
    if action == "inspect":
        assert result["vocabulary_included"] is result["weights_included"] is False
        assert result["source_execution_provenance_claimed"] is False
    elif action == "evaluate":
        assert result["source"]["conversations"] == 3
        assert len(model.evaluate.call_args.args[0]) == 3
    else:
        assert [row["index"] for row in result["predictions"]] == [0, 1, 2]
        assert ("PRIVATE-ID" in output) is identifiers
        assert result["text_included"] is False and "café" not in output
        assert len(model.transform.call_args.args[0]) == 3


def test_completed_inference_report_failure_is_not_command_failure(monkeypatch, capsys):
    monkeypatch.setattr(command, "_inference", lambda args: {"complete": True})
    monkeypatch.setattr(command, "publish_neural_report", Mock(side_effect=OSError("PRIVATE")))
    assert (
        command.run_neural_ablation_command(
            argparse.Namespace(neural_ablation_action="inspect", output=None)
        )
        == 1
    )
    diagnostic = capsys.readouterr().err
    assert "inference completed" in diagnostic and "PRIVATE" not in diagnostic


def test_prediction_does_not_infer_future_cut_from_outcome_metadata(tmp_path, monkeypatch, capsys):
    model = fake_model()
    monkeypatch.setattr(artifact, "load_ablation_forecaster", lambda path: model)
    source = partition(tmp_path / "complete-conversations.jsonl", "unseen", observed=False)
    assert invoke(["predict", tmp_path / "stand-in.tsa", source]) == 0
    prefixes = model.transform.call_args.args[0]
    assert [len(prefix.turns) for prefix in prefixes] == [3, 3, 3]
    assert all(prefix.turns[-1].text == "PRIVATE-FUTURE-MARKER" for prefix in prefixes)
    assert "PRIVATE-FUTURE-MARKER" not in capsys.readouterr().out


def environment():
    return {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "PYTHONIOENCODING": "utf-8",
    }


def test_real_closed_stdout_pipe_preserves_completed_status_one(tmp_path):
    target = tmp_path / "model-was-published"
    script = """
import argparse, pathlib, sys
from turnscope import neural_ablation_cli as command
target=pathlib.Path(sys.argv[1])
def fit(args):
    target.write_bytes(b'authored-publication-stand-in')
    return {'complete':True,'padding':'x'*1000000}
command._fit=fit
args=argparse.Namespace(neural_ablation_action='fit',output=None)
raise SystemExit(command.run_neural_ablation_command(args))
"""
    with subprocess.Popen(
        [sys.executable, "-X", "utf8", "-c", script, str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment(),
    ) as child:
        child.stdout.close()
        error = child.stderr.read().decode("utf-8")
        assert child.wait(timeout=30) == 1
    assert target.exists()
    assert "was published" in error and "Exception ignored" not in error


def test_entrypoint_help_does_not_import_numpy_or_torch():
    script = """
import importlib.abc,sys
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'numpy','torch'}:
            raise AssertionError('help imported numerical dependency')
sys.meta_path.insert(0,Forbidden())
from turnscope.cli import main
raise SystemExit(main(['neural-ablation','fit','--help']))
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        env={**environment(), "PYTHON_COLORS": "1"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    assert "--variant" in result.stdout and "--policy-validation" in result.stdout
    assert "\x1b" not in result.stdout + result.stderr


def actual_process(arguments, *, fit=False):
    before = (
        "import torch; torch.set_num_threads(1)"
        if fit
        else """
import importlib.abc
class Forbidden(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('inference imported Torch')
sys.meta_path.insert(0,Forbidden())
"""
    )
    script = (
        "import sys\n"
        + before
        + "\nfrom turnscope.cli import main\nraise SystemExit(main(sys.argv[1:]))"
    )
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script, "neural-ablation", *map(str, arguments)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment(),
        timeout=60,
    )


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_actual_fit_then_torch_forbidden_inspect_predict_evaluate(sources, tmp_path, variant):
    pytest.importorskip("torch", reason="actual CPU CLI fit requires optional Torch")
    report = tmp_path / "fit-report.json"
    fitted = actual_process([*fit_arguments(sources, variant), "--output", report], fit=True)
    assert fitted.returncode == 0, fitted.stderr
    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["completed"] and value["variant"] == variant
    assert value["training"]["policy_reuses_model_validation"] is False
    assert value["training"]["torch_threads"] == 1
    assert value["training"]["changed_parameter_names"]
    assert set(value["sources"]) == {"training", "model_validation", "policy_validation"}
    assert "PRIVATE" not in report.read_text(encoding="utf-8")
    inspect = actual_process(["inspect", sources["model"]])
    assert inspect.returncode == 0, inspect.stderr
    checked = json.loads(inspect.stdout)
    assert checked["model_digest"] == value["model_digest"]
    assert checked["vocabulary_included"] is checked["weights_included"] is False
    observed = partition(tmp_path / "observed.jsonl", "new-prediction", observed=True)
    prediction = actual_process(["predict", sources["model"], observed])
    assert prediction.returncode == 0, prediction.stderr
    predicted = json.loads(prediction.stdout)
    assert len(predicted["predictions"]) == 3 and predicted["identifiers_included"] is False
    assert "PRIVATE-ID" not in prediction.stdout and "café" not in prediction.stdout
    identified = actual_process(["predict", sources["model"], observed, "--include-identifiers"])
    assert identified.returncode == 0 and "PRIVATE-ID-new-prediction" in identified.stdout
    heldout = partition(tmp_path / "heldout.jsonl", "heldout")
    evaluated = actual_process(["evaluate", sources["model"], heldout])
    assert evaluated.returncode == 0, evaluated.stderr
    assert json.loads(evaluated.stdout)["format"] == "turnscope.neural-ablation-report.v1"
    reused = actual_process(["evaluate", sources["model"], sources["policy"]])
    assert reused.returncode == 2 and "PRIVATE" not in reused.stderr
    original = sources["model"].read_bytes()
    repeated = actual_process(fit_arguments(sources, variant))
    assert repeated.returncode == 2 and sources["model"].read_bytes() == original
