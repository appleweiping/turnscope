"""Shipped demo boundaries and real report mutations, not real-corpus quality."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from turnscope import prepare_sequence_forecasts
from turnscope.neural_cli_io import read_neural_jsonl

PATH = Path(__file__).resolve().parents[1] / "examples/neural_ablation_demo.py"
SPEC = importlib.util.spec_from_file_location("ablation_installed_demo", PATH)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def test_authored_inputs_are_disjoint_and_supervision_is_not_observed(tmp_path):
    groups = set()
    for partition in ("training", "validation", "policy", "heldout"):
        path = tmp_path / f"{partition}.jsonl"
        demo._write_jsonl(path, demo._records(partition))
        raw, source = read_neural_jsonl(path)
        data = prepare_sequence_forecasts(raw)
        assert source["sha256"] == demo._sha(path)
        assert len(data.observations) == len(data.examples) == 4
        assert {item.label for item in data.examples} == {True, False}
        assert not groups & data.group_digests
        groups.update(data.group_digests)
        assert all(
            len(item.turns) == 2 and not any("annotated event" in turn.text for turn in item.turns)
            for item in data.observations
        )
    observed = demo._records("heldout", observed=True)
    assert all(len(row["utterances"]) == 2 for row in observed)


def test_file_and_directory_publication_are_exclusive(tmp_path, monkeypatch):
    target = tmp_path / "existing.jsonl"
    target.write_bytes(b"previous writer\n")
    with pytest.raises(FileExistsError):
        demo._write_jsonl(target, demo._records("training"))
    assert target.read_bytes() == b"previous writer\n"
    run = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(demo, "_run", run)
    with pytest.raises(FileExistsError):
        demo.run_demo(tmp_path)
    run.assert_not_called()


@pytest.mark.parametrize("frozen", [False, True])
def test_commands_are_explicit_isolated_and_bounded(frozen, monkeypatch):
    run = Mock()
    monkeypatch.setattr(demo.subprocess, "run", run)
    demo._run(["inspect", "model.tsa"], frozen=frozen)
    assert run.call_args.args[0][:4] == [sys.executable, "-I", "-X", "utf8"]
    assert run.call_args.args[0][-3:] == ["neural-ablation", "inspect", "model.tsa"]
    assert run.call_args.args[0][4:6] == (
        ["-c", demo._FROZEN_COMMAND if frozen else demo._TRAIN_COMMAND]
    )
    assert run.call_args.kwargs == {
        "capture_output": True,
        "check": True,
        "encoding": "utf-8",
        "timeout": 180,
    }


def test_failed_first_command_preserves_attempt_and_never_emits_acceptance(tmp_path, monkeypatch):
    run = Mock(side_effect=subprocess.CalledProcessError(2, ["authored failure"]))
    monkeypatch.setattr(demo, "_run", run)
    target = tmp_path / "attempt"
    with pytest.raises(subprocess.CalledProcessError):
        demo.run_demo(target)
    assert (target / "training.jsonl").exists()
    assert not (target / "demo-report.json").exists()
    assert run.call_count == 1


def test_closed_stdout_preserves_delivered_demo_file(tmp_path, monkeypatch):
    report = {"completed": True, "authored_fixture_only": True}

    def complete(directory):
        directory.mkdir()
        (directory / "demo-report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(demo, "run_demo", complete)
    closed = io.StringIO()
    closed.close()
    if sys.version_info >= (3, 14):
        import _colorize

        monkeypatch.setattr(_colorize, "can_colorize", lambda **_: closed.isatty())
    monkeypatch.setattr(sys, "stdout", closed)
    target = tmp_path / "delivered"
    assert demo.main([str(target)]) == 1
    assert json.loads((target / "demo-report.json").read_text(encoding="utf-8")) == report


@pytest.fixture(scope="module")
def actual_demo(tmp_path_factory):
    torch = pytest.importorskip("torch", reason="real control report templates need CPU training")
    from turnscope.cli import main

    calls = []

    def current_interpreter(arguments, *, frozen):
        # This genuine command fixture is not proof of subprocess isolation.
        # The separate installed -I demo enforces Torch-free child processes.
        calls.append((arguments[0], frozen))
        assert main(["neural-ablation", *arguments]) == 0

    previous = torch.get_num_threads()
    output = tmp_path_factory.mktemp("ablation-demo-actual") / "run"
    try:
        torch.set_num_threads(1)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(demo, "_run", current_interpreter)
            aggregate = demo.run_demo(output)
    finally:
        torch.set_num_threads(previous)
    assert calls == [("fit", False), ("inspect", True), ("predict", True), ("evaluate", True)] * 3
    assert aggregate["completed"] is True
    assert [item["variant"] for item in aggregate["variants"]] == list(demo.VARIANTS)
    reports = {
        variant: {
            action: json.loads(
                (output / variant / f"{action}-report.json").read_text(encoding="utf-8")
            )
            for action in ("training", "inspection", "prediction", "evaluation")
        }
        for variant in demo.VARIANTS
    }
    return output, reports, aggregate


@pytest.mark.parametrize("variant", demo.VARIANTS)
def test_real_report_support_and_parameter_inventories_validate(actual_demo, variant):
    directory, reports, aggregate = actual_demo
    result = demo._validate_reports(directory, directory / variant, variant, reports[variant])
    assert result in aggregate["variants"]
    assert result["optimizer_steps_executed"] == 4
    assert aggregate["source_sha256"] == demo._sources()
    for name, digest in aggregate["file_sha256"].items():
        assert demo._sha(directory / name) == digest


@pytest.mark.parametrize(
    "path,value",
    [
        (("training", "format"), "wrong"),
        (("training", "variant"), "main"),
        (("training", "completed"), False),
        (("inspection", "model_digest"), "0" * 64),
        (("training", "publication", "sha256"), "0" * 64),
        (("training", "publication", "bytes_written"), True),
        (("inspection", "training", "selected_epoch"), 0),
        (("training", "sources", "policy_validation", "sha256"), "0" * 64),
        (("training", "sources", "training", "bytes"), 0),
        (("prediction", "source", "conversations"), 0),
        (("evaluation", "source", "utterances"), True),
        (("inspection", "vocabulary_included"), True),
        (("inspection", "weights_included"), True),
        (("training", "source_execution_provenance_claimed"), True),
        (("prediction", "identifiers_included"), True),
        (("prediction", "text_included"), True),
        (("prediction", "predictions", 0, "index"), False),
        (("prediction", "predictions", 0, "observed_turns"), 3),
        (("prediction", "predictions", 0, "variant"), "mean-word.v1"),
        (("prediction", "predictions", 0, "model_digest"), "0" * 64),
        (("evaluation", "support", "observed_turns"), 0),
        (("evaluation", "metrics", "prefixes"), 0),
        (("evaluation", "probability_calibration_claimed"), True),
        (("evaluation", "whole_repository_parity_claimed"), True),
    ],
)
def test_corrupted_real_receipts_do_not_produce_demo_acceptance(actual_demo, path, value):
    directory, templates, _ = actual_demo
    variant = "current-turn.v1"
    reports = copy.deepcopy(templates[variant])
    target = reports
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        demo._validate_reports(directory, directory / variant, variant, reports)


@pytest.mark.parametrize(
    "key,value",
    [
        ("training_conversations", 0),
        ("policy_validation_prefixes", True),
        ("policy_reuses_model_validation", True),
        ("parameter_count", 1),
        ("canonical_main_parameter_count", 1),
        ("fixed_pad_parameters", 0),
        ("initial_parameter_sha256", {}),
        ("main_initial_parameter_sha256", {}),
        ("history", []),
    ],
)
def test_consistently_wrong_fit_and_reload_metadata_is_rejected(actual_demo, key, value):
    directory, templates, _ = actual_demo
    variant = "current-turn.v1"
    reports = copy.deepcopy(templates[variant])
    for action in ("training", "inspection"):
        reports[action]["training"][key] = value
    with pytest.raises(ValueError):
        demo._validate_reports(directory, directory / variant, variant, reports)
