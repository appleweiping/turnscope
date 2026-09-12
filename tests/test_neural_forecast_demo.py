"""Helper boundaries only; real installed training is an independent demo run."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from turnscope import prepare_sequence_forecasts
from turnscope.neural_cli_io import read_neural_jsonl

PATH = Path(__file__).resolve().parents[1] / "examples" / "neural_forecast_demo.py"
SPEC = importlib.util.spec_from_file_location("neural_demo", PATH)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


def test_authored_fixture_has_disjoint_groups_and_only_pre_event_features(tmp_path):
    groups = set()
    for name in ("training", "validation", "policy", "heldout"):
        path = tmp_path / f"{name}.jsonl"
        demo._write_jsonl(path, demo._records(name))
        source, binding = read_neural_jsonl(path)
        prepared = prepare_sequence_forecasts(source)
        assert len(prepared.observations) == len(prepared.examples) == 4
        assert {item.label for item in prepared.examples} == {True, False}
        assert not groups & prepared.group_digests
        groups.update(prepared.group_digests)
        assert binding["sha256"] == demo._sha(path)
        assert all(
            len(observation.turns) == 2
            and not any("authored annotated" in turn.text for turn in observation.turns)
            for observation in prepared.observations
        )
    observed = demo._records("heldout", observations_only=True)
    assert all(len(row["utterances"]) == 2 for row in observed)
    assert all(not turn["metadata"]["event"] for row in observed for turn in row["utterances"])


def test_fixture_output_is_exclusive(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_bytes(b"another writer\n")
    with pytest.raises(FileExistsError):
        demo._write_jsonl(path, demo._records("training"))
    assert path.read_bytes() == b"another writer\n"


def test_demo_rejects_existing_output_directory_before_commands(tmp_path, monkeypatch):
    runner = Mock(side_effect=AssertionError("no command may run"))
    monkeypatch.setattr(demo, "_run", runner)
    with pytest.raises(FileExistsError):
        demo.run_demo(tmp_path)
    runner.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("frozen", [False, True])
def test_commands_are_isolated_and_frozen_commands_forbid_torch(monkeypatch, frozen):
    runner = Mock()
    monkeypatch.setattr(demo.subprocess, "run", runner)
    demo._run(["inspect", "model.tsn"], frozen=frozen)
    argv = runner.call_args.args[0]
    assert argv[:4] == [sys.executable, "-I", "-X", "utf8"]
    assert argv[-3:] == ["neural-forecast", "inspect", "model.tsn"]
    if frozen:
        assert argv[4:6] == ["-c", demo._FROZEN_COMMAND]
        assert "sys.meta_path.insert" in argv[5] and "attempted to import Torch" in argv[5]
    else:
        assert argv[4:6] == ["-m", "turnscope"]
    assert runner.call_args.kwargs == {
        "check": True,
        "capture_output": True,
        "encoding": "utf-8",
        "timeout": 180,
    }


def test_failed_first_command_preserves_inputs_without_false_demo_report(tmp_path, monkeypatch):
    output = tmp_path / "run"
    monkeypatch.setattr(demo, "_run", Mock(side_effect=RuntimeError("injected command failure")))
    with pytest.raises(RuntimeError, match="injected command failure"):
        demo.run_demo(output)
    assert (output / "training.jsonl").is_file()
    assert not (output / "private-model.tsn").exists()
    assert not (output / "demo-report.json").exists()


def test_closed_stdout_does_not_erase_a_delivered_demo_report(tmp_path, monkeypatch):
    report = {"format": "authored-delivery-control", "completed": True}

    def completed(directory):
        directory.mkdir()
        (directory / "demo-report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(demo, "run_demo", completed)
    closed = io.StringIO()
    closed.close()
    if sys.version_info >= (3, 14):
        import _colorize

        monkeypatch.setattr(_colorize, "can_colorize", lambda **_: closed.isatty())
    monkeypatch.setattr(sys, "stdout", closed)
    output = tmp_path / "delivered"
    assert demo.main([str(output)]) == 1
    assert json.loads((output / "demo-report.json").read_text(encoding="utf-8")) == report
