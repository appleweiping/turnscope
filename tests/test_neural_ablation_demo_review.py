"""Independent receipt mutations using one real tiny authored CPU control.

The command fixture intentionally runs in-process; it is not evidence of child
isolation. These tests check that a successful demo cannot bless inconsistent
prediction/metric receipts, and impose no desired quality score.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "examples/neural_ablation_demo.py"
SPEC = importlib.util.spec_from_file_location("independent_ablation_demo_review", PATH)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)
VARIANT = "current-turn.v1"


@pytest.fixture(scope="module")
def actual_receipts(tmp_path_factory):
    torch = pytest.importorskip(
        "torch", reason="authored independent receipt oracle needs CPU Torch"
    )
    from turnscope.cli import main

    directory = tmp_path_factory.mktemp("independent-control-receipts")
    target = directory / VARIANT
    target.mkdir()
    for name in ("training", "validation", "policy", "heldout"):
        demo._write_jsonl(directory / f"{name}.jsonl", demo._records(name))
    demo._write_jsonl(directory / "observed.jsonl", demo._records("heldout", observed=True))
    settings = directory / "settings.json"
    settings.write_text(
        json.dumps(
            {"training_config": {"epochs": 2, "patience": 2, "batch_conversations": 3, "seed": 17}}
        ),
        encoding="utf-8",
    )
    model = target / "private-model.tsa"
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        assert (
            main(
                [
                    "neural-ablation",
                    "fit",
                    str(directory / "training.jsonl"),
                    str(directory / "validation.jsonl"),
                    str(model),
                    "--policy-validation",
                    str(directory / "policy.jsonl"),
                    "--variant",
                    VARIANT,
                    "--settings",
                    str(settings),
                    "--output",
                    str(target / "training-report.json"),
                ]
            )
            == 0
        )
        for command, source, report in (
            ("inspect", None, "inspection"),
            ("predict", "observed.jsonl", "prediction"),
            ("evaluate", "heldout.jsonl", "evaluation"),
        ):
            arguments = ["neural-ablation", command, str(model)]
            if source is not None:
                arguments.append(str(directory / source))
            assert main([*arguments, "--output", str(target / f"{report}-report.json")]) == 0
    finally:
        torch.set_num_threads(previous)
    reports = {
        name: json.loads((target / f"{name}-report.json").read_text(encoding="utf-8"))
        for name in ("training", "inspection", "prediction", "evaluation")
    }
    demo._validate_reports(directory, target, VARIANT, reports)
    return directory, target, reports


def test_authored_prediction_and_evaluation_have_a_hand_computed_metric_oracle(actual_receipts):
    _, _, reports = actual_receipts
    rows = reports["prediction"]["predictions"]
    labels = (True, False, True, False)
    probabilities = [row["probability"] for row in rows]
    metrics = reports["evaluation"]["metrics"]
    threshold = reports["training"]["training"]["threshold"]
    expected_brier = (
        math.fsum((p - int(y)) ** 2 for p, y in zip(probabilities, labels, strict=True)) / 4
    )
    expected_loss = (
        -math.fsum(
            math.log(p) if y else math.log1p(-p) for p, y in zip(probabilities, labels, strict=True)
        )
        / 4
    )
    assert metrics["conversation_weighted_prefix_brier"] == pytest.approx(expected_brier, abs=1e-14)
    assert metrics["conversation_weighted_prefix_log_loss"] == pytest.approx(
        expected_loss, abs=1e-14
    )
    assert metrics["threshold"] == threshold
    assert all(
        row["threshold"] == threshold and row["alert"] is (row["probability"] >= threshold)
        for row in rows
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.25, 1.25, True])
def test_invalid_prediction_probability_cannot_be_omitted_from_success_aggregate(
    actual_receipts, value
):
    directory, target, original = actual_receipts
    reports = copy.deepcopy(original)
    reports["prediction"]["predictions"][0]["probability"] = value
    with pytest.raises(ValueError):
        demo._validate_reports(directory, target, VARIANT, reports)


@pytest.mark.parametrize(
    "field", ["alert", "prediction_threshold", "evaluation_threshold", "brier", "log_loss"]
)
def test_consistent_model_strings_cannot_hide_inconsistent_deployment_results(
    actual_receipts, field
):
    directory, target, original = actual_receipts
    reports = copy.deepcopy(original)
    row = reports["prediction"]["predictions"][0]
    metrics = reports["evaluation"]["metrics"]
    if field == "alert":
        row["alert"] = not row["alert"]
    elif field == "prediction_threshold":
        row["threshold"] = 0.0 if row["threshold"] != 0.0 else 1.0
    elif field == "evaluation_threshold":
        metrics["threshold"] = 0.0 if metrics["threshold"] != 0.0 else 1.0
    elif field == "brier":
        metrics["conversation_weighted_prefix_brier"] += 0.1
    else:
        metrics["conversation_weighted_prefix_log_loss"] += 0.1
    with pytest.raises(ValueError):
        demo._validate_reports(directory, target, VARIANT, reports)


def test_raw_text_cannot_hide_inside_the_metrics_copied_to_aggregate(actual_receipts):
    directory, target, original = actual_receipts
    reports = copy.deepcopy(original)
    reports["evaluation"]["metrics"]["raw_source_text"] = "PRIVATE_ORACLE_MARKER"
    with pytest.raises(ValueError):
        demo._validate_reports(directory, target, VARIANT, reports)


@pytest.mark.parametrize("mode", ["clean", "attempt_import", "already_present"])
def test_actual_child_guard_mechanism_rejects_torch_even_without_installed_torch(mode):
    # A tiny original stand-in isolates the guard itself, not the real CLI.
    # Actual deployment/command isolation remains the installed-demo acceptance.
    setup = f"""
import sys, types
package = types.ModuleType('turnscope')
package.__path__ = []
cli = types.ModuleType('turnscope.cli')
def main(arguments):
    if {mode!r} == 'attempt_import':
        __import__('torch')
    return 0
cli.main = main
sys.modules['turnscope'] = package
sys.modules['turnscope.cli'] = cli
if {mode!r} == 'already_present':
    sys.modules['torch'] = types.ModuleType('torch')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", "-c", setup + demo._FROZEN_COMMAND],
        capture_output=True,
        encoding="utf-8",
        timeout=15,
    )
    if mode == "clean":
        assert result.returncode == 0 and not result.stderr
    else:
        assert result.returncode != 0
        assert "AssertionError: frozen control" in result.stderr
        assert "Torch" in result.stderr
