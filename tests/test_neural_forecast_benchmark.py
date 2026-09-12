"""Driver stage/failure boundaries; these controls do not train or score CGA."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from turnscope import NeuralSaveResult

PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_neural_forecast.py"
SPEC = importlib.util.spec_from_file_location("neural_benchmark_driver", PATH)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


@pytest.mark.parametrize("seed", [17, 101, 202])
def test_protocol_is_fresh_closed_and_does_not_promise_final_evaluation(seed):
    plan = bench.fit_protocol(seed)
    assert plan["training_config"]["seed"] == seed
    assert plan["training_config"]["epochs"] == 8
    assert plan["model_config"]["embedding_dim"] == 64
    assert plan["source_partition_protocol"]["policy_validation_buckets"] == [0, 1]
    assert plan["seed_selection_by_test_results"] is False
    assert plan["real_data_quality_claimed"] is False
    assert len(plan["required_later_comparisons"]) == 6
    assert "previously inspected" in plan["test_cohort_history"]
    plan["training_config"]["seed"] = 99
    assert bench.fit_protocol(seed)["training_config"]["seed"] == seed


@pytest.mark.parametrize("seed", [True, 0, 1, -1, 17.0, "17", None])
def test_no_unregistered_seed_or_coercion(seed):
    with pytest.raises(ValueError, match="preregistered"):
        bench.fit_protocol(seed)


@pytest.fixture
def driver_control(tmp_path, monkeypatch):
    # Explicit non-numerical driver fixture. Actual artifact/CLI/training
    # acceptance lives in their own tests and installed-workflow verification.
    training, validation, policy, heldout = object(), object(), object(), object()
    expected_plan = bench.fit_protocol(17)
    prepared = {
        name: {
            "partition_digest": str(index) * 64,
            "audit": {"eligible_conversations": 2, "prefixes": 3},
        }
        for index, name in enumerate(("training", "validation", "policy_validation", "test"), 1)
    }
    expected_summary: dict[str, Any] = {
        "fixture": "not real training evidence",
        "config": expected_plan["model_config"],
        "training_config": expected_plan["training_config"],
        "eligibility_policy": expected_plan["eligibility_policy"],
        "data_limits": expected_plan["data_limits"],
        "numeric_limits": expected_plan["numeric_limits"],
        "policy_reuses_model_validation": False,
        "torch_threads": 1,
        "probability_calibration_claimed": False,
    }
    for name, partition in (
        ("training", "training"),
        ("model_validation", "validation"),
        ("policy_validation", "policy_validation"),
    ):
        expected_summary[name + "_partition_digest"] = prepared[partition]["partition_digest"]
        expected_summary[name + "_conversations"] = 2
        expected_summary[name + "_prefixes"] = 3
    partitions = SimpleNamespace(
        training=training,
        validation=validation,
        policy_validation=policy,
        test=heldout,
        prepared_audit=lambda: prepared,
        audit_dict=lambda: {"fixture": "authored driver control"},
    )
    seen = {"fit": 0, "save": 0, "threads": 7, "deterministic": False, "warn_only": True}

    def set_threads(value):
        seen["threads"] = value

    def set_deterministic(value, *, warn_only=False):
        seen["deterministic"], seen["warn_only"] = value, warn_only
        if seen.get("fail_setup") and value:
            raise RuntimeError("injected setup failure")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            get_num_threads=lambda: seen["threads"],
            set_num_threads=set_threads,
            are_deterministic_algorithms_enabled=lambda: seen["deterministic"],
            is_deterministic_algorithms_warn_only_enabled=lambda: seen["warn_only"],
            use_deterministic_algorithms=set_deterministic,
        ),
    )

    class DriverModel:
        digest = "a" * 64
        training_summary: ClassVar[dict[str, Any]] = expected_summary

        def __init__(self, **kwargs):
            assert kwargs["training_config"].seed == 17

        def fit(self, supplied_training, supplied_validation, *, policy_validation):
            assert (supplied_training, supplied_validation, policy_validation) == (
                training,
                validation,
                policy,
            )
            assert seen["threads"] == 1 and seen["deterministic"] is True
            seen["fit"] += 1
            if seen.get("fail_fit"):
                raise RuntimeError("injected fitting failure")
            return self

        def evaluate(self, *_):
            raise AssertionError("this stage must never score heldout")

        def save(self, path):
            seen["save"] += 1
            path.write_bytes(b"not-a-real-model;driver-control")
            return NeuralSaveResult(str(path), bench._sha(path.read_bytes()), path.stat().st_size)

        @staticmethod
        def load(path):
            assert path.read_bytes() == b"not-a-real-model;driver-control"
            return SimpleNamespace(
                digest="b" * 64 if seen.get("fail_reload") else DriverModel.digest,
                training_summary=DriverModel.training_summary,
            )

    adapter = SimpleNamespace(protocol=lambda: {}, load_neural_cga=lambda _: partitions)
    monkeypatch.setattr(bench, "_adapter", lambda: adapter)
    monkeypatch.setattr(bench, "HierarchicalEventForecaster", DriverModel)
    monkeypatch.setattr(bench, "source_bindings", lambda: {"authored": "a" * 64})
    return tmp_path / "not-read-archive", tmp_path / "private-run", seen


def test_candidate_stage_never_calls_final_evaluation_and_restores_process_policy(driver_control):
    archive, directory, seen = driver_control
    result = bench.fit_candidate(archive, directory)
    assert seen == {"fit": 1, "save": 1, "threads": 7, "deterministic": False, "warn_only": True}
    assert result["candidate_fit_completed"] is True
    assert result["official_test_predictions_produced"] is False
    assert result["final_evaluation_completed"] is False
    assert json.loads((directory / "candidate-receipt.json").read_bytes()) == result
    assert (directory / "fit-plan.json").is_file()
    assert (directory / "source-audit.json").is_file()


@pytest.mark.parametrize("failure", ["fail_setup", "fail_fit"])
def test_failed_fit_restores_policy_and_leaves_no_success_receipt(driver_control, failure):
    archive, directory, seen = driver_control
    seen[failure] = True
    with pytest.raises(RuntimeError, match="injected"):
        bench.fit_candidate(archive, directory)
    assert seen["threads"] == 7 and seen["deterministic"] is False and seen["warn_only"] is True
    assert not (directory / "candidate-receipt.json").exists()
    assert not (directory / "private-model.tsn").exists()
    assert (directory / "fit-plan.json").is_file()


def test_source_change_rejects_before_publication(driver_control, monkeypatch):
    archive, directory, seen = driver_control
    bindings = iter([{"source": "a" * 64}, {"source": "b" * 64}])
    monkeypatch.setattr(bench, "source_bindings", lambda: next(bindings))
    with pytest.raises(RuntimeError, match="source changed"):
        bench.fit_candidate(archive, directory)
    assert seen["save"] == 0
    assert not (directory / "candidate-receipt.json").exists()


def test_published_model_survives_failed_reload_without_success_receipt(driver_control):
    archive, directory, seen = driver_control
    seen["fail_reload"] = True
    with pytest.raises(RuntimeError, match="published candidate"):
        bench.fit_candidate(archive, directory)
    assert (directory / "private-model.tsn").is_file()
    assert not (directory / "candidate-receipt.json").exists()


def test_no_output_inside_checkout_and_no_reuse(driver_control):
    archive, directory, seen = driver_control
    with pytest.raises(ValueError, match="outside"):
        bench.fit_candidate(archive, PATH.parent / "private-forbidden")
    directory.mkdir()
    with pytest.raises(FileExistsError):
        bench.fit_candidate(archive, directory)
    assert seen["fit"] == 0


def test_closed_stdout_is_failure_after_candidate_receipt(driver_control, monkeypatch):
    archive, directory, _ = driver_control
    closed = io.StringIO()
    closed.close()
    if sys.version_info >= (3, 14):
        import _colorize

        monkeypatch.setattr(_colorize, "can_colorize", lambda **_: closed.isatty())
    monkeypatch.setattr(sys, "stdout", closed)
    assert bench.main([str(archive), str(directory)]) == 1
    assert (directory / "candidate-receipt.json").is_file()
