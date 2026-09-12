"""Independent authored driver faults; no CGA loading or numerical training.

Real causal preparation supplies the fixture's inventory and partition digests.
The model/CPU policy objects below are explicit non-numerical control doubles,
not evidence of a trained candidate or real-data quality.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from turnscope import NeuralSaveResult
from turnscope.models import Conversation, Utterance
from turnscope.neural_forecast import NeuralForecastConfig
from turnscope.neural_forecast_data import (
    SequenceLimits,
    SequencePolicy,
    prepare_sequence_forecasts,
)
from turnscope.neural_forecast_math import NeuralNumericLimits

PATH = Path(__file__).resolve().parents[1] / "benchmarks/benchmark_neural_forecast.py"
SPEC = importlib.util.spec_from_file_location("review_neural_candidate_driver", PATH)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def _partition(name):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return tuple(
        Conversation(
            f"{name}-{index}",
            tuple(
                Utterance(
                    str(position),
                    "participant",
                    text,
                    base + timedelta(seconds=position),
                    metadata={"event": index % 2 == 0 and position == 2},
                )
                for position, text in enumerate(("a question", "a reply", "future text"))
            ),
            metadata={"forecast_groups": [f"scope:{name}:{index}"]},
        )
        for index in range(4)
    )


@pytest.fixture
def controlled_fit(tmp_path, monkeypatch):
    source = {name: _partition(name) for name in ("training", "validation", "policy_validation")}
    data = {name: prepare_sequence_forecasts(rows) for name, rows in source.items()}
    audit = {
        name: {
            "partition_digest": value.digest,
            "group_count": len(value.group_digests),
            "audit": value.audit.to_dict(),
        }
        for name, value in data.items()
    }
    audit["test"] = {"fixture": "audit only; no test model calls"}
    control = {
        "save_calls": 0,
        "load_calls": 0,
        "fit_calls": 0,
        "source_changed": False,
        "change_during": None,
        "summary_change": None,
        "threads": 9,
        "deterministic": False,
        "warn_only": True,
    }

    class Partitions:
        training = source["training"]
        validation = source["validation"]
        policy_validation = source["policy_validation"]

        @property
        def test(self):
            raise AssertionError("candidate fitting must not request final-test model inputs")

        @staticmethod
        def prepared_audit():
            return copy.deepcopy(audit)

        @staticmethod
        def audit_dict():
            return {"fixture": "authored driver control, not CGA origin"}

    class Model:
        digest = "a" * 64
        last_summary = None

        def __init__(self, *, training_config):
            self.summary = {
                "config": NeuralForecastConfig().to_dict(),
                "training_config": training_config.to_dict(),
                "eligibility_policy": SequencePolicy().to_dict(),
                "data_limits": SequenceLimits().to_dict(),
                "numeric_limits": NeuralNumericLimits().to_dict(),
                "policy_reuses_model_validation": False,
                "probability_calibration_claimed": False,
                "torch_threads": 1,
            }
            for name, key in (
                ("training", "training"),
                ("validation", "model_validation"),
                ("policy_validation", "policy_validation"),
            ):
                self.summary[f"{key}_partition_digest"] = data[name].digest
                self.summary[f"{key}_conversations"] = len(data[name].observations)
                self.summary[f"{key}_prefixes"] = len(data[name].examples)

        @property
        def training_summary(self):
            return copy.deepcopy(self.summary)

        def fit(self, training, validation, *, policy_validation):
            assert training is source["training"] and validation is source["validation"]
            assert policy_validation is source["policy_validation"]
            assert control["threads"] == 1 and control["deterministic"] is True
            control["fit_calls"] += 1
            if control["summary_change"] is not None:
                path, value = control["summary_change"]
                target = self.summary
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
            return self

        def save(self, path):
            control["save_calls"] += 1
            if control["change_during"] == "save":
                control["source_changed"] = True
            path.write_bytes(b"explicit-non-numerical-control")
            Model.last_summary = self.training_summary
            return NeuralSaveResult(str(path), bench._sha(path.read_bytes()), path.stat().st_size)

        @staticmethod
        def load(path):
            control["load_calls"] += 1
            if control["change_during"] == "load":
                control["source_changed"] = True
            assert path.read_bytes() == b"explicit-non-numerical-control"
            return SimpleNamespace(digest=Model.digest, training_summary=Model.last_summary)

        def predict(self, *_):
            raise AssertionError("candidate driver must not produce model test predictions")

        def evaluate(self, *_):
            raise AssertionError("candidate driver must not produce final-test metrics")

    def set_threads(value):
        control["threads"] = value

    def set_deterministic(value, *, warn_only=False):
        control["deterministic"], control["warn_only"] = value, warn_only

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            get_num_threads=lambda: control["threads"],
            set_num_threads=set_threads,
            are_deterministic_algorithms_enabled=lambda: control["deterministic"],
            is_deterministic_algorithms_warn_only_enabled=lambda: control["warn_only"],
            use_deterministic_algorithms=set_deterministic,
        ),
    )
    monkeypatch.setattr(bench, "HierarchicalEventForecaster", Model)
    monkeypatch.setattr(
        bench,
        "_adapter",
        lambda: SimpleNamespace(protocol=lambda: {}, load_neural_cga=lambda _: Partitions()),
    )
    monkeypatch.setattr(
        bench,
        "source_bindings",
        lambda: {"authored-source": ("b" if control["source_changed"] else "a") * 64},
    )
    return tmp_path / "unread-source.zip", tmp_path / "candidate", control, audit


def test_consistent_causal_state_passes_without_accessing_final_test(controlled_fit):
    archive, directory, control, _audit = controlled_fit
    receipt = bench.fit_candidate(archive, directory)
    assert control["fit_calls"] == control["save_calls"] == control["load_calls"] == 1
    assert receipt["training"]["training_conversations"] == 4
    assert receipt["official_test_predictions_produced"] is False
    assert receipt["final_evaluation_completed"] is False
    assert (control["threads"], control["deterministic"], control["warn_only"]) == (9, False, True)


@pytest.mark.parametrize("stage", ["save", "load"])
def test_source_change_during_artifact_work_prevents_success_receipt(controlled_fit, stage):
    archive, directory, control, _audit = controlled_fit
    control["change_during"] = stage
    with pytest.raises(RuntimeError, match="source changed"):
        bench.fit_candidate(archive, directory)
    # Publication may already have happened: retain the private candidate but
    # never issue a completed receipt that claims the old source inventory.
    assert (directory / "private-model.tsn").exists()
    assert not (directory / "candidate-receipt.json").exists()


@pytest.mark.parametrize(
    "path,value",
    [
        (("training_config", "seed"), 101),
        (("config", "word_hidden"), 32),
        (("training_partition_digest",), "0" * 64),
        (("model_validation_partition_digest",), "0" * 64),
        (("policy_validation_partition_digest",), "0" * 64),
        (("training_conversations",), 0),
        (("training_prefixes",), 0),
        (("policy_reuses_model_validation",), True),
        (("torch_threads",), 2),
    ],
)
def test_fitted_state_must_match_declared_protocol_and_prepared_inputs(controlled_fit, path, value):
    archive, directory, control, _audit = controlled_fit
    control["summary_change"] = (path, value)
    with pytest.raises((ValueError, RuntimeError)):
        bench.fit_candidate(archive, directory)
    assert control["save_calls"] == 0
    assert not (directory / "candidate-receipt.json").exists()


def test_prepared_audit_cannot_claim_a_different_training_inventory(controlled_fit):
    archive, directory, control, audit = controlled_fit
    audit["training"]["audit"]["eligible_conversations"] = 99
    with pytest.raises((ValueError, RuntimeError)):
        bench.fit_candidate(archive, directory)
    assert control["save_calls"] == 0


def test_final_receipt_delivery_failure_retains_private_model_without_success_claim(
    controlled_fit, monkeypatch
):
    archive, directory, control, _audit = controlled_fit
    publish = bench.publish_neural_report

    def fail_receipt(payload, output):
        if output is not None and output.name == "candidate-receipt.json":
            raise OSError("injected private receipt delivery failure")
        return publish(payload, output)

    monkeypatch.setattr(bench, "publish_neural_report", fail_receipt)
    with pytest.raises(OSError, match="injected"):
        bench.fit_candidate(archive, directory)
    assert control["save_calls"] == 1
    assert (directory / "private-model.tsn").exists()
    assert not (directory / "candidate-receipt.json").exists()
    assert (
        json.loads((directory / "fit-plan.json").read_bytes())["protocol"]["training_config"][
            "seed"
        ]
        == 17
    )
