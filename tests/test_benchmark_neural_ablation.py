"""Fit-driver boundary tests; authored sources are never claimed as real CGA."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from turnscope.models import Conversation, Utterance
from turnscope.neural_ablation_math import ABLATION_VARIANTS, ablation_parameter_shapes
from turnscope.neural_forecast_artifact import NeuralSaveResult
from turnscope.neural_forecast_math import NeuralArchitecture


def load_driver():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_neural_ablation.py"
    spec = importlib.util.spec_from_file_location("_authored_ablation_driver_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def driver():
    return load_driver()


def conversations(scope):
    return tuple(
        Conversation(
            f"{scope}-{index}",
            tuple(
                Utterance(
                    str(turn),
                    "person",
                    text,
                    datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=turn),
                    metadata={"event": index == 0 and turn == 2},
                )
                for turn, text in enumerate(
                    ("red bright" if index == 0 else "blue calm", "please explain", "future")
                )
            ),
            metadata={"forecast_groups": [f"group-{scope}-{index}"]},
        )
        for index in range(2)
    )


def authored_adapter(driver, monkeypatch):
    """Only the file loader is mocked; schema/preparation/partition digests are real."""
    helper = driver._helper()
    adapter = helper._adapter()
    native = adapter.NeuralCgaPartitions(
        conversations("t"),
        conversations("v"),
        conversations("p"),
        conversations("test-audit-only"),
        json.dumps(
            {
                "protocol": adapter.protocol(),
                "protocol_sha256": adapter.protocol_digest(),
                "source_archive_verified": True,
                "archive_sha256": adapter.ARCHIVE_SHA256,
                "fixture_notice": "authored source; archive verification flag is a test double",
            }
        ).encode(),
    )
    adapter.load_neural_cga = lambda _: native
    helper._adapter = lambda: adapter
    monkeypatch.setattr(driver, "_helper", lambda: helper)
    return native


class TorchPolicyDouble(ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.threads = 7
        self.deterministic = False
        self.warn = True

    def get_num_threads(self):
        return self.threads

    def set_num_threads(self, value):
        self.threads = value

    def are_deterministic_algorithms_enabled(self):
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self):
        return self.warn

    def use_deterministic_algorithms(self, value, *, warn_only=False):
        self.deterministic, self.warn = value, warn_only


@pytest.fixture
def flow(driver, monkeypatch):
    """NONNUMERIC driver double: actual model training is covered separately below."""
    partitions = authored_adapter(driver, monkeypatch)
    fake_torch = TorchPolicyDouble()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    calls = []
    state = {"summary_mutation": None, "fit_error": None, "load_mutation": None}

    class Model:
        def __init__(self, **kwargs):
            self.settings = kwargs
            self.digest = "a" * 64

        def fit(self, training, validation, *, policy_validation):
            assert training is partitions.training and validation is partitions.validation
            assert policy_validation is partitions.policy_validation
            assert (fake_torch.threads, fake_torch.deterministic, fake_torch.warn) == (
                1,
                True,
                False,
            )
            calls.append("fit")
            if state["fit_error"]:
                raise ValueError(state["fit_error"])
            protocol = driver.fit_protocol(
                self.settings["config"].variant, self.settings["training_config"].seed
            )
            prepared = partitions.prepared_audit()
            vocabulary = driver._vocabulary_pin(partitions, prepared)
            arch = NeuralArchitecture(vocabulary["size"])
            shapes = ablation_parameter_shapes(protocol["variant"], arch)
            import math

            summary = {
                "format": "turnscope.neural-ablation-forecast.v1",
                "config": protocol["model_config"],
                "reference_architecture": arch.to_dict(),
                "vocabulary_digest": vocabulary["digest"],
                "vocabulary_size": vocabulary["size"],
                "parameter_count": sum(math.prod(shape) for shape in shapes.values()),
                "canonical_main_parameter_count": arch.parameter_count,
                "fixed_pad_parameters": 64,
                "policy_reuses_model_validation": False,
                "probability_calibration_claimed": False,
                "torch_threads": 1,
                **{
                    name: protocol[name]
                    for name in (
                        "variant",
                        "training_config",
                        "eligibility_policy",
                        "data_limits",
                        "numeric_limits",
                        "pooling_limits",
                        "training_limits",
                        "initialization_version",
                        "main_initialization_version",
                        "numerical_version",
                    )
                },
            }
            for name, partition in (
                ("training", "training"),
                ("model_validation", "validation"),
                ("policy_validation", "policy_validation"),
            ):
                summary[name + "_partition_digest"] = prepared[partition]["partition_digest"]
                summary[name + "_conversations"] = prepared[partition]["audit"][
                    "eligible_conversations"
                ]
                summary[name + "_prefixes"] = prepared[partition]["audit"]["prefixes"]
            if state["summary_mutation"]:
                state["summary_mutation"](summary)
            self.training_summary = summary
            state["model"] = self
            return self

        def predict(self, *_):
            pytest.fail("driver produced prediction")

        def evaluate(self, *_):
            pytest.fail("driver evaluated official test")

    def save(model, path):
        calls.append("save")
        raw = b"AUTHORED NONMODEL BYTES FOR DRIVER TEST ONLY"
        with path.open("xb") as stream:
            stream.write(raw)
        return NeuralSaveResult(
            str(path.absolute()), hashlib.sha256(raw).hexdigest(), len(raw), None
        )

    def load(path):
        calls.append("load")
        model = state["model"]
        restored = SimpleNamespace(
            digest=model.digest, training_summary=json.loads(json.dumps(model.training_summary))
        )
        if state["load_mutation"]:
            state["load_mutation"](restored, path)
        return restored

    monkeypatch.setattr(driver, "AblationEventForecaster", Model)
    monkeypatch.setattr(driver, "save_ablation_forecaster", save)
    monkeypatch.setattr(driver, "load_ablation_forecaster", load)
    return driver, partitions, fake_torch, calls, state


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
@pytest.mark.parametrize("seed", [17, 101, 202])
def test_frozen_protocol_all_variants_seeds_and_defaults(driver, variant, seed):
    protocol = driver.fit_protocol(variant, seed)
    settings = protocol["training_config"]
    assert (
        settings["epochs"],
        settings["patience"],
        settings["batch_conversations"],
        settings["learning_rate"],
        settings["seed"],
    ) == (8, 3, 16, 0.001, seed)
    assert (
        settings["max_features"],
        settings["min_document_frequency"],
        settings["max_batch_token_positions"],
        settings["gradient_clip"],
    ) == (10000, 1, 8192, 5.0)
    assert protocol["reference_dimensions"] == {
        "embedding_dim": 64,
        "word_hidden": 64,
        "turn_hidden": 64,
        "word_layers": 1,
        "turn_layers": 1,
    }
    assert (
        protocol["model_config"]["variant"] == variant
        and protocol["model_config"]["max_turn_tokens"] == 128
    )
    assert protocol["model_config"]["long_turn_policy"] == "head"
    assert protocol["source_partition_protocol"]["policy_validation_buckets"] == [0, 1]
    assert protocol["source_partition_protocol"]["seed"] == "turnscope-neural-v1/2026-09-12"
    assert protocol["prior_trained_parameters_used"] is False
    assert protocol["real_data_quality_claimed"] is False


@pytest.mark.parametrize(
    "variant,seed",
    [
        (None, 17),
        (True, 17),
        ("main", 17),
        ("mean-word.v1", True),
        ("mean-word.v1", 0),
        ("mean-word.v1", 17.0),
    ],
)
def test_invalid_protocol_before_output(driver, tmp_path, variant, seed):
    with pytest.raises(ValueError):
        driver.fit_candidate(
            tmp_path / "archive", tmp_path / "candidate", variant=variant, seed=seed
        )
    assert not (tmp_path / "candidate").exists()


def test_actual_source_bindings_include_helpers_and_imported_package(driver, monkeypatch):
    binding = driver.source_bindings()
    for name in driver._HELPERS:
        path = Path(driver.__file__).with_name(name)
        assert binding["benchmarks/" + name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "src/turnscope/neural_ablation_artifact.py" in binding
    assert "src/turnscope/neural_ablation_train.py" in binding
    module = ModuleType("turnscope.foreign")
    module.__file__ = "D:/outside-package/foreign.py"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="mixed"):
        driver.source_bindings()


def test_full_nonnumeric_flow_pins_artifact_and_never_scores_test(flow, tmp_path):
    driver, _, torch, calls, _ = flow
    output = tmp_path / "candidate"
    result = driver.fit_candidate(tmp_path / "archive", output, variant="mean-word.v1", seed=17)
    assert calls == ["fit", "save", "load"]
    assert (torch.threads, torch.deterministic, torch.warn) == (7, False, True)
    assert result == json.loads((output / "candidate-receipt.json").read_text())
    assert result["candidate_fit_completed"] is True
    assert result["official_test_predictions_produced"] is False
    assert result["final_evaluation_completed"] is False
    assert result["real_data_quality_claimed"] is False
    plan = json.loads((output / "fit-plan.json").read_text())
    assert plan["protocol_sha256"] == result["protocol_sha256"]
    assert result["source_sha256"] == plan["source_sha256"] == driver.source_bindings()
    assert (
        result["publication"]["sha256"]
        == hashlib.sha256((output / "private-model.tsa").read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError):
        driver.fit_candidate(tmp_path / "archive", output, variant="mean-word.v1", seed=17)
    assert calls == ["fit", "save", "load"]


def test_fit_failure_restores_process_policy_and_keeps_plan(flow, tmp_path):
    driver, _, torch, calls, state = flow
    state["fit_error"] = "private-sensitive-input"
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="private-sensitive-input"):
        driver.fit_candidate(tmp_path / "archive", output, variant="order-erased.v1", seed=101)
    assert calls == ["fit"] and (torch.threads, torch.deterministic, torch.warn) == (7, False, True)
    assert (output / "fit-plan.json").is_file() and (output / "source-audit.json").is_file()
    assert not (output / "candidate-receipt.json").exists()


@pytest.mark.parametrize(
    "field",
    [
        "variant",
        "numerical_version",
        "initialization_version",
        "main_initialization_version",
        "vocabulary_digest",
        "vocabulary_size",
        "parameter_count",
        "canonical_main_parameter_count",
        "fixed_pad_parameters",
        "training_conversations",
        "policy_validation_prefixes",
        "training_partition_digest",
        "reference_architecture",
        "pooling_limits",
        "training_limits",
        "config",
        "torch_threads",
        "policy_reuses_model_validation",
    ],
)
def test_mismatched_actual_summary_rejected_before_publication(flow, tmp_path, field):
    driver, _, _, calls, state = flow

    def mutate(summary):
        prior = summary[field]
        summary[field] = (
            not prior
            if type(prior) is bool
            else True
            if type(prior) is int
            else {}
            if type(prior) is dict
            else "mismatch"
        )

    state["summary_mutation"] = mutate
    with pytest.raises(ValueError):
        driver.fit_candidate(
            tmp_path / "archive", tmp_path / "failed", variant="mean-word.v1", seed=17
        )
    assert calls == ["fit"]


@pytest.mark.parametrize("phase", ["before-fit", "after-fit", "after-load"])
def test_source_change_prevents_completed_receipt(flow, tmp_path, monkeypatch, phase):
    driver, _, _, calls, _ = flow
    original = driver.source_bindings
    bound = original()
    seen = 0
    threshold = {"before-fit": 2, "after-fit": 3, "after-load": 4}[phase]

    def change():
        nonlocal seen
        seen += 1
        return bound if seen < threshold else {**bound, "changed": "0" * 64}

    monkeypatch.setattr(driver, "source_bindings", change)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="bound source"):
        driver.fit_candidate(tmp_path / "archive", output, variant="current-turn.v1", seed=202)
    assert not (output / "candidate-receipt.json").exists()
    assert (
        calls
        == {"before-fit": [], "after-fit": ["fit"], "after-load": ["fit", "save", "load"]}[phase]
    )


@pytest.mark.parametrize("mutation", ["digest", "summary-bool", "file"])
def test_reload_corruption_does_not_publish_completed_receipt(flow, tmp_path, mutation):
    driver, _, _, _, state = flow

    def change(restored, path):
        if mutation == "digest":
            restored.digest = "0" * 64
        elif mutation == "summary-bool":
            restored.training_summary["torch_threads"] = True
        else:
            path.write_bytes(b"different-private-candidate")

    state["load_mutation"] = change
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        driver.fit_candidate(tmp_path / "archive", output, variant="mean-word.v1", seed=17)
    assert (output / "private-model.tsa").is_file() and not (
        output / "candidate-receipt.json"
    ).exists()


def test_final_receipt_failure_keeps_private_model_and_restored_flags(flow, tmp_path, monkeypatch):
    driver, _, torch, _, _ = flow
    original = driver.publish_neural_report

    def publish(payload, output):
        if output and output.name == "candidate-receipt.json":
            raise OSError("injected receipt write failure")
        return original(payload, output)

    monkeypatch.setattr(driver, "publish_neural_report", publish)
    output = tmp_path / "failed"
    with pytest.raises(OSError):
        driver.fit_candidate(tmp_path / "archive", output, variant="order-erased.v1", seed=17)
    assert (output / "private-model.tsa").is_file()
    assert (torch.threads, torch.deterministic, torch.warn) == (7, False, True)


@pytest.mark.parametrize(
    "field,value",
    [("source_archive_verified", 1), ("archive_sha256", "0" * 64), ("protocol_sha256", "0" * 64)],
)
def test_source_identity_mismatch_rejected_before_fitting(
    flow, tmp_path, monkeypatch, field, value
):
    driver, partitions, _, calls, _ = flow
    adapter = driver._helper()._adapter()
    audit = partitions.audit_dict()
    audit[field] = value
    altered = adapter.NeuralCgaPartitions(
        partitions.training,
        partitions.validation,
        partitions.policy_validation,
        partitions.test,
        json.dumps(audit).encode(),
    )
    adapter.load_neural_cga = lambda _: altered
    with pytest.raises(ValueError, match="source audit"):
        driver.fit_candidate(
            tmp_path / "archive", tmp_path / "failure", variant="mean-word.v1", seed=17
        )
    assert calls == []


def test_publication_receipt_wrong_hash_rejected(flow, tmp_path, monkeypatch):
    driver, _, _, calls, _ = flow
    original = driver.save_ablation_forecaster

    def mismatch(model, path):
        saved = original(model, path)
        return NeuralSaveResult(saved.path, "0" * 64, saved.bytes_written, None)

    monkeypatch.setattr(driver, "save_ablation_forecaster", mismatch)
    with pytest.raises(RuntimeError, match="publication receipt"):
        driver.fit_candidate(
            tmp_path / "archive", tmp_path / "failure", variant="mean-word.v1", seed=17
        )
    assert calls == ["fit", "save"]


def test_output_alias_checkout_and_existing_paths_rejected_without_optimization(flow, tmp_path):
    driver, _, _, calls, _ = flow
    checkout = Path(driver.__file__).resolve().parents[1]
    for target, archive in (
        (checkout / "private-output", tmp_path / "archive"),
        (tmp_path / "same", tmp_path / "same"),
    ):
        with pytest.raises(ValueError):
            driver.fit_candidate(archive, target, variant="order-erased.v1", seed=17)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("untouched")
    with pytest.raises(FileExistsError):
        driver.fit_candidate(tmp_path / "archive", existing, variant="order-erased.v1", seed=17)
    assert (existing / "keep.txt").read_text() == "untouched" and calls == []


def test_report_cleanup_warning_is_successful_and_receipt_not_mutated(
    flow, tmp_path, monkeypatch, capsys
):
    driver, _, _, _, _ = flow
    original = driver.publish_neural_report

    def cleanup_warning(payload, path):
        original(payload, path)
        return "published; private temporary cleanup failed"

    monkeypatch.setattr(driver, "publish_neural_report", cleanup_warning)
    output = tmp_path / "candidate"
    receipt = driver.fit_candidate(tmp_path / "archive", output, variant="order-erased.v1", seed=17)
    assert receipt == json.loads((output / "candidate-receipt.json").read_text())
    assert len(receipt["report_cleanup_warnings"]) == 2
    assert "receipt published" in capsys.readouterr().err


def test_stdout_failure_returns_nonzero_but_keeps_completed_receipt(flow, tmp_path, monkeypatch):
    driver, _, _, _, _ = flow
    original = driver.publish_neural_report

    def closed_stdout(payload, path):
        if path is None:
            raise OSError("private stdout detail")
        return original(payload, path)

    monkeypatch.setattr(driver, "publish_neural_report", closed_stdout)
    output = tmp_path / "candidate"
    assert (
        driver.main(
            [str(tmp_path / "archive"), str(output), "--variant", "order-erased.v1", "--seed", "17"]
        )
        == 1
    )
    assert (
        json.loads((output / "candidate-receipt.json").read_text())["candidate_fit_completed"]
        is True
    )


def test_cli_redacts_private_exception_and_requires_both_choices(
    driver, tmp_path, monkeypatch, capsys
):
    for missing in ([], ["--variant", "mean-word.v1"], ["--seed", "17"]):
        with pytest.raises(SystemExit):
            driver.main([str(tmp_path / "archive"), str(tmp_path / "out"), *missing])

    def fail(*_, **__):
        raise ValueError("PRIVATE raw record and path")

    monkeypatch.setattr(driver, "fit_candidate", fail)
    capsys.readouterr()
    assert (
        driver.main(
            [
                str(tmp_path / "archive"),
                str(tmp_path / "out"),
                "--variant",
                "mean-word.v1",
                "--seed",
                "17",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "PRIVATE" not in captured.err and "private output may remain" in captured.err


@pytest.mark.parametrize("variant", ABLATION_VARIANTS)
def test_genuine_authored_fit_distinct_artifact_reload_not_real_cga(
    driver, tmp_path, monkeypatch, variant
):
    torch = pytest.importorskip("torch", reason="optional native CPU training dependency")
    authored_adapter(driver, monkeypatch)
    flags = (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    output = tmp_path / "actual-candidate"
    report = driver.fit_candidate(
        tmp_path / "mock-loader-not-cga.zip", output, variant=variant, seed=17
    )
    assert report["training"]["changed_parameter_names"]
    assert report["training"]["training_config"]["epochs"] == 8
    assert report["training"]["training_conversations"] == 2
    restored = driver.load_ablation_forecaster(output / "private-model.tsa")
    assert restored.digest == report["model_digest"]
    assert restored.training_summary == report["training"]
    assert report["official_test_predictions_produced"] is False
    assert flags == (
        torch.get_num_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
