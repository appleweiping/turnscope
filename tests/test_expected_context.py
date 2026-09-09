from __future__ import annotations

import hashlib
import json
import math
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone

import numpy as np
import pytest

from turnscope import Conversation, ExpectedContextModel, Utterance, iter_context_pairs
from turnscope.cli import main
from turnscope.io import conversation_to_dict


def utterance(identifier, text, parent=None):
    return Utterance(identifier, "user", text, datetime(2026, 1, 1, tzinfo=timezone.utc), parent)


def training():
    return [
        Conversation("a", [utterance("s", "ask"), utterance("c", "yes", "s")]),
        Conversation("b", [utterance("s", "tell"), utterance("c", "no", "s")]),
    ]


def trained(**kwargs):
    return ExpectedContextModel(n_components=2, **kwargs).fit(training())


def resign(artifact):
    artifact["sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "sha256"},
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return artifact


def test_hand_calculated_ridge_is_not_a_training_mean():
    model = trained()
    state = model.state
    assert state.source_tfidf.vocabulary == ("ask", "tell")
    assert state.context_tfidf.vocabulary == ("no", "yes")
    # Work in original context coordinates to avoid assumptions about tied SVD axes.
    basis = np.array(state.basis)
    assert np.array(model.predict("ASK").vector) @ basis.T == pytest.approx([0.25, 0.75])
    assert np.array(model.predict("tell").vector) @ basis.T == pytest.approx([0.75, 0.25])
    assert np.array(state.context_mean) @ basis.T == pytest.approx([0.5, 0.5])
    assert state.training_mse == pytest.approx(0.0625)
    assert state.mean_baseline_mse == pytest.approx(0.25)
    result = model.evaluate(training())
    assert result["mse"] == pytest.approx(0.0625)
    assert result["training_mean_baseline_mse"] == pytest.approx(0.25)
    assert result["pairs"] == 2
    assert result["zero_source_vectors"] == result["zero_context_vectors"] == 0


def test_independent_augmented_least_squares_oracle():
    # Source tokens have equal DF, so normalized rows can be calculated without runtime TF-IDF.
    texts = ["a a b", "a c", "b c c", "a b c"]
    targets = ["red", "blue", "red blue", "red red blue"]
    documents = [
        Conversation(str(i), [utterance("s", text), utterance("c", target, "s")])
        for i, (text, target) in enumerate(zip(texts, targets, strict=True))
    ]
    penalty = 0.7
    model = ExpectedContextModel(n_components=2, regularization=penalty).fit(documents)
    x = np.array([[2, 1, 0], [1, 0, 1], [0, 1, 2], [1, 1, 1]], dtype=float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = np.array([[0, 1], [1, 0], [1, 1], [1, 2]], dtype=float)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    # Solve an augmented least-squares problem directly in original context coordinates.
    # This is independent of both runtime SVD and its normal-equation implementation.
    design = np.column_stack([np.ones(4), x])
    regularizer = np.column_stack([np.zeros(3), math.sqrt(penalty) * np.eye(3)])
    augmented = np.vstack([design, regularizer])
    outcome = np.vstack([y, np.zeros((3, 2))])
    solution = np.linalg.lstsq(augmented, outcome, rcond=None)[0]
    basis = np.array(model.state.basis)
    actual = np.array([model.predict(text).vector for text in texts]) @ basis.T
    assert actual == pytest.approx(design @ solution, abs=1e-12)
    assert model.state.training_mse == pytest.approx(np.mean((actual - y) ** 2))
    assert model.state.mean_baseline_mse == pytest.approx(np.mean((y - y.mean(axis=0)) ** 2))
    # A separate normal-equation oracle verifies centering and the unpenalized intercept.
    centered_x, centered_y = x - x.mean(axis=0), y - y.mean(axis=0)
    weights = (
        np.linalg.inv(centered_x.T @ centered_x + penalty * np.eye(3)) @ centered_x.T @ centered_y
    )
    assert actual == pytest.approx(centered_x @ weights + y.mean(axis=0), abs=1e-12)


def test_branch_endpoints_fit_once_but_edges_weight_regression():
    document = Conversation(
        "branch",
        [utterance("p", "ask"), utterance("a", "yes", "p"), utterance("b", "no", "p")],
    )
    model = ExpectedContextModel().fit([document])
    assert model.state.training_pairs == 2
    assert model.state.source_tfidf.documents == 1
    assert model.state.context_tfidf.documents == 2
    assert model.predict("ask").vector == model.state.context_mean
    reverse = ExpectedContextModel(relation="predecessor").fit([document])
    assert reverse.state.source_tfidf.documents == 2
    assert reverse.state.context_tfidf.documents == 1
    assert reverse.state.dimensions == 1
    assert reverse.state.mean_baseline_mse == 0
    assert reverse.predict("yes").vector == pytest.approx((1,))


def test_reply_and_sequence_contracts_and_input_order():
    messages = [utterance("r", "root"), utterance("a", "alpha", "r"), utterance("b", "beta", "r")]
    document = Conversation("c", messages)
    pairs = list(iter_context_pairs([document]))
    assert {(p.source.id, p.context.id) for p in pairs} == {("r", "a"), ("r", "b")}
    assert {
        (p.source.id, p.context.id) for p in iter_context_pairs([document], relation="predecessor")
    } == {("a", "r"), ("b", "r")}
    assert [
        (p.source.id, p.context.id)
        for p in iter_context_pairs([document], relation="sequence-successor")
    ] == [("r", "a"), ("a", "b")]
    assert (
        ExpectedContextModel().fit([document]).to_dict()
        == ExpectedContextModel().fit([Conversation("c", list(reversed(messages)))]).to_dict()
    )
    unlinked = Conversation("plain", [utterance("1", "ask"), utterance("2", "yes")])
    assert list(iter_context_pairs([unlinked])) == []
    with pytest.raises(ValueError, match="at least one"):
        ExpectedContextModel().fit([unlinked])
    assert (
        ExpectedContextModel(relation="sequence-successor").fit([unlinked]).state.training_pairs
        == 1
    )
    assert (
        trained().to_dict()
        == ExpectedContextModel(n_components=2).fit(reversed(training())).to_dict()
    )


@pytest.mark.parametrize(
    "documents,match",
    [
        ([Conversation("x", [utterance("a", "a", "missing")])], "unknown reply"),
        ([Conversation("x", [utterance("a", "a", "b"), utterance("b", "b", "a")])], "cycle"),
        ([Conversation("x", [utterance("a", "a"), utterance("a", "b")])], "unique"),
        ([Conversation("x", []), Conversation("x", [])], "unique"),
    ],
)
def test_invalid_edges_or_identity_are_not_repaired(documents, match):
    with pytest.raises(ValueError, match=match):
        list(iter_context_pairs(documents))


def test_frozen_holdout_oov_projection_and_atomic_fit():
    model = trained()
    artifact = model.to_dict()
    heldout = Conversation(
        "heldout", [utterance("x", "ASK unknown unknown"), utterance("y", "unseen")]
    )
    result = model.transform(heldout)
    assert result["x"].vector == model.predict("ask").vector
    assert result["x"].to_dict()["coverage"] == pytest.approx(1 / 3)
    assert result["y"].coverage == 0
    assert model.predict("").vector == result["y"].vector
    assert model.predict("").tokens == 0
    assert model.project_context("unseen").vector == (0, 0)
    assert model.project_context("YES unseen").coverage == 0.5
    assert model.to_dict() == artifact
    with pytest.raises(TypeError):
        result["x"] = result["y"]
    with pytest.raises(FrozenInstanceError):
        model.state.training_pairs = 9
    with pytest.raises(TypeError):
        model.state.source_tfidf.idf["ask"] = 9
    with pytest.raises(ValueError, match="at least one"):
        model.fit([])
    assert model.to_dict() == artifact
    # Returned serializable values are deep snapshots, not parameter aliases.
    artifact["state"]["coefficients"][0][0] = 999
    assert model.to_dict() != artifact
    unknown = Conversation("test", [utterance("s", "zz"), utterance("c", "zz", "s")])
    assert model.evaluate([unknown])["zero_source_vectors"] == 1
    assert model.evaluate([unknown])["zero_context_vectors"] == 1


def test_oov_intercept_is_not_always_context_mean():
    documents = [
        Conversation(str(i), [utterance("s", text), utterance("c", target, "s")])
        for i, (text, target) in enumerate([("a", "x"), ("a", "x"), ("b", "y")])
    ]
    model = ExpectedContextModel().fit(documents)
    expected = np.array(model.state.context_mean) - np.array(model.state.source_mean) @ np.array(
        model.state.coefficients
    )
    assert model.predict("unknown").vector == pytest.approx(expected)
    assert model.predict("unknown").vector != pytest.approx(model.state.context_mean)


@pytest.mark.parametrize("source,context", [("", "x"), ("x", "!!!"), ("!!!", "")])
def test_empty_training_vocabulary_rejected(source, context):
    with pytest.raises(ValueError, match="non-empty"):
        ExpectedContextModel().fit(
            [Conversation("x", [utterance("s", source), utterance("c", context, "s")])]
        )


def test_rank_deficient_constant_training_and_feature_pruning():
    documents = [
        Conversation(str(i), [utterance("s", "a b"), utterance("c", "x y", "s")]) for i in range(3)
    ]
    model = ExpectedContextModel(n_components=20).fit(documents)
    assert model.state.dimensions == 1
    assert np.array(model.state.coefficients) == pytest.approx(np.zeros((2, 1)))
    assert model.predict("a").vector == pytest.approx((1,))
    assert model.state.training_mse == pytest.approx(0)
    assert ExpectedContextModel.from_dict(model.to_dict()).to_dict() == model.to_dict()
    capped = ExpectedContextModel(max_features=1).fit(documents)
    assert capped.state.source_tfidf.vocabulary == ("a",)
    with pytest.raises(ValueError, match="non-empty"):
        ExpectedContextModel(min_document_frequency=4).fit(documents)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_components": True},
        {"n_components": 0},
        {"n_components": 257},
        {"max_features": 1025},
        {"max_features": 1.5},
        {"min_document_frequency": 0},
        {"max_training_pairs": 20001},
        {"max_dense_cells": 64000001},
        {"regularization": True},
        {"regularization": 0},
        {"regularization": -1},
        {"regularization": float("nan")},
        {"regularization": float("inf")},
        {"regularization": "1"},
        {"regularization": 10**500},
        {"relation": "adjacent"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ExpectedContextModel(**kwargs)


def test_budgets_and_validation_precede_dense_allocation(monkeypatch):
    from turnscope import _context_numeric

    def forbidden(*args, **kwargs):
        pytest.fail("dense fit must not be called")

    monkeypatch.setattr(_context_numeric, "fit_numeric_context", forbidden)
    with pytest.raises(ValueError, match="max_training_pairs"):
        ExpectedContextModel(max_training_pairs=1).fit(training())
    with pytest.raises(ValueError, match="max_dense_cells"):
        ExpectedContextModel(max_dense_cells=1).fit(training())
    with pytest.raises(TypeError, match="Conversation"):
        ExpectedContextModel().fit([None])
    with pytest.raises(ValueError, match="relation"):
        list(iter_context_pairs([], relation="invalid"))


def test_unfitted_and_bad_prediction_input():
    model = ExpectedContextModel()
    for action in (
        lambda: model.state,
        model.to_dict,
        lambda: model.predict("x"),
        lambda: model.transform(Conversation("x", [])),
    ):
        with pytest.raises(ValueError, match="fitted"):
            action()
    model = trained()
    with pytest.raises(TypeError, match="string"):
        model.predict(3)
    with pytest.raises(TypeError, match="Conversation"):
        model.transform(None)
    with pytest.raises(ValueError, match="unique"):
        model.transform(Conversation("x", [utterance("x", "a"), utterance("x", "b")]))
    with pytest.raises(ValueError, match="at least one"):
        model.evaluate([])


def test_numerical_failures_preserve_previous_model(monkeypatch):
    from turnscope import _context_numeric

    model = trained()
    before = model.to_dict()
    original = _context_numeric.np.linalg.svd
    monkeypatch.setattr(
        _context_numeric.np.linalg,
        "svd",
        lambda *a, **k: (_ for _ in ()).throw(np.linalg.LinAlgError("fail")),
    )
    with pytest.raises(ValueError, match="numerical fit failed"):
        model.fit(training())
    assert model.to_dict() == before
    monkeypatch.setattr(_context_numeric.np.linalg, "svd", original)
    monkeypatch.setattr(_context_numeric.np.linalg, "solve", lambda a, b: np.full_like(b, np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        model.fit(training())
    assert model.to_dict() == before


def test_private_numerical_rank_zero_rejected():
    from turnscope._context_numeric import fit_numeric_context

    with pytest.raises(ValueError, match="rank zero"):
        fit_numeric_context([[1.0]], [[0.0]], [0], 1, 1.0)


def test_prediction_checks_nonfinite_and_overflow():
    model = trained()
    model._state = replace(model.state, coefficients=((float("inf"), 0), (0, 0)))
    with pytest.raises(ValueError, match="finite numeric"):
        model.predict("ask")
    model._state = replace(
        model.state, coefficients=((1.7e308, 0), (1.7e308, 0)), source_mean=(0, 0)
    )
    with pytest.raises(ValueError, match="finite numeric"):
        model.predict("ask tell")


def test_artifact_roundtrip_and_file_boundaries(tmp_path, monkeypatch):
    model = trained()
    path = tmp_path / "model.json"
    model.save(path)
    loaded = ExpectedContextModel.load(path)
    assert loaded.to_dict() == model.to_dict()
    assert loaded.digest == model.digest
    assert loaded.predict("ask").vector == model.predict("ask").vector
    assert loaded.evaluate(training()) == model.evaluate(training())
    from turnscope import context_artifact

    monkeypatch.setattr(context_artifact, "MAX_CONTEXT_MODEL_BYTES", 2)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="32 MiB"):
        model.save(path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="32 MiB"):
        ExpectedContextModel.load(path)
    with pytest.raises(ValueError, match="cannot load"):
        ExpectedContextModel.load(tmp_path / "missing.json")
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="cannot load"):
        ExpectedContextModel.load(path)


@pytest.mark.parametrize("raw", ['{"format": 1, "format": 2}', "NaN", "[", "{"])
def test_malformed_json_rejected(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        ExpectedContextModel.load(path)


@pytest.mark.parametrize(
    "section,key,value",
    [
        (None, "format", "v999"),
        (None, "unexpected", 1),
        (None, "sha256", "0" * 64),
        ("config", "n_components", False),
        ("config", "regularization", True),
        ("config", "extra", 1),
        ("state", "extra", 1),
        ("state", "basis", [[1]]),
        ("state", "basis", [[0, 0], [0, 0]]),
        ("state", "basis", [[-1, 0], [0, 1]]),
        ("state", "basis", [[2, 0], [0, 1]]),
        ("state", "coefficients", [[1, 2]]),
        ("state", "coefficients", [[1], [2]]),
        ("state", "coefficients", [[True, 0], [0, 1]]),
        ("state", "coefficients", [[float("nan"), 0], [0, 1]]),
        ("state", "coefficients", [[10**500, 0], [0, 1]]),
        ("state", "source_mean", [-1, 0]),
        ("state", "source_mean", [1, 1]),
        ("state", "context_mean", [2, 0]),
        ("state", "context_mean", [1, 1]),
        ("state", "singular_values", []),
        ("state", "singular_values", [1, 1, 1]),
        ("state", "singular_values", [0, 1]),
        ("state", "singular_values", [0.5, 1]),
        ("state", "singular_values", [3, 1]),
        ("state", "training_pairs", 0),
        ("state", "training_mse", -1),
        ("state", "training_mse", 0.9),
        ("state", "mean_baseline_mse", 2),
        ("state", "estimated_dense_cells", 1),
        ("state", "estimated_dense_cells", True),
        ("source_tfidf", "features", []),
        ("context_tfidf", "documents", 3),
        ("source_tfidf", "max_features", 4),
        ("context_tfidf", "min_df", 2),
    ],
)
def test_corrupt_artifacts_are_rejected_before_use(section, key, value):
    artifact = trained().to_dict()
    (artifact if section is None else artifact[section])[key] = value
    with pytest.raises(ValueError):
        ExpectedContextModel.from_dict(artifact)


def test_schema_checks_apply_even_with_recomputed_checksum():
    artifact = trained().to_dict()
    artifact["state"]["basis"] = [[0.5, 0], [0, 0.5]]
    with pytest.raises(ValueError, match="orthonormal"):
        ExpectedContextModel.from_dict(resign(artifact))
    artifact = trained().to_dict()
    artifact["context_tfidf"] = None
    with pytest.raises(ValueError, match="nested"):
        ExpectedContextModel.from_dict(resign(artifact))
    with pytest.raises(ValueError, match="fields"):
        ExpectedContextModel.from_dict([])
    artifact = trained().to_dict()
    artifact["config"]["max_dense_cells"] = 1
    with pytest.raises(ValueError, match="workspace budget"):
        ExpectedContextModel.from_dict(resign(artifact))


def test_finite_parameters_with_overflowing_loss_raise_domain_error():
    artifact = trained().to_dict()
    artifact["state"]["coefficients"] = [[1e308, 0], [0, 0]]
    model = ExpectedContextModel.from_dict(resign(artifact))
    assert all(math.isfinite(value) for value in model.predict("ask").vector)
    with pytest.raises(ValueError, match="evaluation exceeded finite"):
        model.evaluate(training())


def test_atomic_save_keeps_existing_target_on_replace_failure(tmp_path, monkeypatch):
    from pathlib import Path

    target = tmp_path / "model.json"
    target.write_text("existing", encoding="utf-8")

    def fail_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        trained().save(target)
    assert target.read_text() == "existing"
    assert list(tmp_path.iterdir()) == [target]


def test_cli_fit_transform_evaluate_and_input_protection(tmp_path, capsys):
    source, model_path, output = (
        tmp_path / name for name in ("train.json", "model.json", "result.json")
    )
    source.write_text(
        json.dumps([conversation_to_dict(item) for item in training()]), encoding="utf-8"
    )
    assert main(["context", "fit", str(source), str(model_path), "--components", "2"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["training_pairs"] == 2
    assert main(["context", "transform", str(model_path), str(source)]) == 0
    transformed = json.loads(capsys.readouterr().out)
    assert transformed["model_digest"] == summary["model_digest"]
    assert len(transformed["predictions"]) == 2
    assert "text" not in transformed["predictions"][0]["predictions"]["s"]
    assert main(["context", "evaluate", str(model_path), str(source), "-o", str(output)]) == 0
    assert json.loads(output.read_text())["mse"] == pytest.approx(0.0625)
    before = {path: path.read_bytes() for path in (source, model_path)}
    for args in (
        ["fit", str(source), str(source)],
        ["transform", str(model_path), str(source), "-o", str(model_path)],
        ["evaluate", str(model_path), str(source), "-o", str(source)],
    ):
        assert main(["context", *args]) == 2
        assert "differ" in capsys.readouterr().err
    assert before == {path: path.read_bytes() for path in before}
    duplicated = [conversation_to_dict(training()[0])] * 2
    source.write_text(json.dumps(duplicated), encoding="utf-8")
    output.write_text("unchanged", encoding="utf-8")
    assert main(["context", "transform", str(model_path), str(source), "-o", str(output)]) == 2
    assert "unique" in capsys.readouterr().err
    assert output.read_text() == "unchanged"


def test_cli_missing_optional_fitter_is_a_clean_error(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([conversation_to_dict(item) for item in training()]), encoding="utf-8"
    )

    def missing(*args):
        raise ImportError("install turnscope[context]")

    monkeypatch.setattr(ExpectedContextModel, "fit", missing)
    assert main(["context", "fit", str(source), str(tmp_path / "model.json")]) == 2
    assert "install turnscope[context]" in capsys.readouterr().err
