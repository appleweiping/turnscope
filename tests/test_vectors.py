from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from turnscope import (
    Conversation,
    SparseSimilarityIndex,
    TfidfVectorizer,
    Utterance,
    normalize_sparse,
    sparse_cosine,
)
from turnscope.cli import main
from turnscope.io import conversation_to_dict


def conversation(identifier: str, *texts: str) -> Conversation:
    return Conversation(
        identifier,
        [
            Utterance(str(index), "user", text, datetime(2026, 1, 1, tzinfo=timezone.utc))
            for index, text in enumerate(texts)
        ],
    )


def trained() -> TfidfVectorizer:
    return TfidfVectorizer().fit(
        [conversation("train-a", "red red", "blue"), conversation("train-b", "blue green")]
    )


def test_arithmetic_conversation_tf_idf_and_legacy_utterance_projection() -> None:
    model = trained()
    rare = math.log(3 / 2) + 1
    assert model.state.vocabulary == ("blue", "green", "red")
    assert dict(model.state.idf) == pytest.approx({"blue": 1, "green": rare, "red": rare})
    item = conversation("test", "red red", "blue unseen")
    # DF counts a whole training conversation once; TF counts every known word.
    assert model.transform_conversation(item, normalization="none") == pytest.approx(
        {"red": 2 * rare / 3, "blue": 1 / 3}
    )
    assert model.transform(item)["0"] == pytest.approx({"red": rare})
    assert model.transform(item)["1"] == {"blue": 1}
    assert model.transform_conversation(item, normalization="l1") == pytest.approx(
        {"red": 2 * rare / (2 * rare + 1), "blue": 1 / (2 * rare + 1)}
    )
    assert model.transform_conversation(item) == pytest.approx(
        {"red": 2 * rare / math.sqrt(4 * rare**2 + 1), "blue": 1 / math.sqrt(4 * rare**2 + 1)}
    )


def test_holdout_vocabulary_and_document_frequency_are_frozen() -> None:
    model = trained()
    before = model.to_dict()
    vectors = model.transform_corpus(
        iter([conversation("z", "unseen"), conversation("a", "red unseen unseen")])
    )
    assert list(vectors) == ["a", "z"]
    assert vectors == {"a": {"red": 1}, "z": {}}
    assert model.to_dict() == before
    with pytest.raises(TypeError):
        vectors["a"]["red"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        vectors["z"] = {}  # type: ignore[index]


def test_training_order_pruning_unicode_and_empty_vocabulary() -> None:
    documents = [conversation("b", "BLUE blue Straße İ can't well-known"), conversation("a", "red")]
    first = TfidfVectorizer(max_features=2).fit(iter(documents))
    second = TfidfVectorizer(max_features=2).fit(reversed(documents))
    assert first.to_dict() == second.to_dict()
    assert first.state.vocabulary == ("blue", "can't")
    unicode_model = TfidfVectorizer().fit(documents)
    assert "strasse" in unicode_model.state.vocabulary
    assert TfidfVectorizer.from_dict(unicode_model.to_dict()).to_dict() == unicode_model.to_dict()
    empty = TfidfVectorizer(min_document_frequency=3).fit(documents)
    assert empty.state.vocabulary == ()
    assert empty.transform_conversation(documents[0]) == {}
    assert TfidfVectorizer.from_dict(empty.to_dict()).state.documents == 2
    assert TfidfVectorizer().fit([conversation("empty", "!!!")]).state.vocabulary == ()


def test_fit_failure_keeps_previous_fitted_state() -> None:
    model = trained()
    before = model.to_dict()
    with pytest.raises(ValueError, match="unique"):
        model.fit([conversation("same", "one"), conversation("same", "two")])
    assert model.to_dict() == before
    with pytest.raises(TypeError, match="Conversation"):
        model.fit(["invalid"])  # type: ignore[list-item]
    assert model.to_dict() == before
    with pytest.raises(TypeError, match="Conversation"):
        model.transform_conversation("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Conversation"):
        model.transform_corpus([None])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="unique"):
        model.transform_corpus([conversation("same", "a"), conversation("same", "b")])
    with pytest.raises(ValueError, match="normalization"):
        model.transform_corpus([], normalization="invalid")
    with pytest.raises(ValueError, match="fitted"):
        TfidfVectorizer().transform_corpus([])


def test_sparse_geometry_and_large_finite_coordinates() -> None:
    assert sparse_cosine({"x": 3, "y": 4}, {"y": 5}) == pytest.approx(0.8)
    assert sparse_cosine({"x": 1}, {"x": -1}) == -1
    assert sparse_cosine({}, {"x": 1}) == 0
    assert sparse_cosine({"x": 0}, {"y": 1}) == 0
    assert sparse_cosine({"x": 1e308, "y": 1e308}, {"x": 1}) == pytest.approx(1 / math.sqrt(2))
    assert normalize_sparse({"x": -3, "y": 4, "z": 0}, normalization="l1") == pytest.approx(
        {"x": -3 / 7, "y": 4 / 7}
    )


@pytest.mark.parametrize(
    "vector", [{"": 1}, {"x": True}, {"x": math.nan}, {"x": math.inf}, {"x": "1"}]
)
def test_sparse_geometry_rejects_invalid_coordinates(vector) -> None:
    with pytest.raises(ValueError):
        normalize_sparse(vector)


def test_sparse_index_scores_ties_cutoffs_and_snapshot() -> None:
    data = {"c": {"red": 3, "blue": 4}, "b": {"blue": 2}, "a": {"blue": 5}, "d": {"red": 8}}
    index = SparseSimilarityIndex(data)
    data["b"]["red"] = 20
    hits = index.query({"blue": 5}, limit=3)
    assert [hit.id for hit in hits] == ["a", "b", "c"]
    assert [hit.score for hit in hits] == pytest.approx([1, 1, 0.8])
    assert [hit.id for hit in index.query({"blue": 1}, minimum_score=0.8, exclude_id="a")] == ["b"]
    assert index.query({"missing": 1}) == ()
    assert index.query({"blue": 1}, minimum_score=1) == ()
    cancelling = SparseSimilarityIndex({"x": {"a": 1, "b": -1}})
    assert cancelling.query({"a": 1, "b": 1}) == ()


@pytest.mark.parametrize(
    "options",
    [
        {"limit": 0},
        {"limit": True},
        {"limit": 1.5},
        {"minimum_score": -1},
        {"minimum_score": math.nan},
        {"minimum_score": True},
    ],
)
def test_index_rejects_invalid_query_options_even_on_empty_index(options) -> None:
    with pytest.raises(ValueError):
        SparseSimilarityIndex({}).query({}, **options)


def test_sparse_type_contracts() -> None:
    with pytest.raises(TypeError, match="mapping"):
        normalize_sparse([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping"):
        SparseSimilarityIndex([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="IDs"):
        SparseSimilarityIndex({"": {}})


def rechecksum(artifact: dict) -> None:
    artifact["sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in artifact.items() if key != "sha256"},
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("version", 2, "version"),
        ("version", True, "version"),
        ("format", "other", "format"),
        ("tokenizer", "other", "tokenizer"),
        ("documents", 0, "positive"),
        ("documents", True, "positive"),
        ("min_df", 0, "positive"),
        ("max_features", 0, "positive"),
        ("max_features", 1, "exceed"),
        ("features", {}, "list"),
        ("features", [{"term": "x", "df": 3}], "between"),
        ("features", [{"term": "x", "df": 0}], "positive"),
        ("features", [{"term": "X", "df": 1}], "terms"),
        ("features", [{"term": " ", "df": 1}], "terms"),
        ("features", [{"term": "x", "df": 1}, {"term": "x", "df": 1}], "terms"),
        ("features", [{"term": "z", "df": 1}, {"term": "a", "df": 1}], "ordered"),
        ("features", [{"term": "x"}], "fields"),
    ],
)
def test_model_schema_validation_with_valid_checksum(key, value, message) -> None:
    artifact = copy.deepcopy(trained().to_dict())
    artifact[key] = value
    rechecksum(artifact)
    with pytest.raises(ValueError, match=message):
        TfidfVectorizer.from_dict(artifact)


def test_model_corruption_and_roundtrip(tmp_path: Path) -> None:
    model = trained()
    path = tmp_path / "model.json"
    model.save(path)
    restored = TfidfVectorizer.load(path)
    query = conversation("holdout", "red green new")
    assert restored.transform_conversation(query) == model.transform_conversation(query)
    assert restored.to_dict() == model.to_dict()
    assert "train-a" not in path.read_text()
    assert list(tmp_path.iterdir()) == [path]
    damaged = model.to_dict()
    damaged["documents"] = 5
    with pytest.raises(ValueError, match="checksum"):
        TfidfVectorizer.from_dict(damaged)
    with pytest.raises(ValueError, match="fields"):
        TfidfVectorizer.from_dict([])
    path.write_text('{"version":1,"version":2}')
    with pytest.raises(ValueError, match="duplicate"):
        TfidfVectorizer.load(path)
    path.write_text('{"unfinished":')
    with pytest.raises(ValueError, match="cannot load"):
        TfidfVectorizer.load(path)
    with pytest.raises(ValueError, match="cannot load"):
        TfidfVectorizer.load(tmp_path / "missing")


def test_atomic_save_cleans_temp_after_replace_failure(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "model.json"
    path.write_text("original")

    def fail_replace(*args):
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        trained().save(path)
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]


def write_dataset(path: Path, *items: Conversation) -> None:
    path.write_text(
        "\n".join(json.dumps(conversation_to_dict(item)) for item in items), encoding="utf-8"
    )


def test_vector_cli_fit_transform_and_query_with_frozen_model(tmp_path: Path, capsys) -> None:
    train, test, model, output = [
        tmp_path / name for name in ("train.jsonl", "test.jsonl", "model.json", "result.json")
    ]
    write_dataset(train, conversation("a", "red blue"), conversation("b", "blue"))
    write_dataset(test, conversation("query", "red unseen"))
    assert main(["vectors", "fit", str(train), str(model)]) == 0
    assert json.loads(capsys.readouterr().out) == {"documents": 2, "features": 2}
    saved = model.read_bytes()
    assert main(["vectors", "transform", str(model), str(test), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == {"query": {"red": 1}}
    assert main(["vectors", "query", str(model), str(train), str(test)]) == 0
    hits = json.loads(capsys.readouterr().out)["query"]
    rare = math.log(3 / 2) + 1
    assert hits == [{"id": "a", "score": pytest.approx(rare / math.sqrt(rare**2 + 1))}]
    assert (
        main(
            [
                "vectors",
                "query",
                str(model),
                str(train),
                str(train),
                "--exclude-self",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["a"][0]["id"] == "b"
    assert model.read_bytes() == saved
    for args in (
        ["fit", str(train), str(train)],
        ["transform", str(model), str(test), "--output", str(model)],
        ["query", str(model), str(train), str(test), "--output", str(test)],
        ["query", str(model), str(train), str(test), "--limit", "0"],
    ):
        assert main(["vectors", *args]) == 2
    assert model.read_bytes() == saved
