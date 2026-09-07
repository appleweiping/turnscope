import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from turnscope import Conversation, CorpusStore


def test_persistence(tmp_path: Path, conversation: Conversation) -> None:
    path = tmp_path / "corpus.db"
    with CorpusStore(path) as store:
        assert store.counts() == (0, 0)
        assert store.put([conversation]) == 1
        assert store.counts() == (1, 4)
    with CorpusStore(path) as store:
        assert store.get("demo") == conversation
        assert store.get("demo") is not conversation
        assert store.ids() == ("demo",)


def test_iter_conversations_supports_keyset_pages_and_limits(conversation: Conversation) -> None:
    with CorpusStore(":memory:") as store:
        store.put([Conversation("a", []), conversation, Conversation("z", [])])
        assert tuple(item.id for item in store.iter_conversations(limit=2)) == ("a", "demo")
        assert tuple(item.id for item in store.iter_conversations(after="demo")) == ("z",)
        with pytest.raises(ValueError, match="limit"):
            tuple(store.iter_conversations(limit=0))


def test_failed_batch_restores_replacement(conversation: Conversation) -> None:
    with CorpusStore(":memory:") as store:
        store.put([conversation])
        with pytest.raises(ValueError, match="duplicate"):
            store.put([Conversation("demo", []), Conversation("demo", [])], replace=True)
        assert store.get("demo") == conversation
        with pytest.raises(ValueError, match="already exists"):
            store.put([Conversation("new", []), conversation])
        assert store.ids() == ("demo",)
        assert store.put([Conversation("demo", [])], replace=True) == 1
        assert store.counts() == (1, 0)


def test_iterator_errors_are_atomic() -> None:
    def broken() -> Iterator[Conversation]:
        yield Conversation("first", [])
        raise RuntimeError("input failed")

    def broken_ids() -> Iterator[str]:
        yield "first"
        raise RuntimeError("input failed")

    with CorpusStore(":memory:") as store:
        with pytest.raises(RuntimeError, match="input failed"):
            store.put(broken())
        assert store.counts() == (0, 0)
        store.put([Conversation("first", [])])
        with pytest.raises(RuntimeError, match="input failed"):
            store.delete(broken_ids())
        assert store.ids() == ("first",)


def test_keyset_and_safe_ids() -> None:
    names = ["z", "a", "é", "' OR 1=1 --"]
    with CorpusStore(":memory:") as store:
        store.put(Conversation(name, []) for name in names)
        first = store.ids(limit=2)
        assert first + store.ids(after=first[-1], limit=2) == tuple(sorted(names))
        assert store.ids(after="é") == ()
        assert store.get(names[-1]).id == names[-1]
        assert store.delete([names[-1], names[-1], "absent"]) == 1
        assert store.counts() == (3, 0)
        with pytest.raises(KeyError):
            store.get(names[-1])


@pytest.mark.parametrize("limit", [0, -1, 10001, True, 1.5, "1"])
def test_invalid_limit(limit: int) -> None:
    with CorpusStore(":memory:") as store, pytest.raises(ValueError, match="limit"):
        store.ids(limit=limit)


def test_close() -> None:
    store = CorpusStore(":memory:")
    store.close()
    store.close()
    with pytest.raises(ValueError, match="closed"):
        store.__enter__()
    with pytest.raises(ValueError, match="closed"):
        store.counts()


def test_unrelated_database_not_modified(tmp_path: Path) -> None:
    path = tmp_path / "other.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE important (value TEXT)")
        connection.execute("INSERT INTO important VALUES ('keep')")
    with pytest.raises(ValueError, match="not a supported"):
        CorpusStore(path)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("SELECT * FROM important").fetchall() == [("keep",)]
        assert connection.execute("PRAGMA user_version").fetchone() == (0,)


def test_committed_visibility(tmp_path: Path, conversation: Conversation) -> None:
    path = tmp_path / "shared.db"
    with CorpusStore(path) as writer, CorpusStore(path) as reader:
        writer.put([conversation])
        assert reader.get("demo") == conversation
        writer.delete(["demo"])
        assert reader.counts() == (0, 0)


def test_invalid_metadata_rolls_back() -> None:
    with CorpusStore(":memory:") as store:
        with pytest.raises(ValueError):
            store.put([Conversation("ok", []), Conversation("bad", [], {"x": float("nan")})])
        assert store.counts() == (0, 0)
