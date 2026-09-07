"""Transactional, disk-backed storage for validated conversation corpora.

Conversation order and duplicate utterance IDs are preserved: the auditor, not
the storage layer, decides whether a conversation is suitable for analysis.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType

from .io import conversation_from_dict, conversation_to_dict, parse_json_value
from .models import Conversation


class CorpusStore:
    """A single-threaded SQLite corpus with atomic batch writes.

    Use as a context manager. Each write is committed before returning; a failed
    batch rolls back completely, including replacements. Readers in other store
    instances only see committed changes. IDs are case-sensitive and pagination
    uses SQLite's binary string ordering. No pickle or executable serialization
    is used. Do not open unrelated SQLite databases with this class.
    """

    def __init__(self, path: str | Path) -> None:
        self._connection = sqlite3.connect(str(path))
        self._closed = False
        try:
            with self._connection:
                version = self._connection.execute("PRAGMA user_version").fetchone()[0]
                tables = self._connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
                if version == 0 and not tables:
                    self._connection.execute(
                        "CREATE TABLE conversations ("
                        "id TEXT PRIMARY KEY NOT NULL, body TEXT NOT NULL, "
                        "utterances INTEGER NOT NULL CHECK (utterances >= 0))"
                    )
                    self._connection.execute("PRAGMA user_version = 1")
                    self._connection.execute("PRAGMA application_id = 1414742864")
                application = self._connection.execute("PRAGMA application_id").fetchone()[0]
                version = self._connection.execute("PRAGMA user_version").fetchone()[0]
                if application != 1414742864 or version != 1:
                    raise ValueError("not a supported TurnScope corpus database")
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> CorpusStore:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("corpus store is closed")

    def close(self) -> None:
        """Close the connection; repeated calls are harmless."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def put(self, conversations: Iterable[Conversation], *, replace: bool = False) -> int:
        """Store a batch atomically, rejecting existing IDs unless replace is set.

        An ID repeated inside one batch is always rejected. Serialization and
        input-iterator failures also roll back the entire batch. Memory overhead
        is one serialized conversation plus the set of batch IDs.
        """
        self._ensure_open()
        seen: set[str] = set()
        sql = "INSERT INTO conversations (id, body, utterances) VALUES (?, ?, ?)"
        if replace:
            sql += (
                " ON CONFLICT(id) DO UPDATE SET body = excluded.body, "
                "utterances = excluded.utterances"
            )
        with self._connection:
            for conversation in conversations:
                if conversation.id in seen:
                    raise ValueError(f"duplicate conversation in batch: {conversation.id!r}")
                seen.add(conversation.id)
                # Revalidate metadata as well as records before persisting.
                body = json.dumps(
                    conversation_to_dict(conversation), ensure_ascii=True, allow_nan=False
                )
                conversation_from_dict(parse_json_value(body))
                try:
                    self._connection.execute(
                        sql, (conversation.id, body, len(conversation.utterances))
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"conversation already exists: {conversation.id!r}") from error
        return len(seen)

    def get(self, conversation_id: str) -> Conversation:
        """Load an independent record or raise KeyError for an unknown ID."""
        self._ensure_open()
        row = self._connection.execute(
            "SELECT body FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(conversation_id)
        return conversation_from_dict(parse_json_value(row[0]))

    def ids(self, *, after: str | None = None, limit: int = 100) -> tuple[str, ...]:
        """Return a bounded keyset page; pass the last ID as the next ``after``.

        Pages are not a cross-call snapshot when other connections write. Each
        call sees committed state; deleted IDs never shift an offset cursor.
        """
        self._ensure_open()
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise ValueError("limit must be an integer between 1 and 10000")
        if after is None:
            rows = self._connection.execute(
                "SELECT id FROM conversations ORDER BY id LIMIT ?", (limit,)
            )
        else:
            rows = self._connection.execute(
                "SELECT id FROM conversations WHERE id > ? ORDER BY id LIMIT ?", (after, limit)
            )
        return tuple(row[0] for row in rows)

    def iter_conversations(
        self, *, after: str | None = None, limit: int | None = None
    ) -> Iterator[Conversation]:
        """Yield detached conversations in ID order using an optional keyset page."""

        self._ensure_open()
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise ValueError("limit must be a positive integer or None")
        query = "SELECT body FROM conversations"
        parameters: tuple[object, ...] = ()
        if after is not None:
            query += " WHERE id > ?"
            parameters = (after,)
        query += " ORDER BY id"
        if limit is not None:
            query += " LIMIT ?"
            parameters += (limit,)
        rows = self._connection.execute(query, parameters)
        for row in rows:
            yield conversation_from_dict(parse_json_value(row[0]))

    def delete(self, conversation_ids: Iterable[str]) -> int:
        """Atomically delete IDs, ignoring missing IDs; return records removed."""
        self._ensure_open()
        count = 0
        with self._connection:
            for conversation_id in conversation_ids:
                count += self._connection.execute(
                    "DELETE FROM conversations WHERE id = ?", (conversation_id,)
                ).rowcount
        return count

    def counts(self) -> tuple[int, int]:
        """Return conversation and utterance totals in one database snapshot."""
        self._ensure_open()
        row = self._connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(utterances), 0) FROM conversations"
        ).fetchone()
        return int(row[0]), int(row[1])
