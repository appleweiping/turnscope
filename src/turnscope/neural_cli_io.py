"""Bounded private JSONL inputs and exclusive reports for neural commands."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .io import conversation_from_dict, parse_json_value
from .models import Conversation

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024 * 1024
MAX_JSON_NODES = 2_000_000
MAX_RECORDS = 10_000
MAX_SOURCE_TURNS = 100_000
MAX_OUTPUT_BYTES = 32 * 1024 * 1024


class NeuralOutputError(OSError):
    """The host output stream failed; an earlier model publication may have succeeded."""


def _silence_failed_stdout() -> None:
    # CPython flushes again at shutdown. A failed real pipe must not turn the
    # explicit failure status into 120; embedded streams without an FD are left alone.
    try:
        descriptor = sys.stdout.fileno()
        if type(descriptor) is not int or descriptor < 0:
            return
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, descriptor)
        finally:
            os.close(null)
    except (OSError, ValueError, AttributeError):
        return


def _json_structure(raw: bytes, *, remaining_nodes: int) -> int:
    """Count JSON value/key starts and nesting before JSON container allocation."""
    stack: list[int] = []
    quoted = escaped = primitive = False
    nodes = 0
    for character in raw:
        if quoted:
            if escaped:
                escaped = False
            elif character == 92:
                escaped = True
            elif character == 34:
                quoted = False
            continue
        if character in (9, 10, 13, 32, 44, 58):
            primitive = False
            continue
        if character in (93, 125):
            if not stack or (stack.pop(), character) not in ((91, 93), (123, 125)):
                raise ValueError("malformed neural JSON structure")
            primitive = False
            continue
        if character == 34:
            quoted = True
            primitive = False
            nodes += 1
        elif character in (91, 123):
            stack.append(character)
            if len(stack) > 32:
                raise ValueError("neural JSON exceeds depth limit")
            primitive = False
            nodes += 1
        elif not primitive:
            primitive = True
            nodes += 1
        if nodes > remaining_nodes:
            raise ValueError("neural JSON exceeds node limit")
    if quoted or stack:
        raise ValueError("incomplete neural JSON structure")
    return nodes


def _regular(path: Path, maximum: int) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
        raise ValueError("neural input must be a bounded regular file, not a link or pipe")
    return metadata


def _same_file(before: os.stat_result, after: os.stat_result) -> None:
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("neural input changed during reading")


def read_neural_jsonl(path: Path) -> tuple[tuple[Conversation, ...], dict[str, Any]]:
    """Strict JSONL conversations; no text/metadata is echoed in parse errors."""
    before = _regular(path, MAX_FILE_BYTES)
    records: list[Conversation] = []
    seen = set()
    byte_count = node_count = turn_count = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        _same_file(before, os.fstat(stream.fileno()))
        while True:
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            byte_count += len(raw)
            if len(raw) > MAX_LINE_BYTES or byte_count > MAX_FILE_BYTES:
                raise ValueError("neural JSONL exceeds a byte limit")
            if len(records) == MAX_RECORDS:
                raise ValueError("neural JSONL exceeds its conversation limit")
            node_count += _json_structure(raw, remaining_nodes=MAX_JSON_NODES - node_count)
            try:
                value = parse_json_value(raw.decode("utf-8"))
                if type(value) is not dict or type(value.get("utterances")) is not list:
                    raise ValueError("expected a conversation object")
                turn_count += len(value["utterances"])
                if turn_count > MAX_SOURCE_TURNS:
                    raise ValueError("source turn limit exceeded")
                record = conversation_from_dict(value)
            except (ValueError, UnicodeError, RecursionError, OverflowError):
                raise ValueError(
                    f"invalid neural conversation at JSONL line {len(records) + 1}"
                ) from None
            if record.id in seen:
                raise ValueError("neural input has duplicate conversation IDs")
            if len({turn.id for turn in record.utterances}) != len(record.utterances):
                raise ValueError("neural conversation has duplicate utterance IDs")
            seen.add(record.id)
            digest.update(raw)
            records.append(record)
        _same_file(before, os.fstat(stream.fileno()))
    _same_file(before, path.lstat())
    if byte_count != before.st_size:
        raise ValueError("neural input byte accounting changed during reading")
    return tuple(records), {
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "conversations": len(records),
        "utterances": turn_count,
        "json_nodes": node_count,
    }


def read_neural_settings(path: Path) -> dict[str, Any]:
    before = _regular(path, 64 * 1024)
    with path.open("rb") as stream:
        _same_file(before, os.fstat(stream.fileno()))
        raw = stream.read(64 * 1024 + 1)
        _same_file(before, os.fstat(stream.fileno()))
    _same_file(before, path.lstat())
    if len(raw) != before.st_size:
        raise ValueError("neural settings changed during reading")
    _json_structure(raw, remaining_nodes=4096)
    try:
        value = parse_json_value(raw.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("invalid neural settings JSON") from None
    allowed = {"config", "training_config", "policy", "data_limits", "numeric_limits"}
    if (
        type(value) is not dict
        or not value.keys() <= allowed
        or any(type(item) is not dict for item in value.values())
    ):
        raise ValueError("neural settings require only known configuration objects")
    return value


def check_new_output(output: Path | None, protected: Iterable[Path]) -> None:
    if output is None:
        return
    for source in protected:
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError("neural output must differ from every input")
    if output.exists() or output.is_symlink():
        raise ValueError("neural output must be a new file")
    if not output.parent.is_dir():
        raise ValueError("neural output parent must already exist")


def publish_neural_report(payload: dict[str, Any], output: Path | None) -> str | None:
    """Publish one complete bounded report; return a post-publication cleanup warning."""
    parts = bytearray()
    encoder = json.JSONEncoder(
        sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    for fragment in encoder.iterencode(payload):
        for offset in range(0, len(fragment), 4096):
            chunk = fragment[offset : offset + 4096].encode("utf-8")
            if len(parts) + len(chunk) + 1 > MAX_OUTPUT_BYTES:
                raise ValueError("neural report exceeds the output byte limit")
            parts.extend(chunk)
    parts.extend(b"\n")
    if output is None:
        rendered = parts.decode("utf-8")
        try:
            count = sys.stdout.write(rendered)
            if type(count) is not int or count != len(rendered):
                raise OSError("incomplete neural stdout report")
            sys.stdout.flush()
        except (OSError, ValueError, UnicodeError):
            _silence_failed_stdout()
            raise NeuralOutputError("neural stdout report failed after command work") from None
        return None
    check_new_output(output, ())
    temporary: Path | None = None
    published = False
    warning = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".turnscope-neural-report-", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            written = stream.write(parts)
            if type(written) is not int or written != len(parts):
                raise OSError("incomplete neural report file write")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
        published = True
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                if published:
                    warning = "report published; temporary-file cleanup failed"
                else:
                    # Do not hide the original failure or claim publication succeeded.
                    pass
    return warning
