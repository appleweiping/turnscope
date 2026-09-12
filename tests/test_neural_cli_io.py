"""Small-cap input admission and publication fault probes; no models or Torch."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from turnscope import neural_cli_io as api


def row(identifier="c", *, text="private text", turns=1):
    return {
        "id": identifier,
        "utterances": [
            {
                "id": str(index),
                "role": "person",
                "text": text,
                "timestamp": f"2026-01-01T00:00:{index:02d}+00:00",
                "metadata": {"event": False},
            }
            for index in range(turns)
        ],
        "metadata": {"forecast_groups": ["private-group"]},
    }


def raw_row(identifier="c", **kwargs):
    return (json.dumps(row(identifier, **kwargs), ensure_ascii=False) + "\n").encode("utf-8")


def test_exact_input_counts_digests_unicode_and_order_without_raw_provenance(tmp_path):
    raw = raw_row("z", text="\U0001f600\u0000é") + raw_row("a", text="second", turns=2)
    path = tmp_path / "input.jsonl"
    path.write_bytes(raw)
    records, source = api.read_neural_jsonl(path)
    assert [record.id for record in records] == ["z", "a"]
    assert records[0].utterances[0].text == "\U0001f600\u0000é"
    assert source == {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "conversations": 2,
        "utterances": 3,
        "json_nodes": sum(
            api._json_structure(line, remaining_nodes=10000)
            for line in raw.splitlines(keepends=True)
        ),
    }
    assert "private" not in json.dumps(source) and str(path) not in json.dumps(source)


def test_empty_file_and_no_final_newline_are_explicit(tmp_path):
    path = tmp_path / "empty"
    path.write_bytes(b"")
    records, source = api.read_neural_jsonl(path)
    assert records == () and source["bytes"] == source["json_nodes"] == 0
    assert source["sha256"] == hashlib.sha256(b"").hexdigest()
    path.write_bytes(raw_row().rstrip(b"\n"))
    assert len(api.read_neural_jsonl(path)[0]) == 1


@pytest.mark.parametrize("mode", [stat.S_IFDIR, stat.S_IFIFO, stat.S_IFSOCK, stat.S_IFLNK])
def test_nonregular_and_link_admission_happens_before_open(tmp_path, monkeypatch, mode):
    path = tmp_path / "input"
    monkeypatch.setattr(Path, "lstat", lambda self: SimpleNamespace(st_mode=mode, st_size=0))

    def forbidden(*args, **kwargs):
        raise AssertionError("nonregular path was opened")

    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="regular file"):
        api.read_neural_jsonl(path)


def test_wholefile_and_line_byte_limits_before_parse_and_exact_boundary(tmp_path, monkeypatch):
    raw = raw_row()
    path = tmp_path / "data"
    path.write_bytes(raw)
    monkeypatch.setattr(api, "MAX_FILE_BYTES", len(raw))
    monkeypatch.setattr(api, "MAX_LINE_BYTES", len(raw))
    assert api.read_neural_jsonl(path)[1]["bytes"] == len(raw)

    def forbidden(*args, **kwargs):
        raise AssertionError("parse before byte admission")

    monkeypatch.setattr(api, "parse_json_value", forbidden)
    monkeypatch.setattr(api, "MAX_LINE_BYTES", len(raw) - 1)
    with pytest.raises(ValueError, match="byte limit"):
        api.read_neural_jsonl(path)
    monkeypatch.setattr(api, "MAX_FILE_BYTES", len(raw) - 1)
    monkeypatch.setattr(Path, "open", forbidden)
    with pytest.raises(ValueError, match="regular file"):
        api.read_neural_jsonl(path)


def test_conversation_and_turn_limits_precede_extra_typed_records(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(raw_row("one") + raw_row("two"))
    original = api.parse_json_value
    parses = 0

    def counted(value):
        nonlocal parses
        parses += 1
        return original(value)

    monkeypatch.setattr(api, "parse_json_value", counted)
    monkeypatch.setattr(api, "MAX_RECORDS", 1)
    with pytest.raises(ValueError, match="conversation limit"):
        api.read_neural_jsonl(path)
    assert parses == 1
    path.write_bytes(raw_row(turns=2))
    monkeypatch.setattr(api, "MAX_SOURCE_TURNS", 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("typed conversation expanded before turn admission")

    monkeypatch.setattr(api, "conversation_from_dict", forbidden)
    with pytest.raises(ValueError, match="invalid neural conversation"):
        api.read_neural_jsonl(path)


def test_json_depth_nodes_and_quoted_brackets_are_checked_before_parser(tmp_path, monkeypatch):
    assert api._json_structure(b'[[{"x":"[\\"{}]","y":false}]]', remaining_nodes=20) == 7
    assert api._json_structure(b"[" * 32 + b"0" + b"]" * 32, remaining_nodes=33) == 33
    path = tmp_path / "data"

    def forbidden(*args, **kwargs):
        raise AssertionError("JSON parser called before structural admission")

    monkeypatch.setattr(api, "parse_json_value", forbidden)
    path.write_bytes(b"[" * 33 + b"0" + b"]" * 33)
    with pytest.raises(ValueError, match="depth limit"):
        api.read_neural_jsonl(path)
    path.write_bytes(raw_row())
    monkeypatch.setattr(api, "MAX_JSON_NODES", 2)
    with pytest.raises(ValueError, match="node limit"):
        api.read_neural_jsonl(path)


def test_aggregate_nodes_not_just_per_line(tmp_path, monkeypatch):
    one, two = raw_row("one"), raw_row("two")
    count = api._json_structure(one, remaining_nodes=10000)
    path = tmp_path / "data"
    path.write_bytes(one + two)
    monkeypatch.setattr(api, "MAX_JSON_NODES", 2 * count)
    assert api.read_neural_jsonl(path)[1]["json_nodes"] == 2 * count
    monkeypatch.setattr(api, "MAX_JSON_NODES", 2 * count - 1)
    with pytest.raises(ValueError, match="node limit"):
        api.read_neural_jsonl(path)


@pytest.mark.parametrize(
    "raw",
    [
        b"\n",
        b"\xff",
        b'{"id":"private-secret","id":"duplicate","utterances":[]}',
        b'{"id":"c","utterances":[],"metadata":{"private-secret":NaN}}',
        b'{"id":"c","utterances":[],"metadata":{"private-secret":1e9999}}',
        b'{"id":"\\ud800","utterances":[]}',
        b'{"id":"private-secret","utterances":false}',
        b'{"id":"private-secret","utterances":[{}]}',
        b'{"private-secret": [}',
        b'{"private-secret":"unterminated}',
    ],
)
def test_invalid_input_errors_are_strict_and_redacted(tmp_path, raw):
    path = tmp_path / "data"
    path.write_bytes(raw)
    with pytest.raises(ValueError) as caught:
        api.read_neural_jsonl(path)
    assert "private-secret" not in str(caught.value)
    assert "\ud800" not in str(caught.value)


def test_duplicate_conversation_ids_rejected(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(raw_row() * 2)
    with pytest.raises(ValueError, match="duplicate conversation"):
        api.read_neural_jsonl(path)


def test_duplicate_utterance_ids_rejected_by_neural_reader(tmp_path):
    value = row(turns=2)
    value["utterances"][1]["id"] = value["utterances"][0]["id"]
    path = tmp_path / "data"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate|invalid neural conversation"):
        api.read_neural_jsonl(path)


def test_changed_open_file_detected_after_read(tmp_path, monkeypatch):
    path = tmp_path / "data"
    raw = raw_row()
    path.write_bytes(raw)
    original = api.parse_json_value

    def mutate(text):
        result = original(text)
        with path.open("ab") as stream:
            stream.write(b" ")
        return result

    monkeypatch.setattr(api, "parse_json_value", mutate)
    with pytest.raises(ValueError):
        api.read_neural_jsonl(path)


def test_changed_path_identity_detected_even_with_stable_open_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.write_bytes(raw_row("old"))
    before = path.lstat()
    calls = 0

    def changed_identity(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            return before
        # POSIX permits replacing an open pathname; Windows may deny it. This
        # portable probe keeps the real fd stable and changes only path identity.
        return SimpleNamespace(
            st_dev=before.st_dev,
            st_ino=before.st_ino + 1,
            st_size=before.st_size,
            st_mtime_ns=before.st_mtime_ns,
            st_mode=before.st_mode,
        )

    monkeypatch.setattr(Path, "lstat", changed_identity)
    with pytest.raises(ValueError, match="changed"):
        api.read_neural_jsonl(path)


@pytest.mark.parametrize("value", [{}, {"config": {}}, {"training_config": {}, "policy": {}}])
def test_settings_closed_outer_fields(tmp_path, value):
    path = tmp_path / "settings"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert api.read_neural_settings(path) == value


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b'{"unknown":{}}',
        b'{"config":null}',
        b'{"config":{},"config":{}}',
        b'{"config":{"private-secret":Infinity}}',
        b'"\xff"',
    ],
)
def test_settings_reject_duplicates_nonfinite_unknown_or_nondict(tmp_path, raw):
    path = tmp_path / "settings"
    path.write_bytes(raw)
    with pytest.raises(ValueError) as caught:
        api.read_neural_settings(path)
    assert "private-secret" not in str(caught.value)


def test_input_output_aliases_hardlinks_existing_and_missing_parent(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"private")
    with pytest.raises(ValueError, match="differ"):
        api.check_new_output(source, (source,))
    link = tmp_path / "link"
    os.link(source, link)
    with pytest.raises(ValueError, match="differ"):
        api.check_new_output(link, (source,))
    with pytest.raises(ValueError, match="new file"):
        api.check_new_output(link, ())
    with pytest.raises(ValueError, match="parent"):
        api.check_new_output(tmp_path / "missing" / "output", ())
    api.check_new_output(tmp_path / "new", (source,))
    api.check_new_output(None, (source,))
    assert source.read_bytes() == b"private"


def test_report_is_canonical_complete_and_exclusive(tmp_path):
    output = tmp_path / "report.json"
    payload = {"z": "é\u0000", "a": [1, True]}
    assert api.publish_neural_report(payload, output) is None
    expected = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert output.read_bytes() == expected
    assert list(tmp_path.glob(".turnscope-neural-report-*")) == []
    with pytest.raises(ValueError, match="new file"):
        api.publish_neural_report({"other": True}, output)
    assert output.read_bytes() == expected


def test_output_byte_cap_includes_utf8_and_final_newline_before_tempfile(tmp_path, monkeypatch):
    expected = b'{"x":"\xc3\xa9"}\n'
    monkeypatch.setattr(api, "MAX_OUTPUT_BYTES", len(expected))
    output = tmp_path / "ok"
    api.publish_neural_report({"x": "é"}, output)
    assert output.read_bytes() == expected
    monkeypatch.setattr(api, "MAX_OUTPUT_BYTES", len(expected) - 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("temporary file before output admission")

    monkeypatch.setattr(api.tempfile, "NamedTemporaryFile", forbidden)
    with pytest.raises(ValueError, match="byte limit"):
        api.publish_neural_report({"x": "é"}, tmp_path / "too-large")
    assert not (tmp_path / "too-large").exists()


def test_short_binary_write_cannot_publish_partial_report(tmp_path, monkeypatch):
    real_factory = api.tempfile.NamedTemporaryFile

    class ShortWriter:
        def __init__(self, *args, **kwargs):
            self.stream = real_factory(*args, **kwargs)
            self.name = self.stream.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, raw):
            return self.stream.write(raw[:2])

        def flush(self):
            return self.stream.flush()

        def fileno(self):
            return self.stream.fileno()

    monkeypatch.setattr(api.tempfile, "NamedTemporaryFile", ShortWriter)
    output = tmp_path / "report"
    with pytest.raises(OSError, match=r"incomplete|short"):
        api.publish_neural_report({"complete": True}, output)
    assert not output.exists()


def test_link_race_does_not_overwrite_external_winner(tmp_path, monkeypatch):
    output = tmp_path / "report"
    original_link = api.os.link

    def raced_link(source, destination):
        Path(destination).write_bytes(b"other writer")
        return original_link(source, destination)

    monkeypatch.setattr(api.os, "link", raced_link)
    with pytest.raises(FileExistsError):
        api.publish_neural_report({"candidate": True}, output)
    assert output.read_bytes() == b"other writer"
    assert list(tmp_path.glob(".turnscope-neural-report-*")) == []


def test_postpublication_cleanup_failure_returns_redacted_success_warning(tmp_path, monkeypatch):
    original = Path.unlink

    def failure(path, *args, **kwargs):
        if path.name.startswith(".turnscope-neural-report-"):
            raise OSError("private-secret-local-path")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failure)
    output = tmp_path / "report"
    warning = api.publish_neural_report({"complete": True}, output)
    assert json.loads(output.read_bytes()) == {"complete": True}
    assert warning == "report published; temporary-file cleanup failed"
    assert "private-secret" not in warning
    assert len(list(tmp_path.glob(".turnscope-neural-report-*"))) == 1


def test_prepublication_failure_cleanup_does_not_claim_success(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("publication blocked")

    monkeypatch.setattr(api.os, "link", denied)
    monkeypatch.setattr(Path, "unlink", denied)
    output = tmp_path / "report"
    with pytest.raises(PermissionError, match="publication blocked"):
        api.publish_neural_report({"candidate": True}, output)
    assert not output.exists()


@pytest.mark.parametrize("count", [0, None, True])
def test_stdout_partial_writes_are_errors(monkeypatch, count):
    class Partial(io.StringIO):
        def write(self, text):
            super().write(text[:1])
            return count

    stream = Partial()
    monkeypatch.setattr(api.sys, "stdout", stream)
    with pytest.raises(api.NeuralOutputError, match="stdout report failed"):
        api.publish_neural_report({"complete": True}, None)


def test_stdout_complete_unicode_and_closed_stream(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(api.sys, "stdout", stream)
    assert api.publish_neural_report({"text": "é"}, None) is None
    assert stream.getvalue() == '{"text":"é"}\n'
    stream.close()
    with pytest.raises(api.NeuralOutputError):
        api.publish_neural_report({"complete": True}, None)


def test_real_closed_stdout_pipe_preserves_declared_failure_exit_code():
    script = """
import sys
from turnscope.neural_cli_io import NeuralOutputError, publish_neural_report
try:
    publish_neural_report({"complete": True}, None)
except NeuralOutputError:
    sys.exit(1)
raise AssertionError("closed pipe was not detected")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    with subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    ) as process:
        assert process.stdout is not None and process.stderr is not None
        process.stdout.close()
        process.stdout = None
        try:
            _, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
        code = process.returncode
    assert code == 1
    assert b"Exception ignored" not in stderr


def test_settings_file_and_node_admission_before_parse(tmp_path, monkeypatch):
    path = tmp_path / "settings"
    path.write_bytes(b" " * (64 * 1024 + 1))

    def forbidden(*args, **kwargs):
        raise AssertionError("settings parsed before admission")

    monkeypatch.setattr(api, "parse_json_value", forbidden)
    with pytest.raises(ValueError, match="regular file"):
        api.read_neural_settings(path)
    # Small JSON text with more than the fixed4096 settings graph-node budget.
    path.write_bytes(b'{"config":{"x":[' + b"0," * 4096 + b"0]}}")
    with pytest.raises(ValueError, match="node limit"):
        api.read_neural_settings(path)


def test_fsync_failure_removes_unpublished_temporary(tmp_path, monkeypatch):
    def denied(*args):
        raise OSError("synchronization denied")

    monkeypatch.setattr(api.os, "fsync", denied)
    output = tmp_path / "report"
    with pytest.raises(OSError, match="synchronization denied"):
        api.publish_neural_report({"candidate": True}, output)
    assert not output.exists()
    assert list(tmp_path.glob(".turnscope-neural-report-*")) == []


@pytest.mark.parametrize("descriptor", [None, True, -1])
def test_stdout_silencing_does_not_touch_os_for_invalid_embedded_descriptor(
    monkeypatch, descriptor
):
    def forbidden(*args, **kwargs):
        raise AssertionError("OS descriptor touched for an embedded stream")

    with monkeypatch.context() as scoped:
        scoped.setattr(api.sys, "stdout", SimpleNamespace(fileno=lambda: descriptor))
        scoped.setattr(api.os, "open", forbidden)
        api._silence_failed_stdout()


def test_stdout_silencing_closes_null_fd_even_when_duplication_fails(monkeypatch):
    closed = []

    def denied(*args):
        raise OSError("duplication denied")

    # Restore process-wide os helpers before pytest resumes its own FD capture.
    with monkeypatch.context() as scoped:
        scoped.setattr(api.sys, "stdout", SimpleNamespace(fileno=lambda: 42))
        scoped.setattr(api.os, "open", lambda *args: 99)
        scoped.setattr(api.os, "dup2", denied)
        scoped.setattr(api.os, "close", closed.append)
        api._silence_failed_stdout()
    assert closed == [99]
