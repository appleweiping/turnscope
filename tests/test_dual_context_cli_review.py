"""Independent regressions for private errors and actual stream publication."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

from turnscope import dual_context_cli as cli


def test_invalid_private_key_is_not_visible_in_the_formatted_exception(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_bytes(b'{"PRIVATE_PATIENT_KEY":1,"PRIVATE_PATIENT_KEY":2}\n')
    with pytest.raises(ValueError) as failed:
        list(cli._read_rows(path, cli.RECORD_FIELDS, 1))
    assert "PRIVATE_PATIENT_KEY" not in str(failed.value)
    assert "PRIVATE_PATIENT_KEY" not in "".join(traceback.format_exception(failed.value))


def test_real_nonblocking_stdout_cannot_report_a_short_write_as_complete(monkeypatch):
    read_fd, write_fd = os.pipe()
    writer = None
    try:
        try:
            os.set_blocking(write_fd, False)
        except (AttributeError, OSError, NotImplementedError):
            pytest.skip("this runtime cannot make anonymous pipe writes nonblocking")
        writer = os.fdopen(write_fd, "wb", buffering=0)
        monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=writer))
        # Larger than the ordinary anonymous-pipe buffer, but below the CLI cap.
        payload = b'{"text":"' + b"x" * 1_000_000 + b'"}\n'
        reported_success = False
        try:
            cli._publish(payload, None)
            reported_success = True
        except (OSError, ValueError):
            pass  # A controlled publication failure must not be claimed complete.
        writer.close()
        observed = os.read(read_fd, len(payload))
        assert not reported_success or observed == payload
    finally:
        if writer is not None:
            writer.close()
        else:
            os.close(write_fd)
        os.close(read_fd)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("returned", [None, 0, 1])
def test_short_stdout_writes_are_rejected_on_every_platform(monkeypatch, binary, returned):
    calls = []

    def write(payload):
        calls.append(payload)
        return returned

    stream = SimpleNamespace(write=write, flush=lambda: None)
    output = SimpleNamespace(buffer=stream) if binary else stream
    monkeypatch.setattr(sys, "stdout", output)
    with pytest.raises(OSError, match="incomplete"):
        cli._publish(b'{"value":true}\n', None)
    assert len(calls) == 1  # Do not resend a prefix after an incomplete write.
    assert isinstance(calls[0], bytes if binary else str)


def test_growing_jsonl_is_checked_against_the_cumulative_byte_limit(tmp_path, monkeypatch):
    path = tmp_path / "input.jsonl"
    row = b'{"conversation_id":"c","utterance_id":"u","text":"authored"}\n'
    path.write_bytes(row)
    monkeypatch.setattr(cli, "MAX_INPUT_BYTES", len(row))
    rows = cli._read_rows(path, cli.RECORD_FIELDS, 2)
    assert next(rows)["text"] == "authored"
    with path.open("ab") as stream:
        stream.write(row)
    with pytest.raises(ValueError, match="byte limit"):
        next(rows)


def test_linked_report_is_successful_when_private_temporary_cleanup_fails(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "report.json"
    original_unlink = Path.unlink

    def unavailable_unlink(path, *args, **kwargs):
        if path.name.startswith(".turnscope-dual-"):
            raise PermissionError("PRIVATE_DIRECTORY_NAME")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unavailable_unlink)
    payload = b'{"private_data":true,"rows":[]}\n'
    try:
        cli._publish(payload, output)
        assert output.read_bytes() == payload
        assert len(list(tmp_path.glob(".turnscope-dual-*"))) == 1
        diagnostics = capsys.readouterr().err
        assert "published" in diagnostics
        assert "cleanup failed" in diagnostics
        assert "PRIVATE_DIRECTORY_NAME" not in diagnostics
    finally:
        for temporary in tmp_path.glob(".turnscope-dual-*"):
            original_unlink(temporary)


@pytest.mark.parametrize("action,expected_status", [("fit", 0), ("terms", 2)])
def test_closed_consumer_keeps_the_honest_exit_status_at_interpreter_shutdown(
    tmp_path, action, expected_status
):
    """Isolate summary publication, but use an actual process and closed OS pipe."""
    script = """
import argparse
import sys
from pathlib import Path
from turnscope import dual_context_cli as cli

class AuthoredModel:
    def __init__(self, **config):
        self.config = cli.DualContextConfig(**config)
    def fit(self, *inputs):
        return self
    @property
    def digest(self):
        return 'a' * 64
    def training_summary(self):
        return {'authored_publication_fixture': True}
    def save(self, path):
        path.write_bytes(b'authored durable artifact')
    @classmethod
    def load(cls, path):
        return cls()
    def term_statistics(self):
        return ({'term': 'authored'},)

cli.DualContextModel = AuthoredModel
parser = argparse.ArgumentParser()
cli.configure_dual_context_parser(parser)
output = Path(sys.argv[1])
arguments = (['fit', 'catalog', 'forward', 'backward', str(output)]
             if sys.argv[2] == 'fit' else ['terms', str(output)])
args = parser.parse_args(arguments)
raise SystemExit(cli.run_dual_context_command(args))
"""
    output = tmp_path / "model.json"
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(output), action],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    process.stdout.close()
    try:
        status = process.wait(timeout=15)
        diagnostics = process.stderr.read().decode("utf-8", errors="replace")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()
    if action == "fit":
        assert output.read_bytes() == b"authored durable artifact"
        assert "model saved; stdout summary could not be delivered" in diagnostics
    else:
        assert not output.exists()
        assert "command failed" in diagnostics
    assert status == expected_status
    assert "Exception ignored" not in diagnostics
