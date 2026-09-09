"""Independent byte, schema and publication boundaries for model command inputs."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from turnscope import dual_context_cli as cli
from turnscope.cli import main


@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b""])
def test_record_reader_preserves_exact_unicode_text(tmp_path, newline):
    value = {"conversation_id": "c", "utterance_id": "u", "text": "café😀\r\n{{participant}}"}
    path = tmp_path / "input.jsonl"
    path.write_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8") + newline)
    assert list(cli._read_rows(path, cli.RECORD_FIELDS, 1)) == [value]


@pytest.mark.parametrize(
    "raw",
    [
        b"\n",
        b"\xff",
        b"[]",
        b'{"text":"PRIVATE", "text":"PRIVATE"}',
        b'{"conversation_id":"c","utterance_id":"u","text":null}',
        b'{"conversation_id":"c","utterance_id":"u","text":true}',
        b'{"conversation_id":"c","utterance_id":"u","text":"\\ud800"}',
        b'{"conversation_id":"c","utterance_id":"u","text":"PRIVATE","extra":1}',
        b"[" * 2000 + b"]" * 2000,
    ],
)
def test_invalid_input_is_controlled_without_echoing_private_values(tmp_path, raw):
    path = tmp_path / "input.jsonl"
    path.write_bytes(raw)
    with pytest.raises(ValueError) as failed:
        list(cli._read_rows(path, cli.RECORD_FIELDS, 10))
    assert "PRIVATE" not in str(failed.value)


def test_record_and_byte_limits_accept_exact_boundary(tmp_path, monkeypatch):
    raw = b'{"conversation_id":"c","source_id":"s","context_id":"t"}\n'
    path = tmp_path / "edges.jsonl"
    path.write_bytes(raw)
    monkeypatch.setattr(cli, "MAX_INPUT_BYTES", len(raw))
    monkeypatch.setattr(cli, "MAX_LINE_BYTES", len(raw))
    assert len(list(cli._read_rows(path, cli.EDGE_FIELDS, 1))) == 1
    with pytest.raises(ValueError, match="record limit"):
        list(cli._read_rows(path, cli.EDGE_FIELDS, 0))
    monkeypatch.setattr(cli, "MAX_LINE_BYTES", len(raw) - 1)
    with pytest.raises(ValueError, match="byte limit"):
        list(cli._read_rows(path, cli.EDGE_FIELDS, 1))
    monkeypatch.setattr(cli, "MAX_INPUT_BYTES", len(raw) - 1)
    with pytest.raises(ValueError, match="byte limit"):
        list(cli._read_rows(path, cli.EDGE_FIELDS, 1))


def test_complete_encoded_rows_have_a_byte_boundary_and_do_not_mutate_header(monkeypatch):
    header = {"private_data": True, "model_digest": "a" * 64}
    rows = [{"text": "café😀"}, {"vector": [0.0, 1.0]}]
    result = cli._encode_rows(header, rows)
    assert json.loads(result) == {**header, "rows": rows}
    assert "rows" not in header
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", len(result))
    assert cli._encode_rows(header, rows) == result
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", len(result) - 1)
    with pytest.raises(ValueError, match="byte limit"):
        cli._encode_rows(header, rows)
    monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        cli._encode_rows(header, ())


def test_empty_encoded_rows_are_valid_and_reserved_rows_rejected():
    assert json.loads(cli._encode_rows({}, ())) == {"rows": []}
    with pytest.raises(ValueError, match="reserved"):
        cli._encode_rows({"rows": []}, ())


def test_new_file_publication_and_existing_alias_protection(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"original")
    alias = tmp_path / "alias"
    alias.hardlink_to(source)
    with pytest.raises(ValueError, match="differ"):
        cli._check_output(alias, [source])
    with pytest.raises(ValueError, match="differ"):
        cli._check_output(source, [source])
    with pytest.raises(ValueError, match="new file"):
        cli._publish(b"changed", source)
    assert source.read_bytes() == alias.read_bytes() == b"original"
    output = tmp_path / "new"
    cli._check_output(output, [source])
    cli._publish(b'{"complete":true}\n', output)
    assert json.loads(output.read_bytes()) == {"complete": True}
    assert not list(tmp_path.glob(".turnscope-dual-*"))


def test_publication_race_never_replaces_destination_and_cleans_temp(tmp_path, monkeypatch):
    output = tmp_path / "out"

    def race(_source, target):
        target.write_bytes(b"other writer")
        raise FileExistsError("simulated destination race")

    monkeypatch.setattr(cli.os, "link", race)
    with pytest.raises(FileExistsError):
        cli._publish(b"ours", output)
    assert output.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".turnscope-dual-*"))


def test_stdout_uses_utf8_binary_when_available_and_text_otherwise(monkeypatch):
    encoded = cli._encode_rows({}, [{"text": "café😀"}])
    text = io.StringIO()
    monkeypatch.setattr(sys, "stdout", text)
    cli._publish(encoded, None)
    assert text.getvalue().encode("utf-8") == encoded
    binary = io.BytesIO()
    wrapper = io.TextIOWrapper(binary, encoding="ascii")
    monkeypatch.setattr(sys, "stdout", wrapper)
    cli._publish(encoded, None)
    assert binary.getvalue() == encoded


def test_actual_public_cli_demo_runs_independent_model_load_processes(tmp_path):
    pytest.importorskip("numpy")
    script = Path(__file__).resolve().parents[1] / "examples/dual_context_demo.py"
    directory = tmp_path / "original-demo"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(directory)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["model_unchanged"] is True
    assert summary["summary"]["all_gold_queries"]["mean_reciprocal_rank"] == 0.5
    report = json.loads((directory / "predictions.json").read_bytes())
    assert report["private_data"] is True
    assert len(report["rows"]) == 2
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    assert (
        main(
            [
                "dual-context",
                "terms",
                str(directory / "model.json"),
                "--output",
                str(directory / "predictions.json"),
            ]
        )
        == 2
    )
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize("action", ["fit", "transform", "terms", "evaluate"])
def test_new_commands_are_reachable_in_public_help(action, capsys):
    with pytest.raises(SystemExit) as finished:
        main(["dual-context", action, "--help"])
    assert finished.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_malformed_catalog_cli_does_not_publish_model_or_echo_text(tmp_path, capsys):
    path = tmp_path / "input.jsonl"
    path.write_text('{"PRIVATE_TOKEN":1,"PRIVATE_TOKEN":2}', encoding="utf-8")
    target = tmp_path / "model.json"
    assert main(["dual-context", "fit", str(path), str(path), str(path), str(target)]) == 2
    assert not target.exists()
    diagnostics = capsys.readouterr()
    assert "command failed" in diagnostics.err and "PRIVATE_TOKEN" not in diagnostics.err
    assert not diagnostics.out
