import json
from datetime import datetime, timezone

import pytest

from turnscope.audit import default_auditor
from turnscope.builder import ContextBuilder
from turnscope.cli import main
from turnscope.models import AuditReport, Conversation, Issue, Severity, Utterance
from turnscope.policies import TurnWindowPolicy
from turnscope.reporting import report_json, report_markdown, windows_json


def test_reporting_formats(conversation: Conversation) -> None:
    report = default_auditor().audit([conversation])
    assert json.loads(report_json(report))["summary"]["conversations"] == 1
    assert "No reliability issues found" in report_markdown(report)
    windows = ContextBuilder(TurnWindowPolicy(1)).build(conversation)
    assert json.loads(windows_json(windows))[-1]["context"][0]["id"] == "u2"


def test_markdown_escapes_every_untrusted_table_cell() -> None:
    issue = Issue("rule|name\nnext", Severity.WARNING, "message|value\nnext", "c|1", ("u|1",))
    rendered = report_markdown(AuditReport((issue,), 1, 1))
    row = rendered.splitlines()[-1]
    assert "rule\\|name next" in row
    assert "c\\|1" in row
    assert "u\\|1" in row
    assert "message\\|value next" in row


def test_json_renderers_refuse_non_finite_values() -> None:
    utterance = Utterance(
        "u",
        "user",
        "text",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        metadata={"score": float("nan")},
    )
    windows = ContextBuilder(TurnWindowPolicy(1)).build(Conversation("c", [utterance]))
    with pytest.raises(ValueError, match="Out of range float"):
        windows_json(windows)
    report = AuditReport(
        (Issue("rule", Severity.ERROR, "message", "c", details={"score": float("inf")}),),
        1,
        1,
    )
    with pytest.raises(ValueError, match="Out of range float"):
        report_json(report)


def test_cli_build_and_audit(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            {
                "id": "demo",
                "utterances": [
                    {"id": "1", "role": "user", "text": "hi", "timestamp": "2026-01-01T00:00:00Z"},
                    {
                        "id": "2",
                        "role": "assistant",
                        "text": "hello",
                        "timestamp": "2026-01-01T00:00:01Z",
                        "reply_to": "1",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "windows.json"
    assert main(["build", str(source), "--policy", "reply-chain", "-o", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))[1]["context"][0]["id"] == "1"
    assert main(["audit", str(source), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["issues"] == 0


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out == "turnscope 0.2.0\n"


def test_cli_exit_threshold_and_errors(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "bad.json"
    source.write_text(
        '{"id":"x","utterances":['
        '{"id":"1","role":"user","text":"a","timestamp":"2026-01-01T00:00:00Z"},'
        '{"id":"2","role":"user","text":"b","timestamp":"2026-01-01T00:00:01Z"}]}',
        encoding="utf-8",
    )
    assert main(["audit", str(source), "--fail-on", "warning"]) == 1
    assert "role-transition" in capsys.readouterr().out
    assert main(["build", str(source), "--target", "missing"]) == 2
    assert "unknown target" in capsys.readouterr().err
    missing = tmp_path / "nope.json"
    assert main(["audit", str(missing)]) == 2
    assert main(["build", str(source), "--policy", "time", "--value", "9" * 1000]) == 2
    assert "Python int too large" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["build", "audit"])
def test_cli_refuses_to_overwrite_its_input(command: str, tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "input.json"
    original = '{"id":"safe","utterances":[]}'
    source.write_text(original, encoding="utf-8")
    assert main([command, str(source), "--output", str(source)]) == 2
    assert source.read_text(encoding="utf-8") == original
    assert "output path must differ" in capsys.readouterr().err

    alias = tmp_path / "hardlink.json"
    alias.hardlink_to(source)
    assert main([command, str(source), "--output", str(alias)]) == 2
    assert source.read_text(encoding="utf-8") == original


def test_cli_targets_are_global_across_multiple_conversations(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "multi.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "c1",
                    "utterances": [
                        {
                            "id": "x",
                            "role": "user",
                            "text": "one",
                            "timestamp": "2026-01-01T00:00:00Z",
                        }
                    ],
                },
                {
                    "id": "c2",
                    "utterances": [
                        {
                            "id": "y",
                            "role": "user",
                            "text": "two",
                            "timestamp": "2026-01-01T00:00:01Z",
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    assert main(["build", str(source), "--target", "y"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert [(item["conversation_id"], item["target"]["id"]) for item in output] == [("c2", "y")]


def test_cli_rejects_missing_and_ambiguous_global_targets(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "multi.json"
    source.write_text(
        '[{"id":"c1","utterances":[{"id":"dup","role":"user","text":"a","timestamp":"2026-01-01T00:00:00Z"}]},'
        '{"id":"c2","utterances":[{"id":"dup","role":"user","text":"b","timestamp":"2026-01-01T00:00:01Z"}]}]',
        encoding="utf-8",
    )
    assert main(["build", str(source), "--target", "missing"]) == 2
    assert "unknown target" in capsys.readouterr().err
    assert main(["build", str(source), "--target", "dup"]) == 2
    assert "ambiguous" in capsys.readouterr().err
