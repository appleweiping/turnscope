import csv
import io
import json

from turnscope.builder import ContextBuilder
from turnscope.cli import main
from turnscope.models import Conversation
from turnscope.policies import TurnWindowPolicy
from turnscope.tabular import window_rows, windows_csv


def test_windows_csv_is_deterministic_and_redacts_text_by_default(
    conversation: Conversation,
) -> None:
    windows = ContextBuilder(TurnWindowPolicy(1)).build(conversation)
    rendered = windows_csv(windows)
    assert rendered == windows_csv(tuple(windows))
    rows = list(csv.DictReader(io.StringIO(rendered)))
    assert rows[-1]["target_id"] == "a2"
    assert json.loads(rows[-1]["context_ids"]) == ["u2"]
    assert "target_text" not in rows[-1]
    assert "a useful answer" not in rendered


def test_windows_csv_can_include_text(conversation: Conversation) -> None:
    window = ContextBuilder(TurnWindowPolicy(1)).build(conversation)[-1]
    row = window_rows([window], include_text=True)[0]
    assert row["target_text"] == "a useful answer"
    assert json.loads(str(row["context_text"])) == ["explain this"]


def test_tabular_cli_writes_csv(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "input.json"
    output = tmp_path / "windows.csv"
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
    assert main(["tabular", str(source), "--output", str(output), "--include-text"]) == 0
    rows = list(csv.DictReader(output.open(encoding="utf-8", newline="")))
    assert rows[-1]["target_text"] == "hello"
    assert capsys.readouterr().out == ""
