import json
from pathlib import Path

import pytest

from turnscope.cli import main


def test_cli_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "input.jsonl"
    source.write_text('{"id":"one","utterances":[]}\n', encoding="utf-8")
    database = tmp_path / "corpus.db"
    assert main(["corpus", "import", str(database), str(source)]) == 0
    assert json.loads(capsys.readouterr().out) == {"imported": 1}
    assert main(["corpus", "stats", str(database)]) == 0
    assert json.loads(capsys.readouterr().out) == {"conversations": 1, "utterances": 0}
    assert main(["corpus", "list", str(database), "--limit", "1"]) == 0
    assert json.loads(capsys.readouterr().out) == ["one"]
    assert main(["corpus", "get", str(database), "one"]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == "one"
    assert main(["corpus", "import", str(database), str(source)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert main(["corpus", "import", str(database), str(source), "--replace"]) == 0


def test_cli_refuses_alias_and_missing_inputs(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_text("[]", encoding="utf-8")
    assert main(["corpus", "import", str(source), str(source)]) == 2
    assert source.read_text(encoding="utf-8") == "[]"
    missing = tmp_path / "missing"
    database = tmp_path / "absent.db"
    assert main(["corpus", "import", str(database), str(missing)]) == 2
    assert main(["corpus", "stats", str(database)]) == 2
    assert not database.exists()


def test_cli_handles_non_database(tmp_path: Path) -> None:
    database = tmp_path / "not.db"
    database.write_text("not a sqlite database", encoding="utf-8")
    assert main(["corpus", "stats", str(database)]) == 2
