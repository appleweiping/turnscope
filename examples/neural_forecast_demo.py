"""Train an authored tiny fixture, then deploy without Torch from an installed wheel.

Run with Python -I; this helper never prepends a source checkout to sys.path.
All generated inputs and models remain in a new caller-selected local directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

_FROZEN_COMMAND = """
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise AssertionError('frozen command attempted to import Torch')
sys.meta_path.insert(0, NoTorch())
from turnscope.cli import main
code = main(sys.argv[1:])
if any(name.split('.')[0] == 'torch' for name in sys.modules):
    raise AssertionError('frozen command imported Torch')
raise SystemExit(code)
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(partition: str, *, observations_only: bool = False) -> list[dict[str, Any]]:
    records = []
    for index in range(4):
        event = index % 2 == 0
        texts = (
            ["We disagree about this evidence.", "Please reconsider this disputed claim."]
            if event
            else ["We agree to compare evidence.", "Thank you for checking this claim."]
        )
        if not observations_only:
            texts.append("An authored annotated event." if event else "A censored ending.")
        records.append(
            {
                "id": f"authored-{partition}-{index}",
                "metadata": {"forecast_groups": [f"authored-group:{partition}:{index}"]},
                "utterances": [
                    {
                        "id": f"turn-{turn}",
                        "role": "person",
                        "text": text,
                        "timestamp": f"2026-01-01T00:00:{turn:02d}+00:00",
                        "metadata": {"event": event and turn == 2},
                    }
                    for turn, text in enumerate(texts)
                ],
            }
        )
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _run(arguments: list[str], *, frozen: bool) -> None:
    command = [sys.executable, "-I", "-X", "utf8"]
    command.extend(["-c", _FROZEN_COMMAND] if frozen else ["-m", "turnscope"])
    subprocess.run(
        [*command, "neural-forecast", *arguments],
        check=True,
        capture_output=True,
        encoding="utf-8",
        timeout=180,
    )


def _validate_reports(directory: Path, reports: dict[str, Any]) -> None:
    """Check delivered command/source identities, not just a shared model string."""
    formats = {
        "training": "turnscope.neural-training-command.v1",
        "inspection": "turnscope.neural-inspection-command.v1",
        "prediction": "turnscope.neural-prediction-command.v1",
        "evaluation": "turnscope.neural-forecast-report.v1",
    }
    if reports.keys() != formats.keys() or any(
        reports[name].get("format") != expected for name, expected in formats.items()
    ):
        raise ValueError("unexpected command report format")
    training = reports["training"]
    if training.get("completed") is not True:
        raise ValueError("training did not report completed publication")
    publication = training["publication"]
    model = directory / "private-model.tsn"
    if (
        Path(publication["path"]) != model
        or publication["sha256"] != _sha(model)
        or publication["bytes_written"] != model.stat().st_size
    ):
        raise ValueError("model receipt does not bind the actual published file")
    if reports["inspection"]["training"] != training["training"]:
        raise ValueError("restored training metadata differs from the training receipt")
    sources = training["sources"]
    if sources.keys() != {"training", "model_validation", "policy_validation"}:
        raise ValueError("fitting source inventory differs")
    for name, file in (
        ("training", "training.jsonl"),
        ("model_validation", "validation.jsonl"),
        ("policy_validation", "policy.jsonl"),
    ):
        if sources[name]["sha256"] != _sha(directory / file) or any(
            type(sources[name][key]) is not int or sources[name][key] != expected
            for key, expected in {
                "bytes": (directory / file).stat().st_size,
                "conversations": 4,
                "utterances": 12,
            }.items()
        ):
            raise ValueError("fitting source receipt differs from authored inputs")
    for name, file in (("prediction", "observed.jsonl"), ("evaluation", "heldout.jsonl")):
        source = reports[name]["source"]
        if source["sha256"] != _sha(directory / file) or any(
            type(source[key]) is not int or source[key] != expected
            for key, expected in {
                "bytes": (directory / file).stat().st_size,
                "conversations": 4,
                "utterances": 8 if name == "prediction" else 12,
            }.items()
        ):
            raise ValueError("deployment source receipt differs from authored inputs")
    prediction = reports["prediction"]
    if (
        prediction["identifiers_included"] is not False
        or prediction["text_included"] is not False
        or any(type(row["index"]) is not int for row in prediction["predictions"])
        or [row["index"] for row in prediction["predictions"]] != list(range(4))
    ):
        raise ValueError("private prediction flags or ordered row inventory differs")
    evaluation = reports["evaluation"]
    if any(
        type(evaluation["metrics"][key]) is not int or evaluation["metrics"][key] != count
        for key, count in {"conversations": 4, "prefixes": 4, "positive_conversations": 2}.items()
    ):
        raise ValueError("heldout metric denominators differ from the authored fixture")
    if any(
        type(evaluation["support"][key]) is not int or evaluation["support"][key] != count
        for key, count in {
            "source_conversations": 4,
            "source_turns": 12,
            "header_turns": 0,
            "eligible_conversations": 4,
            "excluded_conversations": 0,
            "observed_turns": 8,
            "prefixes": 4,
            "positive_conversations": 2,
            "negative_conversations": 2,
        }.items()
    ) or any(
        evaluation[key] is not False
        for key in (
            "policy_reuses_model_validation",
            "probability_calibration_claimed",
            "whole_repository_parity_claimed",
        )
    ):
        raise ValueError("heldout support or claim boundaries differ")


def run_demo(directory: Path) -> dict[str, Any]:
    """Exercise real commands; no retries and no reuse of a pre-existing output tree."""
    import turnscope
    from turnscope.neural_cli_io import publish_neural_report

    directory = directory.absolute()
    directory.mkdir()  # Deliberately exclusive; a failed run is not silently resumed.
    for name in ("training", "validation", "policy", "heldout"):
        _write_jsonl(directory / f"{name}.jsonl", _records(name))
    _write_jsonl(directory / "observed.jsonl", _records("heldout", observations_only=True))
    settings = {
        "config": {"embedding_dim": 4, "word_hidden": 4, "turn_hidden": 8},
        "training_config": {"epochs": 2, "patience": 2, "batch_conversations": 3, "seed": 17},
    }
    publish_neural_report(settings, directory / "settings.json")
    model = directory / "private-model.tsn"
    _run(
        [
            "fit",
            str(directory / "training.jsonl"),
            str(directory / "validation.jsonl"),
            str(model),
            "--policy-validation",
            str(directory / "policy.jsonl"),
            "--settings",
            str(directory / "settings.json"),
            "--output",
            str(directory / "training-report.json"),
        ],
        frozen=False,
    )
    for action, source, report in (
        ("inspect", None, "inspection-report.json"),
        ("predict", "observed.jsonl", "prediction-report.json"),
        ("evaluate", "heldout.jsonl", "evaluation-report.json"),
    ):
        arguments = [action, str(model)]
        if source is not None:
            arguments.append(str(directory / source))
        _run([*arguments, "--output", str(directory / report)], frozen=True)

    reports = {
        name: json.loads((directory / f"{name}-report.json").read_text(encoding="utf-8"))
        for name in ("training", "inspection", "prediction", "evaluation")
    }
    _validate_reports(directory, reports)
    digest = reports["training"]["model_digest"]
    if any(report["model_digest"] != digest for report in reports.values()):
        raise ValueError("commands disagree about the fitted model identity")
    trained = reports["training"]["training"]
    if trained["policy_reuses_model_validation"] or any(
        trained[key] != 4
        for key in (
            "training_conversations",
            "model_validation_conversations",
            "policy_validation_conversations",
            "training_prefixes",
            "model_validation_prefixes",
            "policy_validation_prefixes",
        )
    ):
        raise ValueError("training did not use the declared separate authored partitions")
    if len(trained["history"]) != 2 or any(
        epoch["optimizer_steps"] != 2 for epoch in trained["history"]
    ):
        raise ValueError("training did not include each full and partial batch")
    predictions = reports["prediction"]["predictions"]
    if len(predictions) != 4 or any(
        row["observed_turns"] != 2 or "conversation_id" in row or "text" in row
        for row in predictions
    ):
        raise ValueError("prediction did not preserve the observed-only private output contract")
    package = Path(turnscope.__file__).resolve().parent
    source_bindings = {
        f"src/turnscope/{path.relative_to(package).as_posix()}": _sha(path)
        for path in sorted(package.rglob("*.py"))
    }
    source_bindings["src/turnscope/py.typed"] = _sha(package / "py.typed")
    source_bindings["examples/neural_forecast_demo.py"] = _sha(Path(__file__))
    report = {
        "format": "turnscope.neural-installed-demo.v1",
        "completed": True,
        "authored_fixture_only": True,
        "real_data_quality_claimed": False,
        "whole_repository_parity_claimed": False,
        "model_digest": digest,
        "model_file_sha256": _sha(model),
        "selected_epoch": trained["selected_epoch"],
        "optimizer_steps_executed": sum(epoch["optimizer_steps"] for epoch in trained["history"]),
        "training_conversations": 4,
        "model_validation_conversations": 4,
        "policy_validation_conversations": 4,
        "heldout_predictions": len(predictions),
        "authored_heldout_metrics": reports["evaluation"]["metrics"],
        "frozen_commands_with_torch_forbidden": ["inspect", "predict", "evaluate"],
        "python_version": sys.version.split()[0],
        "torch_training_version": trained["torch_version"],
        "numpy_training_version": trained["numpy_version"],
        "source_sha256": source_bindings,
        "file_sha256": {path.name: _sha(path) for path in sorted(directory.iterdir())},
    }
    publish_neural_report(report, directory / "demo-report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    from turnscope.neural_cli_io import NeuralOutputError, publish_neural_report

    options: dict[str, Any] = {}
    if sys.version_info >= (3, 14):
        options = {"color": False, "formatter_class": partial(argparse.HelpFormatter, color=False)}
    parser = argparse.ArgumentParser(description=__doc__, **options)
    parser.add_argument("directory", type=Path, help="new private output directory")
    args = parser.parse_args(argv)
    report = run_demo(args.directory)
    try:
        publish_neural_report(report, None)
    except NeuralOutputError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
