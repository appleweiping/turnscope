"""Independent report fault probes using genuine authored CLI report templates.

The module fixture runs real tiny training/artifact/inference commands in the
current test interpreter. It deliberately does not claim installed isolation or
Torch-free subprocess execution; the separate installed demonstration proves
those properties. There are no fake predictions or substituted trained weights.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from turnscope.cli import main
from turnscope.neural_forecast import HierarchicalEventForecaster

DEMO_PATH = Path(__file__).resolve().parents[1] / "examples/neural_forecast_demo.py"
SPEC = importlib.util.spec_from_file_location("independent_neural_demo_review", DEMO_PATH)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


@pytest.fixture(scope="module")
def actual_reports(tmp_path_factory):
    pytest.importorskip("torch", reason="genuine report templates require optional CPU training")
    output = tmp_path_factory.mktemp("neural-demo-review") / "authored-run"

    def real_cli_in_current_interpreter(arguments, *, frozen):
        # Isolation is independently verified by the installed -I demo. Keeping
        # this fixture in-process also works for source-checkout test environments
        # where only PYTHONPATH makes the currently developed package available.
        assert type(frozen) is bool
        assert main(["neural-forecast", *arguments]) == 0

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(demo, "_run", real_cli_in_current_interpreter)
        result = demo.run_demo(output)
    assert result["completed"] is True
    reports = {
        name: json.loads((output / f"{name}-report.json").read_text(encoding="utf-8"))
        for name in ("training", "inspection", "prediction", "evaluation")
    }
    return output, reports


def test_actual_report_templates_validate_and_public_model_wrappers_roundtrip(actual_reports):
    directory, reports = actual_reports
    demo._validate_reports(directory, reports)
    model = HierarchicalEventForecaster.load(directory / "private-model.tsn")
    assert model.digest == reports["training"]["model_digest"]
    target = directory.parent / "public-wrapper.tsn"
    publication = model.save(target)
    assert Path(publication.path) == target
    assert publication.sha256 == demo._sha(target)
    assert HierarchicalEventForecaster.load(target).digest == model.digest
    with pytest.raises(FileExistsError):
        model.save(target)


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("training", "completed"), False),
        (("training", "format"), "wrong-format"),
        (("training", "publication", "sha256"), "0" * 64),
        (("training", "publication", "bytes_written"), 0),
        (("inspection", "training", "selected_epoch"), 0),
        (("training", "sources", "training", "sha256"), "0" * 64),
        (("training", "sources", "model_validation", "sha256"), "0" * 64),
        (("training", "sources", "policy_validation", "sha256"), "0" * 64),
        (("prediction", "source", "sha256"), "0" * 64),
        (("evaluation", "source", "sha256"), "0" * 64),
        (("prediction", "identifiers_included"), True),
        (("prediction", "text_included"), True),
        (("prediction", "predictions", 0, "index"), 1),
        (("evaluation", "support", "prefixes"), 0),
        (("evaluation", "probability_calibration_claimed"), True),
        # Three independent gaps reproduced against actual v2 installed outputs:
        # an unchanged SHA alone does not make false byte/count claims valid;
        # False compares equal to index0; metric support must match data support.
        (("training", "sources", "training", "bytes"), 0),
        (("training", "sources", "training", "conversations"), 0),
        (("prediction", "predictions", 0, "index"), False),
        (("evaluation", "metrics", "conversations"), 0),
        (("evaluation", "metrics", "prefixes"), 0),
    ],
)
def test_inconsistent_command_receipt_cannot_be_accepted(actual_reports, path, replacement):
    directory, reports = actual_reports
    changed = copy.deepcopy(reports)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ValueError):
        demo._validate_reports(directory, changed)
