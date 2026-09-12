# Installed three-control demonstration

`examples/neural_ablation_demo.py` trains `current-turn.v1`, `mean-word.v1`, and
`order-erased.v1` on a tiny authored fixture, saves their distinct inference
artifacts, then runs `inspect`, `predict`, and `evaluate` in fresh Python
processes with Torch imports forbidden. It exercises the actual command-line
entry point, not a mock model or a hand-written prediction receipt.

Install a wheel containing the same source version as the shipped helper, with
the `neural-train` extra available. From a source distribution's extracted
helper directory, run:

```bash
python -I -X utf8 examples/neural_ablation_demo.py private-ablation-demo
```

The parent directory must exist and `private-ablation-demo` must not exist.
Neither the helper nor its children prepend a checkout to `sys.path`; `-I`
ignores source-path environment overrides. The three-OS installed CI job first
compares wheel, source distribution and installed runtime bytes, extracts only
the shipped helpers/tests, then verifies the demo's actual imported source
bindings against those distributions. An in-process unit-test fixture is
explicitly **not** evidence of installed isolation; the separate run is.

## Fixed authored workload

Every fitting and evaluation partition has four conversations, two positive and
two censored negative. Each has two complete observations and a third event or
censoring turn, giving four eligible prefixes per partition. Their group
inventories are disjoint. The separate prediction file contains only the two
completed observations per conversation. Text patterns repeat across these
tiny partitions; the demonstration is not a generalization experiment.

All three controls use fixed 64-dimensional, one-layer numerical contracts,
seed 17, two epochs and conversation batches of size three. Each executes one
full and one partial optimizer batch per epoch. Their train-derived vocabulary
and full untrained canonical parameter hash inventories must agree, while the
active arrays remain distinct: 16, 9 and 5, respectively. Each control selects
its own epoch and policy-validation threshold. No score or accuracy threshold
is required to pass: poor heldout results remain in the aggregate report.
The training children explicitly use one CPU thread and deterministic Torch
algorithms; these isolated process settings do not change the invoking process
or the public API's caller-owned Torch settings.

The helper verifies all four command formats, mode/model identities, exact
saved artifact SHA and byte count, full fitted/restored summary equality,
source-file SHA/byte/support receipts, fitting support, actual optimizer-step
counts, mode-specific parameter counts, common initialization and prediction
privacy. Predicted finite logits, clipped probabilities, boolean decisions and
the fitted threshold must agree. The entire closed heldout metric tree is
independently recomputed from the four delivered scores and known labels;
unknown metric fields are rejected rather than copied into the public aggregate.
Heldout metrics must retain the authored support denominators. Runtime,
helper and input file hashes are checked before and after command work.

## Output and failure boundaries

The new local directory contains private JSONL inputs and settings, plus one
subdirectory per mode holding its vocabulary/weight artifact and four command
reports. The final `demo-report.json` includes aggregate metrics, model/source
bindings and hashes of generated files. It omits raw conversations, vocabulary,
weights and per-conversation predictions; only that aggregate is uploaded by CI.
Hashes provide integrity bindings, not identity authentication or anonymization.

There are no retries or output-directory reuse. A failed command leaves its
private attempt for inspection without emitting a successful final demo report.
Failure to deliver final stdout returns a failure status but does not erase the
already delivered aggregate file or earlier artifacts. Keep local failure trees
private; diagnostics or intermediate files may include authored data.

This demo proves a bounded execution path on authored data only. It does not
score CGA-WIKI, claim probability calibration, establish model quality, or show
whole-repository parity. Actual real-corpus candidates follow the separate
[frozen control design](neural-cga-ablation-design.md) and require an external
[artifact/source/protocol receipt](neural-ablation-artifact.md#provenance-refinement-before-research-scoring).
