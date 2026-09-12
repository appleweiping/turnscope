# Private CGA ablation candidate fitting

`benchmarks/benchmark_neural_ablation.py` runs the **fit-only** stage of the
[frozen matched experiment](neural-cga-ablation-design.md). Each invocation fits
one named control and one predeclared seed. It does not produce official-test
predictions, evaluate a candidate, choose a winning seed, or establish model
quality. The official test cohort was previously inspected by lexical/context
work and is not described as a fresh confirmatory cohort.

This is an original driver around the original TurnScope encoders. It imports no
reference repository, executes no reference evaluator, downloads no dataset or
model, makes no provider call and installs no dependency. The local source must
be the already available official archive with SHA-256
`84e2d1ac60a3269251b5e175e549fc65cec875fdc926f99172b5b70d3ca1b122`.

## One explicit attempt

Run from a source checkout or a shipped benchmark directory paired with the
intended installed package, using an environment with the existing NumPy and CPU
PyTorch dependencies:

```text
python benchmarks/benchmark_neural_ablation.py LOCAL_ARCHIVE NEW_PRIVATE_DIRECTORY \
  --variant mean-word.v1 --seed 17
```

Both choices are required. Modes are exactly `current-turn.v1`, `mean-word.v1`
and `order-erased.v1`; seeds are exactly 17, 101 and 202. The Python entry point is
`fit_candidate(archive, directory, *, variant, seed)`. No epochs, learning rate,
architecture, sample-size, threshold or dataset-filter override is provided.
This command example is not authorization to launch all nine expensive runs.
Freeze/review the implementation and experimental inputs before running them.

The directory must be new and outside both the benchmark checkout and the
imported package. Its parent must exist. An existing successful, interrupted or
failed directory is rejected. The driver never overwrites a candidate, resumes
an optimizer, deletes an earlier attempt, retries automatically or silently
changes a failed run's seed. Inspect retained evidence before deciding how to
record a distinct subsequent attempt.

## Fixed fit and source contract

The driver reuses only the neighboring main candidate driver's pure protocol,
canonical JSON/hash and summary-binding helpers and the neighboring
`neural_cga_data.py` source adapter. It names these fixed local files rather than
searching an import path for a plugin or accepting a caller-supplied evaluator.
No previous **trained** parameters are loaded into a fit.

The shared data protocol preserves official training/test partitions. Every
official validation conversation is grouped by shared page OR reciprocal pair
before eligibility filtering; the existing fixed hash protocol assigns modulo-5
buckets 0–1 to policy validation and 2–4 to model validation. First-future-event
labels, excluded-header behavior, complete equal-time blocks, `min_turns=2`,
excluded-group leakage pins and nonterminal negative prefixes are unchanged.
The source adapter validates the fixed ZIP and only reads its bounded required
members; it does not extract or redistribute records.

Official test data may be read, schema-validated and causally **prepared for
aggregate partition auditing**. It is never passed to model fitting, prediction,
threshold selection or evaluation. The only model fit arguments are training,
model validation and policy validation. A source audit includes class/support
counts, not learned test scores. Do not equate source audit with an untouched
test cohort or with model evaluation.

All models have the fixed one-layer 64-wide reference dimensions, 128 retained
ordinary tokens per completed turn plus EOS, and train-only DF≥1 vocabulary with
at most 10,000 ordinary terms. Before optimization, the driver independently
prepares training and fits the expected vocabulary, retaining only its digest,
size, document count and tokenizer/UCD identity as the comparison pin. It verifies
the model's exact resulting vocabulary digest/size against this pin. Other seeds
or modes must independently reproduce the same vocabulary under the same input
and tokenizer policy; no official-test words enlarge it.

Training settings remain 8 maximum epochs, patience 3, batch size 16 and at most
8,192 real encoded positions per batch, learning rate 0.001, clip 5, and the
selected predeclared seed. Each control starts from the canonical untrained
initializer's active subset and fits independently. It selects its own earliest
best model-validation checkpoint, then its own policy-validation threshold using
the frozen NumPy scorer. Different selected epochs are retained, not forced to
match another model's outcome. Parameter counts, pooling/affine quotas and all
data/numeric/training settings are recorded and checked against the admitted plan.

The explicit benchmark process policy sets Torch CPU threads to 1 and enables
deterministic algorithms. `finally` restores the prior thread count,
deterministic setting and warn-only setting on success or failure. These scoped
changes belong to this driver, not the library's training API. Cross-version or
cross-machine bitwise training equivalence is not promised.

## Four retained artifacts

| File | Meaning |
| --- | --- |
| `fit-plan.json` | Full fixed protocol/digest and code-source bindings, published before source loading |
| `source-audit.json` | Source/partition audit, training-only vocabulary pin, and source/preparation time |
| `private-model.tsa` | Distinct validated ablation artifact with private vocabulary and weights |
| `candidate-receipt.json` | Completed fit, verified publication, exact restored state and aggregate evidence |

The model uses `turnscope.neural-ablation-artifact.v1`, never the main forecaster's
artifact identity. Saving and loading call the explicit ablation artifact
functions. The driver verifies the actual regular model file's SHA-256 and size
against the save result, loads it, compares the full canonical training summary
and model digest, and hashes the file again. Numeric support fields are strict
integers, so a Boolean value cannot masquerade as a count in a binding check.

Code binding hashes every `.py` file and `py.typed` from the **imported** TurnScope
package, plus this driver, `benchmark_neural_forecast.py` and `neural_cga_data.py`.
Mixed loaded TurnScope package locations are rejected. Files are checked before
source work, before optimization, after optimization and after publication/reload.
This accounts for reused helpers, not just the top-level script. Source changes
invalidate the attempt; after publication a private model can remain without a
completed receipt. These are local file-integrity checks, not code signing,
memory forensics, authentication or protection against a malicious concurrent
filesystem writer.

The final receipt explicitly records `candidate_fit_completed=True`,
`final_evaluation_completed=False`, `official_test_predictions_produced=False`,
`real_data_quality_claimed=False` and `whole_repository_parity_claimed=False`.
It retains actual loss histories, selected epoch, optimizer-step support,
threshold and policy-selection score through the verified full training summary.
The policy-validation score is selection data, not a final-test result.

`source_load_and_audit_seconds` includes loading, all partition preparation and
the independent training vocabulary pin. `fit_seconds` includes model fitting,
policy selection and scoped runtime-setting calls, but excludes source work and
save/reload. It is not a peak-RSS measurement. The source timing scope differs
from the older main driver's narrower source-load timer; compare named scopes,
not just similarly looking numbers.

## Failure and privacy boundary

Every report is exclusively published as one complete bounded JSON file. Model
publication is also exclusive. No success receipt is emitted before all binding
checks pass. A final receipt-write failure may leave a valid private model and
earlier plan/audit files; it does not undo computation or prove the model should
be discarded. A closed stdout after receipt publication returns a nonzero CLI
status while the completed on-disk receipt remains available. Read it before
considering another run, rather than inferring state from the exit code alone.

Post-publication temporary-cleanup warnings do not pretend that publication
failed. Plan/audit warnings are retained in the candidate receipt, artifact
cleanup status is in its publication record, and a final-receipt cleanup warning
is reported separately without changing the already-written receipt. Routine
handled CLI failures use a short redacted diagnostic, not exception text that
may include a private path or record. Unexpected bugs, forced termination and
external logging policies are not a guarantee of redacted diagnostics.

Vocabulary and weights remain local in the model. Reports omit raw records,
vocabulary entries and test predictions, but contain local publication paths,
hashes, class/support counts and fitted metadata. Review them before any public
sharing. Hashes are not anonymization. No raw-data license, redistribution
permission or broad privacy guarantee is inferred from having downloaded an
archive.

## Driver acceptance, not benchmark results

`tests/test_benchmark_neural_ablation.py` separates nonnumeric workflow doubles
from actual CPU fitting on small original conversations. Mocked file loading is
explicitly labelled as authored data, not verified CGA ingestion evidence. The
native cases really fit all three controls under the fixed protocol and
save/reload distinct artifacts; they do not use the official source or score a
test set. Boundary cases cover protocol choices, actual helper bindings,
source/config/support/vocabulary drift, file/receipt corruption, process-state
restoration, exclusive output, failed receipt publication and private diagnostics.

After implementation acceptance, all declared real candidates, matched NB/prior
controls, prefix-safe ordering diagnostic and separately authorized heldout
evaluation still need their own frozen, source-bound evidence. Neither a driver
test pass nor a successful fit establishes quality or completes those comparisons.
