# Explicit neural-control commands

`turnscope neural-ablation` is separate from `neural-forecast`. Fitting requires
both an explicit `--variant` and a third `--policy-validation` input. There is no
default control, automatic main-model conversion, two-partition reuse, model
download, remote provider, or background retraining.

The modes are exactly `current-turn.v1`, `mean-word.v1`, and `order-erased.v1`.
They use the fixed one-layer, 64-wide architectures described in
[the control design](neural-cga-ablation-design.md). Their smaller parameter
inventories are not interchangeable with the main neural model's artifact.

## Fit and inspect

```console
turnscope neural-ablation fit training.jsonl validation.jsonl private-control.tsa --policy-validation policy.jsonl --variant mean-word.v1 --settings settings.json --output fit-report.json
turnscope neural-ablation inspect private-control.tsa --output inspection.json
```

The three fitting files must be distinct, including resolved-path and hard-link
aliases. Fitting-group identities and partition support must also pass the
model's three-way isolation checks. Equal content under different paths does
not establish independence. Model-validation data select the epoch; the separate
policy-validation data select that candidate's threshold. The command does not
use heldout test data for either decision.

`settings.json` is optional and accepts only the existing five object sections:
`config`, `training_config`, `policy`, `data_limits`, and `numeric_limits`.
Unknown sections or fields reject. `config.variant` is forbidden even when it
matches the command option: `--variant` is the sole mode selector. For example,
this small authored-test configuration retains the fixed architecture:

```json
{
  "config": {"max_turn_tokens": 4},
  "training_config": {"epochs": 1, "batch_conversations": 2, "seed": 17}
}
```

Omitted values use the typed Python API defaults. In particular, this command
does not expose new `pooling_limits` or `training_limits` JSON sections.
`AblationPoolingLimits.max_operations` stays at 100,000,000 per admitted
inference input, and `AblationTrainingLimits.max_total_pooling_operations` stays
at 100,000,000,000 for the configured-epoch training estimate. The existing
`config.max_inference_pooling_operations` remains an independently configurable
aggregate deployment/policy bound. Affine multiplication and pooling operation
budgets are distinct; neither is a measured total instruction count or RSS cap.

Fit requires the existing optional CPU Torch dependency. Parsing help does not
import Torch or NumPy. Inspection, prediction and evaluation load the separate
validated NumPy artifact without importing Torch. No command installs missing
dependencies or changes process-wide Torch thread settings. An actual fit can
be expensive; a controlled failure is not authorization to retry automatically.

## Prediction and heldout evaluation

```console
turnscope neural-ablation predict private-control.tsa observed.jsonl --output predictions.json
turnscope neural-ablation predict private-control.tsa observed.jsonl --include-identifiers
turnscope neural-ablation evaluate private-control.tsa heldout.jsonl --output evaluation.json
```

Inputs use the same bounded conversation JSONL schema as the existing neural
commands: one UTF-8 conversation per nonblank line, with unique conversation and
within-conversation utterance IDs. For `predict`, callers must supply only the
completed observations actually available at the intended prediction time.
Prediction applies the saved skip policy but intentionally ignores outcome/event
metadata: it does not infer or truncate a future boundary from labels. Supplying
a complete conversation, including later text, therefore makes that text part of
the supplied observations. Supervised preparation/evaluation instead constructs
eligible prefixes before the labelled event. Evaluation scores those prefixes,
not the prediction command's one final observed output per record, and rejects
groups belonging to any of the three fitting partitions.

The shared reader limits each file to 64 MiB, each line to 16 MiB, each file to
10,000 conversations and 100,000 source utterances, JSON depth to 32, and the
aggregate JSON node budget to 2,000,000. Files must be regular files, not symlinks
or pipes. Byte/identity checks surround reads; duplicate keys, nonfinite JSON,
invalid UTF-8 and malformed typed records reject. The saved model's possibly
stricter source, token and inference budgets still apply. These input limits are
per file; they are not a newly claimed combined three-file process-memory cap.

## Report contents and privacy

The result formats are:

| Command | Format |
| --- | --- |
| Fit | `turnscope.neural-ablation-training-command.v1` |
| Inspect | `turnscope.neural-ablation-inspection-command.v1` |
| Predict | `turnscope.neural-ablation-prediction-command.v1` |
| Evaluate | `turnscope.neural-ablation-report.v1` |

Fit reports include the explicit mode, model identity, publication receipt,
all three source-file byte hashes/counts, actual training summary and a private
artifact notice. The publication receipt includes its local output path.
Inspection omits vocabulary and weight arrays. Prediction reports stable row
indices, output probabilities/logits, work counters and source/model digests;
conversation IDs appear only with `--include-identifiers`. Input source text is
never added to these reports. Evaluation retains the existing model report's
explicit metric/support denominators and adds its source-file inventory.

Reports are not automatically public or anonymous: hashes, output paths,
configuration field names, small-group metrics, per-input predictions and
optional identifiers may be sensitive. The private bundle includes the full
vocabulary and weights. An arbitrary custom configuration value is not a secret
storage mechanism. Runtime command failures use fixed diagnostics instead of
echoing source rows, filesystem paths or native exception messages; argument
syntax errors can still be handled by the ordinary argument parser, so do not
put secrets in command-line arguments.

Fit and inspection explicitly set `source_execution_provenance_claimed=false`.
A source-file hash and a self-consistent artifact do not authenticate a training
execution or prove scientific provenance. For benchmark acceptance, retain the
separate source/runtime/protocol-bound candidate-fit receipt described in
[the artifact documentation](neural-ablation-artifact.md). No CLI success flag
asserts CGA quality, whole-repository parity or fairness/calibration validation.

## Output and failure contract

All output paths must be new, with an existing parent directory, and differ
from every protected input, model path and settings path. There is no CLI
overwrite switch. The model and optional report are independently published
through their existing exclusive atomic publication helpers; they are **not**
one transaction. The shared JSON report publisher limits complete UTF-8 output
to 32 MiB and emits no partial report file on an ordinary pre-publication failure.
The trusted-local-directory and filesystem race/power-loss limitations from the
artifact documentation still apply.

Exit status has three meanings:

- `0`: command work and report publication succeeded. A cleanup warning may
  still indicate a private temporary link could not be removed; inspect the
  artifact receipt and diagnostics.
- `1`: command work completed but report output failed. After fit, the artifact
  **was already published**. Do not infer rollback or automatically repeat
  training. A stdout consumer may have received a prefix of the report; failed
  output is not automatically resent. A cleanup failure after publication does
  not undo a successfully published model or report.
- `2`: a controlled input, settings, model, dependency, numerical or I/O failure
  prevented completion of command work. No report is presented as successful.
  This is not a general transaction rollback guarantee for native work or a
  failed artifact publication whose cleanup also failed.

Broken stdout after completed work is handled without CPython's later flush
changing the explicit status to 120. Closed or absent embedded output streams
likewise do not hide the chosen result. Diagnostics are best effort when stderr
itself is unavailable. For portable redirected Unicode output, run Python with
`-X utf8` or configure the host's UTF-8 standard streams; the command never
silently changes the process's text encoding.
