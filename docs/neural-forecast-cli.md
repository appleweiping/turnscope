# Neural forecast command workflow

This development workflow trains the [hierarchical event model](neural-forecast.md)
on local conversations. It never calls an online service, downloads a model or
loads a Torch checkpoint. Install the development branch/checkout described in
the README, not an older release with a different command surface.

For training, use a CPU-capable PyTorch installation compatible with your Python
and platform, plus `python -m pip install -e ".[neural-train]"`. To run frozen
inference only, `python -m pip install -e ".[neural]"` requires NumPy but not
Torch. Base-package import and every command's `--help` need neither dependency.
The package does not change the process-global thread count or random seed.

## Input and selection partitions

Each JSONL line is one conversation; blank lines are rejected:

```json
{"id":"discussion-1","utterances":[{"id":"a","role":"person","text":"Can we check this?","timestamp":"2026-01-01T00:00:00Z","metadata":{"event":false}},{"id":"b","role":"person","text":"Let us compare the evidence.","timestamp":"2026-01-01T00:01:00Z","metadata":{"event":false}},{"id":"c","role":"person","text":"An annotated future event.","timestamp":"2026-01-01T00:02:00Z","metadata":{"event":true}}],"metadata":{"forecast_groups":["page:1","pair:1"]}}
```

This is one authored schema illustration, not enough data to fit a model. Every
fitting partition needs eligible positive and negative conversations. Use honest
group identities for related conversations, authors/pages or paired samples as
appropriate. Conversation identity is included in leakage checks even if no
extra group metadata is supplied. Hashes preserve declared identity; they cannot
discover an undeclared relationship or prove real-world sample independence.

Labels are strict booleans under `metadata.event` by default; explicit
`metadata.is_section_header` marks a non-observation. Preparation observes only
complete pre-event turns, keeps equal-time blocks atomic, and excludes the final
block from negatives. Defaults require two observed non-header turns. Future
event text is not a model feature. See [the data contract](neural-forecast-data.md)
for eligibility, resource budgets, exclusions and head truncation.

The training split fits the vocabulary and gradients. Model validation chooses
the earliest minimum-loss checkpoint. Optional policy validation separately
chooses the alert threshold. These three partitions must have disjoint declared
groups, including excluded conversations. Omitting `--policy-validation`
deliberately reuses model validation for threshold selection and records that
fact; it is not independent third-partition selection. Final evaluation must be
disjoint from all fitting partitions and never adjusts the model or threshold.

## Fit, inspect, predict and evaluate

```bash
turnscope neural-forecast fit train.jsonl model-validation.jsonl private-model.tsn \
  --policy-validation policy-validation.jsonl --settings neural-settings.json \
  --output training-report.json
turnscope neural-forecast inspect private-model.tsn --output model-info.json
turnscope neural-forecast predict private-model.tsn observed-now.jsonl --output predictions.json
turnscope neural-forecast evaluate private-model.tsn heldout.jsonl --output evaluation.json
```

All output names above must be new. The CLI has no overwrite or resume switch.
Input/model/settings/report aliases and existing destinations are rejected. A
supported new private directory is required; no network URI is accepted.

Settings contain only the optional objects `config`, `training_config`, `policy`,
`data_limits` and `numeric_limits`, with closed typed constructor fields:

```json
{
  "config": {
    "embedding_dim": 64,
    "word_hidden": 64,
    "turn_hidden": 64,
    "max_turn_tokens": 128,
    "long_turn_policy": "head"
  },
  "training_config": {
    "epochs": 8,
    "patience": 3,
    "batch_conversations": 16,
    "seed": 17
  }
}
```

These are demonstration defaults, not empirically optimal hyperparameters. No
configuration is inferred from the final test set. Unknown options, boolean
integers, nonfinite values and over-budget workloads fail explicitly. Training
pre-admits all configured epochs, even if early stopping may use fewer.

For `predict`, each record must contain only the complete observations actually
available now. The command removes explicit headers, but intentionally does not
use event labels to cut an input conversation. Passing an entire labelled future
conversation to prediction therefore violates the caller's causal-input
responsibility. `evaluate` instead takes labelled conversations and constructs
eligible past-only prefixes using the fixed preparation policy.

## Privacy and reports

Training/inspection reports include configs, loss history, parameter and split
hashes, support, threshold, dependency declarations and work estimates. They omit
raw text and vocabulary. The model bundle **does** contain vocabulary and weights
and is private training information. Do not publish a real-data bundle without
checking the source's permissions and disclosure risks.

Predictions contain input-order `index`, probabilities/alerts, token-retention
counts, input/model digests and declared work. Conversation IDs are omitted
unless `--include-identifiers` is explicitly passed; text is never included.
Hashes are not anonymization, and omitting raw text is not a privacy guarantee.
File paths in the publication receipt can also be sensitive.

Evaluation reports conversation-weighted prefix losses/AUC and any-alert
conversation confusion counts, first-alert lead time with its own denominator,
eligibility/exclusion counts and selection reuse. Probabilities are not claimed
to be calibrated. A model trained on a small authored fixture proves workflow
execution, not useful prediction on human dialogue, fairness or deployment safety.

## Bounded I/O and partial publication

JSONL requires regular files with strict UTF-8, duplicate-key rejection, finite
JSON values and bounded pre-decoder depth/node counts. Current CLI hard caps are
64 MiB per source file, 16 MiB per line, 10,000 conversations and 100,000 source
utterances. Settings are at most 64 KiB; reports are at most 32 MiB. Model-specific
data limits can be stricter. Source digests/counts describe the bytes actually
read, with file identity checked before/after. These are admission safeguards,
not a sandbox against a malicious filesystem or a hard RSS limit.

Reports use complete temporary writes and exclusive atomic publication, or a
checked write and flush to stdout. Short writes and broken/closed stdout fail;
real broken pipes exit with status 1 rather than a later interpreter-flush 120.
Invalid input or an ordinary file/report publication error returns status 2.

Importantly, `fit` publishes the model **before** it publishes/delivers its
aggregate report. If the later report fails, the model remains saved even though
the command exits nonzero. The two files are not a single transaction. Inspect the
existing model before retrying; do not assume rollback or delete it automatically.
A cleanup warning in a successful artifact receipt likewise does not mean that
the artifact failed to publish. The [artifact contract](neural-forecast-artifact.md)
describes both exclusive creation and explicit Python API replacement semantics.

## Installed demonstration

With the training dependencies installed, run the shipped helper from the same
source distribution as the installed wheel:

```bash
python -I examples/neural_forecast_demo.py new-private-demo-directory
```

It authors separate tiny local partitions, trains through the actual CLI, then
loads the resulting artifact for inspection, observed-only prediction and heldout
evaluation in new subprocesses that forbid importing Torch. It checks that all
four commands refer to the same model, preserves receipts and hashes, and writes
an aggregate `demo-report.json`. There is no network call, pretrained checkpoint,
reference corpus or quality claim. The helper uses the installed package and does
not add a checkout to `sys.path`; `-I` deliberately ignores `PYTHONPATH`.
