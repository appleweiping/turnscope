# Hierarchical event forecasting

`HierarchicalEventForecaster` connects ordered causal data preparation,
training, validation-only alert selection and frozen numerical inference. It
is an additive workflow: the existing lexical `PrefixEventForecaster` remains
available and unchanged. The architecture and outputs are original; this is
not a claim to reproduce a pretrained CRAFT model or attain whole-repository
ConvoKit parity.

This page covers the in-memory workflow. It does not define a file artifact,
save/load API, resumable optimizer or CLI. See [sequence data](neural-forecast-data.md)
for causal eligibility and tokenizer contracts, and [CPU training](neural-forecast-training.md)
for the objective and optimization details.

## Separate fitting decisions from final evaluation

There are two validation decisions:

1. **Model validation** chooses the earliest checkpoint with strictly lowest
   conversation-weighted prefix loss, subject to patience.
2. **Policy validation** chooses the alert threshold using complete
   conversation maximum prefix probabilities, maximizing balanced accuracy.
   Exact ties choose the higher threshold.

Pass a third fitting partition as `policy_validation=` to separate these
decisions. Training, model-validation and policy-validation hashed group sets
must be pairwise disjoint. If the third partition is omitted, the same
model-validation data is deliberately reused for threshold selection; the
state and every evaluation report record
`policy_reuses_model_validation=True`. Reuse is not described as independent
validation. Passing the same partition explicitly as a third partition is
rejected rather than silently treated as independent.

Every fitting selection partition requires eligible conversations from both
classes. Group checks include conversations excluded for too few pre-event
observations. A separately supplied evaluation partition must be disjoint from
all fitting partitions. A final evaluation with only one class can still be
reported, but class-dependent undefined metrics remain `None`, not a fabricated
number. A partition with no eligible prefixes is rejected.

```python
from turnscope.neural_forecast import HierarchicalEventForecaster, NeuralForecastConfig
from turnscope.neural_forecast_train import NeuralTrainingConfig


def fit_and_evaluate(train, model_validation, policy_validation, heldout):
    model = HierarchicalEventForecaster(
        config=NeuralForecastConfig(
            embedding_dim=64,
            word_hidden=64,
            turn_hidden=64,
            max_turn_tokens=128,
            long_turn_policy="head",
        ),
        training_config=NeuralTrainingConfig(epochs=8, patience=3, batch_conversations=16, seed=17),
    ).fit(train, model_validation, policy_validation=policy_validation)
    report = model.evaluate(heldout)
    return model, report
```

Arguments may be source `Conversation` iterables or closed
`SequenceForecastDataset` instances. `model.prepare(source)` produces the
compact supervised dataset using the model's configured `SequencePolicy` and
`SequenceLimits`. Already-prepared data must satisfy limits and minimum-turn
checks; it carries compact observations and declared supervision, not the
omitted original source. The caller remains responsible for the original
source provenance and using the intended preparation policy. Hashes cannot
authenticate historical labels or prove groups represent independent samples.

## Predictions accept observations, not future labels

`model.predict(observed_prefix)` accepts exactly an immutable `ObservedPrefix`.
It does not accept a supervised example or a raw conversation as a shortcut.
`model.predict_conversation(conversation)` applies the label-free observation
adapter using the fitted policy; the caller must supply only completed turns
available at the prediction time. It removes explicit non-observation headers
but does not inspect event labels, group metadata, role or reply annotations.

`model.transform(iterable_of_observed_prefixes)` validates the complete bounded
collection before numerical inference and returns predictions in input order.
An empty collection returns an empty tuple. It does not change the vocabulary,
fit statistics, thresholds or model state.

Each `NeuralForecastPrediction` reports:

- The last complete turn's logit, probability, threshold and `alert` decision
  under `probability >= threshold`.
- Observed-turn, raw-token, retained-token, known-token and truncated-turn
  counts, plus derived retention fractions in `to_dict()`.
- The observation digest, fitted-model digest, affine multiplication count
  and estimated inference workspace bytes.

Probabilities are clipped to `[1e-15, 1 - 1e-15]` to keep downstream logarithms
finite. The result explicitly makes no probability-calibration claim. Token
coverage is not confidence: an unknown or empty observation sequence still
passes through the frozen model and can return a probability. EOS is present
on every turn; head truncation or rejection follows the vocabulary's fixed
training policy.

## Metrics and denominators

`evaluate` scores one compact observation catalog per conversation, gathers
its eligible causal endpoints, and reuses the audited lexical workflow's
decision/loss metric definitions without constructing bag-of-words features.
It does not tune a threshold on evaluation data.

The returned `metrics` includes conversation-weighted prefix Brier loss, log
loss and ROC AUC. Each conversation has total prefix weight one. The any-alert
confusion matrix takes the maximum eligible score per conversation and reports
TP/FP/TN/FN, precision/recall, false-positive rate, accuracy, balanced accuracy
and F1. These are not per-message confusion counts. For positive conversations
with alerts, first-alert lead time is the earliest eligible alert's distance
in non-header turns to the first event; its denominator excludes missed events.
The report includes this denominator as `true_positive_first_alerts`.

`support` retains the full `SequenceAudit`, including admitted source counts,
headers, exclusions, eligible class counts and compact token/turn counts.
The report also records model and partition digests, threshold-selection reuse,
`probability_calibration_claimed=False`, and
`whole_repository_parity_claimed=False`. A training loss improvement or successful
synthetic test does not turn either flag into an empirical quality result.

For an independent hand check, a constant zero-logit model produces probability
0.5 at every eligible endpoint. With both policy classes present, balanced
accuracy ties at 0.5 and the conservative higher threshold is 1.0. On any
evaluation class mixture its weighted Brier loss is 0.25 and log loss is
`log(2)`; with both classes ROC AUC is 0.5. All conversations receive no alert.
This is a controlled test fixture, not a learned baseline performance claim.

## Frozen state and failure behavior

`model.state` is an immutable fitted snapshot holding config, policy,
vocabulary, selected frozen parameters, training evidence, threshold and split
identities. Its model digest is computed when the snapshot is built, not by
rehashing every parameter on every prediction. `training_summary` returns a
fresh JSON-shaped summary; mutating that returned dictionary does not mutate
the model. Parameter arrays are backed by immutable bytes.

Fitting builds and validates a private candidate. Only after training,
policy inference, threshold selection and snapshot construction succeed does
it replace the old state. Any exception before that point preserves the exact
previous state, digest and predictions. A first failed fit remains unfitted.
The object does not promise concurrent/reentrant fit coordination or durable
resumption. Direct manipulation of private fields is not a supported state
update API.

Frozen prediction uses NumPy and does not import or require Torch. The
optional Torch dependency is needed only for actual fitting. `model.save(path)`
publishes a validated inference artifact, exclusively by default;
`HierarchicalEventForecaster.load(path)` restores it without Torch. Explicit
`overwrite=True` permits replacement. See the [artifact contract](neural-forecast-artifact.md)
for limits, integrity checks, private vocabulary/weights, cleanup warnings and
why an internally consistent model file is not authenticated training evidence.

## Admission limits

`NeuralForecastConfig` declares architecture, pinned per-turn ordinary-token
policy and an aggregate inference affine-work ceiling. The token cap plus EOS
must fit `NeuralNumericLimits.max_turn_tokens`. The minimum observations must
fit the data-layer turn limit. All options use typed, finite, bounded
contracts rather than silently coercing booleans or malformed values.

Policy-validation input is admitted before expensive training. Single
prediction checks its configured affine cap before numerical inference;
transformation and evaluation sum work over their complete catalogs before
performing any inference. Transformation also checks aggregate conversation,
source-turn, UTF-8 text/identity byte and raw token budgets, not only each
individual prefix. Exceeding a budget raises an error and produces no partial
prediction collection. Encoding and validation still consume bounded work
before such rejection; this is not a zero-cost operation guarantee.

Numerical limits bound declared parameter/input shapes and estimated work,
not wall-clock latency, native allocator RSS or process memory. Neither a low
threshold nor a small expected number of positive alerts reduces work quotas.

## Scope of verification

Authored integration tests exercise actual tiny CPU fit → predict/transform →
held-out evaluation, exact threshold selection against a separate rational
arithmetic oracle, independently hand-expected constant-logit metrics, explicit
two/three-partition policies, excluded-group leakage rejection, cached immutable
identity, atomic failed refits, metadata/future-feature separation and complete
collection admission before inference. An actual fitted tiny candidate is
transferred as an explicitly test-only JSON fixture to a fresh subprocess
which forbids importing Torch and verifies identical frozen predictions.

These tests are engineering evidence, not an authentic CGA quality evaluation.
They do not establish an untouched final-test result, calibrated uncertainty,
fairness, deployment safety or whole-project parity. Compare against the
existing lexical forecast and prior baselines on a separately declared real
source protocol before making performance claims.
