# Explicit three-partition ablation forecasting

`turnscope.neural_ablation.AblationEventForecaster` connects the separately
trained controls to frozen prediction, threshold selection and heldout metrics.
It does not change `HierarchicalEventForecaster` or interpret a main-model
checkpoint as another architecture. A mode must be named explicitly.

| Mode | Representation |
| --- | --- |
| `current-turn.v1` | Completed current message's bidirectional word GRU, independent zero-state turn gate |
| `mean-word.v1` | EOS-inclusive mean word embedding duplicated to 128 dimensions, causal turn GRU |
| `order-erased.v1` | EOS-inclusive mean per turn, mean of completed turns within each prefix |

All three use the versioned 64-wide, one-layer reference architecture. They are
separately trained controls, not post-hoc switches on a trained main model. See
[numerical contracts](neural-ablation-math.md), [training](neural-ablation-training.md)
and the [frozen CGA experiment design](neural-cga-ablation-design.md) for equations,
resource estimates and experimental limits.

## Fit, predict and evaluate

```python
from turnscope.neural_ablation import AblationEventForecaster, AblationForecastConfig
from turnscope.neural_forecast_train import NeuralTrainingConfig

model = AblationEventForecaster(
    config=AblationForecastConfig(variant="mean-word.v1"),
    training_config=NeuralTrainingConfig(seed=17),
)
model.fit(
    training_conversations,
    model_validation_conversations,
    policy_validation=policy_validation_conversations,
)

# A deployment prefix contains only observations available now, never future turns.
prediction = model.predict(observed_prefix)
predictions = model.transform(many_observed_prefixes)
report = model.evaluate(heldout_conversations)
```

The code requires a checkout/distribution containing these modules and the
existing optional CPU training dependencies. It neither installs/downloads
dependencies nor calls a provider. Fit requires PyTorch and NumPy; frozen
prediction/evaluation uses NumPy, not Torch training. The numerical arrays are
stored as privately owned immutable float32 bytes and inferred in float64.

The module performs no filesystem/network I/O. This API slice covers in-memory
fit/selection/prediction/evaluation, not optimizer resume, an artifact format or
a command-line interface. Persistence must use a separately verified ablation
format; a main-model artifact cannot be silently relabelled. No save/load or CLI
capability is claimed by this document.

## Supervision is a declared input contract

Raw fitting/evaluation input is an iterable of `Conversation`. `SequencePolicy`
names the event, header-skip and group fields and declares `min_turns` (default
2). The shared preparation contract derives first-future-event examples,
excludes event/after-event/header text from features, respects complete equal-time
blocks, and does not create a terminal negative prefix. All source text,
including excluded/future text, still consumes source admission budgets. Group
pins include excluded conversations; absence of an eligible example does not
make a group safe to reuse.

`prepare(raw_conversations)` exposes this same preparation. Fitting and
evaluation also accept an already constructed `SequenceForecastDataset`.
Prepared input is **caller-declared supervision**: the model checks the closed
dataset type, structural/resource validity and that no example precedes its
configured `min_turns`, but does not have the removed raw outcome metadata needed
to rederive first-event times, group membership or event-field semantics. A valid
prepared object is not independent evidence that a source annotator or adapter
used the intended policy. Keep source/adapter/protocol provenance separately and
use the same declared policy in each partition. Do not use manual preparation to
smuggle future observations into the catalog.

Deployment `predict` requires an immutable `ObservedPrefix`, not a supervised
example or arbitrary dictionary. The prefix contains conversation identity and
ordered `ObservedTurn(id, timestamp, text)` values, but no labels, event markers,
groups, roles or lead times as model features. `predict_conversation` is a
convenience adapter that removes explicitly marked headers and validates the
observations; it intentionally does not inspect outcome/group metadata. Therefore
the caller must pass only the transcript available at prediction time, not a
complete historical conversation containing future text. This helper cannot
infer what was "future" from intentionally ignored labels.

Source identifiers/timestamps bind the input digest; token IDs, not those
identifiers or timestamps, drive the learned encoders. Changing ignored label
metadata cannot change a deployment probability. An empty text turn is permitted
and contributes EOS; a prefix shorter than `min_turns` is not. An empty
`transform([])` on a fitted model returns `()` without numerical inference.

## Three distinct fitting roles

`fit(training, validation, *, policy_validation)` requires the keyword third
partition. There is no implicit two-partition reuse mode:

1. Training fits the vocabulary and parameters. The trainer re-fits vocabulary
   metadata from training to verify frozen DF, selection, tokenizer/UCD and
   retention identity.
2. Model validation selects the earliest strictly best validation-loss epoch.
   It cannot update the vocabulary or parameters.
3. Policy validation chooses the decision threshold from frozen NumPy outputs.
   It does not select another epoch or update parameters.

All fitting group inventories must be pairwise disjoint, including pins from
excluded source conversations. Each fitting partition needs eligible examples
from both classes; policy validation is checked before optimization. Heldout
evaluation rejects overlap with **any** of the three fitting inventories.

Before using a returned candidate for policy inference, fit checks its exact
type, variant, architecture, training/numeric/pooling configuration, vocabulary
digest, both fitting partition digests and all conversation/prefix support counts
against this request. Internal validity alone would not prove that a result
belongs to the current inputs. These checks also catch future adapter mistakes;
they cannot authenticate an intentionally dishonest source or compromised code.

Fit creates a private candidate and commits one new immutable state only after
training, policy inference, threshold selection, validation and identity hashing
all succeed. Any exception preserves the previous state object, digest and
predictions. CPU time and temporary optimizer work already consumed are not
rolled back, and there is no automatic retry or use of partially trained state.

## Threshold and metric denominators

Each conversation contributes its maximum eligible-prefix probability to policy
selection. The threshold maximizes balanced accuracy and classifies with
`probability >= threshold`. Equal objectives choose the higher threshold using
exact integer comparisons, not floating-point tie approximations. The selection
score is reported as `policy_validation_balanced_accuracy`; it is **not** a heldout
performance estimate or probability calibration.

Evaluation keeps the frozen threshold and reports:

- Conversation-weighted prefix Brier score, clipped log loss, and tied ROC AUC.
  Each conversation has total weight one, divided among its eligible prefixes.
- Any-alert confusion/accuracy/precision/recall/F1/false-positive rate and balanced
  accuracy. A conversation alerts if any eligible prefix meets the threshold.
- True-positive first-alert lead time. Its denominator is only positive
  conversations that actually alerted; if none did, the mean is `None`.

Undefined class-specific metrics remain `None`; they are not fabricated zeros.
Empty evaluation is rejected rather than assigned a score. Prepared heldout
examples retain their declared labels and lead times; the numerical encoder never
uses those values as features. Probability clipping uses the pinned `1e-15`
floor. Explicit report flags retain `policy_reuses_model_validation=False`,
`probability_calibration_claimed=False` and `whole_repository_parity_claimed=False`.

## Budgets, immutable state and privacy

`AblationForecastConfig` fixes mode, ordinary-token cap (default 128), long-turn
policy (`head` or `reject`) and two independent aggregate inference caps:
100 billion affine multiplications and 100 billion logical pooling operations by
default, each with a hard maximum of one trillion. EOS must also fit the numeric
per-turn token limit. The shared data/numeric limits and explicit training/pooling
limits remain separately configurable, typed and bounded.

All policy observations are encoded and admitted before the first optimizer
step. Every `transform` collection and evaluation observation catalog is fully
admitted before the first numeric inference call, including a malformed late
item. Admission aggregates conversation count, observed turns, UTF-8 identity/text
bytes, raw tokens, affine work and pooling work; per-observation numerical bounds
also apply. Evaluation runs each maximum observation sequence once and gathers
eligible endpoints, rather than copying/recomputing all prefixes.

Each fitting source partition has its own data admission bounds, not a promise
that three partitions combined consume one partition's quota. Training performs
its own all-epoch schedule/workspace admission. Work estimates and source caps do
not guarantee process RSS or total native instructions. A quota error is a failed
operation, never an omitted sample, partial result, extra truncation or score zero.

`AblationForecastState` freezes configuration, vocabulary, strict training result,
threshold, three group inventories and policy support. It validates identities,
token policy, DF/support bounds, numeric types and distinct partition identities
before caching its digest. Repeated predictions use that cached state identity
rather than hashing all parameter bytes again. `training_summary` returns new
ordinary dictionaries/lists; editing them cannot change the state. Parameter
arrays cannot be made writable.

Predictions include the explicit variant, probability/logit/alert, threshold,
input/model digests, turn/token retention/known-token counts, work counters and
workspace estimate. Their dictionary does not include raw text. Empty ordinary
token content has retained fraction 1 and known-retained fraction 0; an OOV input
uses learned UNK/EOS behavior rather than a claim of certainty or no risk.

The accessible state contains vocabulary and weights, and group/input digests
can still be sensitive. Hashes are not authentication or anonymization. Keep raw
inputs, states and detailed outputs private unless rights/privacy have been
reviewed; an exception traceback or external logging policy can disclose more
than the aggregate report.

## Bounded acceptance evidence

`tests/test_neural_ablation.py` fits each mode on small original train/model-
validation/policy partitions and evaluates separate original heldout groups.
Independent `Fraction` threshold enumeration and direct pairwise weighted-AUC,
Brier/log-loss and first-alert calculations check exact denominators. Explicitly
labelled zero-weight, non-training fixtures test constant-score tie behavior and
fault paths; they are not substituted for real optimization evidence.

Tests cover required third partition, excluded-group leakage, candidate/request
binding, atomic failed refit, cached immutable state, observation-only metadata,
future-vocabulary exclusion, exact work caps and late-invalid input rejection
before inference. None of these small examples establishes CGA performance,
causal isolation, calibration, cross-domain robustness or whole-repository parity.
The frozen matched experiment, all declared seeds, distinct artifact roundtrips
and any broader quality claims remain separate acceptance work.
