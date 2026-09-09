# First-future-event prefix forecasting

`PrefixEventForecaster` predicts whether a first annotated event will occur later
within an observed conversation record. It is a **conversation-weighted cumulative
multinomial Naive Bayes baseline**, not a neural forecaster or a reproduction of
another toolkit's model. Training, validation-only threshold selection, future
label construction, group separation, model persistence, and final evaluation
are explicit parts of this workflow. It needs only the Python standard library.

## Supervised record contract

Every non-header utterance needs a boolean `metadata["event"]`. In the accompanying
CGA-WIKI experiment this is the supplied human personal-attack annotation, not a
toxicity model's score or an invented label. The configurable boolean
`skip_field="is_section_header"` excludes non-observation headers; missing means
false. The event and skip fields cannot be the same. Every conversation needs a
nonempty list of distinct strings in `metadata["forecast_groups"]`, for example
`["page:123", "pair:conversation-a"]`.

`prepare_forecast_examples` and `model.prepare` build immutable
`ForecastPrefix`/`ForecastExample` snapshots according to these rules:

- Input order must have nondecreasing timestamps; duplicate conversation or
  within-conversation utterance IDs are errors. The generic API never silently
  reorders messages. The CGA loader explicitly sorts by timestamp then ID.
- A prefix contains at least `min_turns=2` non-header observations. It ends before
  the first event; the event itself and all later text are excluded. Conversations
  whose first event occurs too early have no eligible prefix and are reported as
  excluded, not relabeled negative.
- Equal-timestamp blocks are atomic. No boundary splits one or describes an
  event at the same timestamp as occurring in the future.
- A negative conversation supplies prefixes only before its last observed
  time block, so the prefix has at least one strictly later observation. The label
  means **no event in this recorded future**, not that the conversation could
  never produce an event after collection ended. Horizons vary by record.
- `ForecastPrefix` contains IDs, observed turn count/time, and copied read-only
  lexical counts. It has no target annotations, future text, full-record length,
  outcome metadata, or group identities. Supervision and lead turns exist only
  in the separate `ForecastExample` wrapper.

`predict(observed_conversation)` treats the supplied record as the currently
observed prefix. It applies the same explicit boolean header mask, validates IDs
and timestamps, and ignores event/outcome labels and all other metadata. Supply
only text already observed; passing a complete future conversation is not a valid
forecast. If an event has already occurred, the first-future-event task no longer
applies; the supervised extractor enforces this, but unlabeled deployment input
must be managed by the caller. No deployment action or automatic moderation is
performed by this package.

## Train, validation, test are different roles

```python
from turnscope import PrefixEventForecaster
from turnscope.io import load_path

model = PrefixEventForecaster(alpha=1.0, max_features=10_000)
model.fit(load_path("train.jsonl"), load_path("validation.jsonl"))
model.save("forecaster.json")

frozen = PrefixEventForecaster.load("forecaster.json")
report = frozen.evaluate(load_path("untouched-test.jsonl"))
```

Vocabulary and likelihoods use only training prefixes. All eligible prefixes from
one training conversation have total sample weight one: a conversation with four
prefixes contributes weight `1/4` per prefix. Vocabulary document frequency counts
each training conversation once across its eligible observations, with decreasing
frequency then lexical order breaking ties. Tokenization uses Unicode word runs,
internal apostrophes/hyphens, and casefolding, as in the other lexical models.

For class `c`, weighted token counts are `n[c,t]`; the fitted likelihood is
`P(t|c) = (n[c,t] + alpha) / (sum_t n[c,t] + alpha * vocabulary_size)`.
Class priors count eligible training conversations, not prefixes. The posterior
log odds are the training prior log odds plus cumulative token counts times
`log P(t|positive) - log P(t|negative)`. Unknown words do not contribute; an empty
or all-unknown vocabulary produces the training-prior prediction. Scores are
clipped to `[1e-15, 1-1e-15]` for finite binary64 losses. NB probabilities can be
overconfident: they are **not claimed to be calibrated risk estimates**.

Validation does not refit vocabulary, likelihoods, or priors. It selects a threshold
using each conversation's maximum eligible prefix score, maximizing conversation
balanced accuracy for `score >= threshold`. Candidate thresholds are the unique
validation maxima plus 0 and 1. Mathematically tied policies choose the larger
threshold, using an exact integer objective before converting the score to float.
Both train and validation require eligible conversations from both classes.

Conversation IDs and every declared group receive separate stable SHA-256
fingerprints. `fit` rejects any shared identity between training and validation.
`evaluate` rejects groups already seen in either fitting partition, including
groups attached to excluded conversations. Page, matched pair, speaker, thread,
or other dependencies must be declared by the producer; the generic API cannot
discover hidden group relationships. The CGA loader verifies reciprocal pairs,
page IDs, and official splits independently before fitting. Hashes detect identity
overlap; they do not anonymize publicly guessable identifiers.

A failed fit leaves existing state unchanged. Prediction and evaluation cannot
update fitted state. Test results must not guide threshold or hyperparameter
selection; after using a set for tuning, designate it validation data and find an
untouched test set.

## Metrics and CLI

```bash
turnscope forecast fit train.jsonl validation.jsonl forecaster.json
turnscope forecast predict forecaster.json observed-prefixes.jsonl -o scores.json
turnscope forecast evaluate forecaster.json untouched-test.jsonl -o evaluation.json
```

`fit` accepts `--event-field`, `--skip-field`, `--groups-field`, `--min-turns`,
`--alpha`, feature limits, and workload budgets. Input/model/output aliases are
refused. Reports are validated and materialized before writing; `predict` emits
only IDs, scores, decisions, and lexical coverage, not message text. Large reports
should use the Python API one conversation at a time rather than batch CLI JSON.

Evaluation reports two distinct units:

- Prefix Brier score, clipped log loss, and tie-aware ROC-AUC, with each
  conversation's eligible prefixes sharing weight one. AUC is null for a
  single-class evaluation, not a fabricated zero.
- Conversation-level any-alert confusion counts, accuracy, precision, recall,
  false-positive rate, F1, and balanced accuracy using the validation threshold.
  Undefined denominators return null. Positive first alerts report mean lead
  turns, counting the event turn (an immediately preceding alert has lead 1).

The same metrics are emitted for the fixed eligible-training class-prior baseline;
its score is never estimated from validation/test labels. Longer conversations
still have more opportunities to trigger an any-alert decision. Prefix weighting
does not remove that property, so report both units rather than conflating them.
These are observational annotation outcomes, not proof of causality, fair
moderation, population calibration, or intervention effectiveness.

## Artifacts, cost, and limitations

Versioned, closed-schema JSON stores numeric parameters, configuration, group
fingerprints, threshold, and support counts. Loading rejects malformed/duplicate
JSON keys, nonfinite values, impossible class priors, mismatched matrix widths,
unnormalized likelihoods, overlapping groups, and likelihood magnitudes outside
the fitted token-budget bounds. A checksum detects accidental corruption, not
authenticity or dishonest model producers. Saves are atomic, and both saving and
loading enforce the same 32 MiB byte budget. Vocabulary and learned parameters
may reveal information about training text; do not publish them casually.

Defaults bound each prepared partition to 30,000 eligible prefixes, 4,000,000
cumulative sparse count cells, and 2,000,000 token occurrences processed before
eligible cutoff points. Configuration maxima are 100,000 prefixes, 16,000,000
cells, 10,000,000 tokens, and 100,000 features; preparation also caps 100,000
conversations and 400,000 group fingerprints. These are workload limits, **not
RSS guarantees**. Parsing a large record and tokenizing one long message can
allocate memory before a count limit is checked. Fit retains train and validation
snapshots concurrently, and raw unique vocabulary is counted before pruning.

Let `T` be processed tokens, `S` all snapshot nonzero count cells, `V` raw candidate
terms, and `C_val` eligible validation conversations. Preparation uses `O(T+S)`
work/storage, fitting and scoring `O(S + V log V)`, and threshold selection
`O(C_val log C_val)` after prefix scores. A single prediction is linear in its
observed tokens plus distinct known terms. This is a bounded lexical baseline,
not the full neural forecasting, generative simulation, or decision-policy
ecosystem of a mature conversational research toolkit.

The [CGA-WIKI benchmark](forecast-benchmark.md) provides authentic human-event
heldout evidence with explicit group checks and no raw-data redistribution.
