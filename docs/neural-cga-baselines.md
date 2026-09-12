# Input-matched lexical and prior baselines

This experimental helper is on the unreleased development branch. It is not a
pretrained model or a claim of forecast quality. This protocol is specified before
real CGA training/evaluation outcomes; the helper performs no downloads or model
calls. Importing it does not load data or run an experiment.

`benchmarks/neural_cga_baselines.py` accepts only immutable, already-prepared
`SequenceForecastDataset` objects. The experiment coordinator must compare their
digests with the neural training, policy-validation and heldout partitions.
Preparation retains eligible causal observations and includes excluded-source
conversation/page/pair groups in overlap checks. The baseline cannot reconstruct
missing raw future records or prove a dishonest upstream dataset's provenance.

## Fixed protocol

- Fit `SequenceVocabulary` internally on eligible **training observations only**,
  never repeated prefix copies: minimum conversation document frequency 1, maximum
  10,000 lexical features, existing Unicode-version-pinned regex/casefold policy.
  No caller-provided vocabulary, frequency table or neural threshold is accepted.
- Apply the existing `128/head` encoder: at most 128 lexical IDs per observed
  turn **plus one EOS**, hence at most 129 IDs. EOS is an explicit observed-turn
  boundary feature; it is not a censored/future turn. UNK has its own category.
  PAD is never present and is excluded from the multinomial smoothing support.
- Use cumulative multinomial naive Bayes with alpha 1. Each eligible conversation
  has total training weight 1, divided equally among its eligible prefixes.
  Its prior is positive eligible training conversations divided by all eligible
  training conversations. A separate prior-only comparator uses that constant.
- Select **two independent thresholds** from policy-validation conversation-max
  probabilities using the existing exact balanced-accuracy sweep. Prediction is
  `probability >= threshold`; exact ties choose the highest threshold among
  `{0, 1, observed policy conversation maxima}`. Never reuse the neural threshold.
- No epoch selection or neural model-validation input is used. There is no test
  tuning. Fitted vocabulary, counts, likelihoods, prior and both thresholds are
  immutable before heldout evaluation. Training/policy overlap is rejected before
  fitting; heldout overlap with either is rejected, including excluded groups.

This matches preparation, token exposure, training support/weights and policy
partition, **not model capacity**. Naive Bayes ignores token order and uses EOS
only as a count. Neural validation for early stopping remains unseen by the
baseline. A shared hash does not prove causal preparation by itself.

## Count and probability definitions

For conversation c with m eligible endpoints, let x(c,j,k) count encoded category
k in the cumulative observations at endpoint j. The class sufficient statistic
is `N[y,k] = sum(c: label=y) sum(j) x(c,j,k)/m`. Each category begins at token ID 1:
UNK, EOS, then fitted words. PAD ID 0 is not in the parameter arrays.

`P(k|y) = (N[y,k] + 1) / (sum(k) N[y,k] + K)`, where K excludes PAD.
Prefix log odds are `log(prior/(1-prior)) + sum(k) x[k]*log(P(k|1)/P(k|0))`.
The sign-stable sigmoid is clipped to `[1e-15, 1-1e-15]`, matching the neural
probability floor. Clipping prevents nonfinite losses; it is not calibration.

Each unique observation is encoded once per fitting/evaluation partition. A token
at turn t contributes `(number of eligible endpoints >= t)/m` to its training
class; this equals the average cumulative-prefix count without copying every
prefix. Inference updates cumulative log odds once per turn and emits only at
the prepared endpoints. Feature encoding accepts observations/vocabulary only,
not labels, roles, page metadata, attack text or raw future records.

## API, evidence and bounds

`fit_neural_baselines(training, policy_validation, *, data_limits=None, limits=None)`
returns a frozen `FittedNeuralBaselines` with `.state`, `.summary()` and
`.evaluate(heldout)`. The state exposes immutable vocabulary/count/likelihood
tuples for independent mathematical checks, but reports omit token strings,
count arrays, raw text, source IDs, groups and per-prefix predictions.
Summary dictionaries are fresh copies and cannot mutate the model.

Evaluation returns `lexical_nb` and `prior` metric dictionaries, support including
excluded conversations, encoding/truncation/UNK/EOS counts, partition/observation/
prediction/state digests and explicit no-tuning flags. Existing metrics provide
conversation-weighted prefix Brier/log loss/AUC, conversation any-alert confusion
counts/precision/recall/F1/balanced accuracy, and first-alert lead turns. Both
training and policy partitions require both classes; single-class heldout sets
are allowed with undefined AUC/BA, but empty eligible evaluation is rejected.
No synthetic test result is presented as real CGA quality.

`SequenceLimits` source/admission bounds still apply. Additional default per-
partition bounds are two million encoded IDs including EOS, two million distinct
category-per-turn updates, and 100,000 fitted count/likelihood scalar cells
(`4*K`). Counts/likelihoods use O(K) memory; only one observation's encoding and
small per-prefix scores/metric handles are retained, not all raw prefix copies.
Limits are strict positive integers, not booleans. Failures raise explicitly;
there is no sampling, silent truncation beyond the declared 128/head policy or
automatic smaller-model fallback. These work counts are not RSS/time guarantees.
Dataset and fitted-state digests are checked around execution for ordinary
accidental mutation, not malicious concurrent Python memory modification.
