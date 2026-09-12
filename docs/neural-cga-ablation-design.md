# Matched neural CGA ablations: frozen implementation specification

Status: design only, 2026-09-12. This specification was written before any
official-test scores from the new neural candidates were produced or inspected.
No model was fitted or scored for this document. It does not authorize a run or
claim that the variants exist yet. The official test cohort was already examined
by earlier lexical/context work; it is not a fresh confirmatory cohort.

The three separately trained controls below implement the candidate driver's
declared current-turn-only, mean-word, and order-erased comparisons. The fourth
experiment is a frozen-model, prefix-local intervention, **not** another trained
baseline. It must remain separately labelled in reports. No existing main-model
artifact may silently acquire any of these alternate forward meanings.

## 1. Evidence and shared experimental contract

The specification uses the current original implementation:

- `benchmarks/benchmark_neural_forecast.py::fit_protocol` and
  `validate_fit_summary`: three declared seeds, fixed settings, independent
  model/policy validation, source/actual-state binding, no final-test prediction
  in the candidate-fit stage.
- `benchmarks/neural_cga_data.py::protocol`, `NeuralCgaPartitions.prepared_audit`:
  pinned source, page/pair-connected validation subdivision before eligibility.
- `neural_forecast_data.py::prepare_sequence_forecasts` and
  `neural_token_data.py::fit_sequence_vocabulary`, `encode_observed_prefix`:
  compact eligible endpoints, training-only vocabulary and frozen token policy.
- `neural_forecast_math.py::parameter_shapes`, `_gru_step`,
  `infer_encoded_turns`: reset-after r/z/n arithmetic and the actual tensor shapes.
- `neural_forecast_train.py::make_torch_hierarchy`, `forward_encoded_batch`,
  `_batch_loss`, `train_sequence_model`: initialization, inverse packed ordering,
  conversation-weighted BCE, Adam and earliest-best checkpoint selection.

Planning snapshots: driver SHA-256
`29f9718434a8802b3139300c8355bd14e31c6e6f8b6219e364d57dcbcc1ae3c5`, math
`36ce76a6e06f1d35134ff1b4e618a8af7ab6419cc6edb561f6971a8b0ad6cb3e`, training
`f54ede8cb9375ae9b2863259146f3325995d61ecc33f9fdfdbc36630980bdfca`.
These identify the inspected definitions, not a claim that future implementation
or a complete distribution has already passed acceptance.

Freeze all of the following for the main model and each trained variant:

| Item | Fixed policy |
| --- | --- |
| Source | Existing official CGA-WIKI archive SHA-256 `84e2d1ac60a3269251b5e175e549fc65cec875fdc926f99172b5b70d3ca1b122` |
| Partitions | Official train/test; validation connected by shared page OR reciprocal pair, protocol seed `turnscope-neural-v1/2026-09-12`, hash modulo 5: buckets 0–1 policy, 2–4 epoch selection |
| Supervision | First future human-labelled event, no header/event/later-text features; `min_turns=2`, equal-time blocks atomic, no terminal-negative endpoint |
| Vocabulary | Fit once from eligible training observations, DF per conversation, minimum DF 1, at most 10,000 ordinary tokens; reuse and revalidate the exact vocabulary/UCD digest for every variant and seed |
| Retention | First 128 ordinary tokens per completed turn, then EOS; no extra history/token truncation, no sampling to fit a budget |
| IDs | PAD 0, UNK 1, EOS 2; none of the source identities, roles, groups, labels or future counts is an embedding feature |
| Dimensions | Embedding 64, reference word hidden 64 per direction, reference turn hidden 64, head hidden 32; one word and one turn layer where present |
| Seeds | Exactly 17, 101, 202 for initialization and canonical conversation shuffling; report every attempt, never select a seed by test results |
| Optimization | CPU float32 Adam, learning rate 0.001, betas `(0.9,0.999)`, epsilon `1e-8`, zero weight decay, non-fused/non-foreach, global gradient clip 5, dropout 0 |
| Batching | At most 16 conversations and 8,192 real encoded positions per batch, including EOS; identical conversation/position admission and seeded schedule policy for every mode, final partial batch included |
| Selection | At most 8 epochs; patience 3; earliest strictly smallest conversation-weighted model-validation BCE selects that variant's weights |
| Decision policy | Each variant/seed gets its own policy-validation threshold from its frozen NumPy outputs; conversation-maximum balanced accuracy, `>=`, exact higher-threshold tie break |
| Process policy | Benchmark-only CPU threads 1 and deterministic algorithms enabled; restore prior thread/deterministic/warn-only settings on success or failure |

The official validation split currently audits as 526 epoch-selection and 314
policy-selection source conversations. Implementation must check the actual
pinned adapter's partition digests and eligible support, not replace those
checks with these two source counts. Excluded conversations still contribute
group leakage pins. Every variant uses the same exact eligible handles and
labels; a current-only encoder does not create new single-turn examples.

No baseline gets a larger epoch budget, favorable learning rate, special class
weighting, wider head, alternate vocabulary, or per-seed split. Each has a
separate optimizer and checkpoint; do not reuse trained main-model embeddings.
Independent early stopping can produce different executed epochs. Preserve the
complete traces rather than force every model to use the main model's selected
epoch. The unchanged shared upper batch schedule is used until each stops.

For conversation c with eligible endpoints E_c, optimize

```text
L_c = (1 / |E_c|) sum_{t in E_c} [softplus(logit_ct) - y_c * logit_ct]
L_batch = mean_{c in batch} L_c
```

Validation uses the same conversation weighting. Reported minibatch training
loss is weighted by the batch's conversation count; Adam still acts on each
batch mean, including a smaller final batch. Do not describe sequential Adam
steps as one full-dataset gradient. Every numerical/optimization quota failure
is a failed run, never score zero or permission to drop a conversation.

## 2. Main equations and the three alternative forward functions

Let x_tj be retained token IDs for completed turn t, including its one EOS;
L_t is their nonzero count. E has shape `(V,64)`, where V includes all three
special rows. PAD never enters an encoded turn. Define the common classifier

```text
H(v) = w_out @ tanh(W_head @ v + b_head) + b_out
```

with shapes `(32,64)`, `(32,)`, `(1,32)`, `(1,)`. Stable sigmoid and the existing
saved probability floor `1e-15` supply deployment scores and metric clipping.

The main model forms u_t by concatenating the final forward/backward word-GRU
states, giving 128 dimensions. It then uses h_t = GRU(u_t,h_{t-1}), h_0 = 0,
and logit_t = H(h_t). Its reset-after convention is:

```text
r = sigmoid(W_ir u + b_ir + W_hr h + b_hr)
z = sigmoid(W_iz u + b_iz + W_hz h + b_hz)
n = tanh(W_in u + b_in + r * (W_hn h + b_hn))
GRU(u,h) = (1-z)*n + z*h
```

### A. Separately trained current-turn-only encoder

Keep the same trainable bidirectional word encoder and classifier, but reset
the turn state to zero **for each completed turn**:

```text
r_t = sigmoid(W_ir u_t + b_ir + b_hr)
z_t = sigmoid(W_iz u_t + b_iz + b_hz)
n_t = tanh(W_in u_t + b_in + r_t * b_hn)
v_t = (1-z_t) * n_t
logit_t = H(v_t)
```

There is no dependency on u_1 through u_{t-1}. This is exactly a one-layer
reset-after GRU evaluated with h=0, not an arbitrary new MLP. In particular,
**the recurrent biases remain relevant**: `b_hn` is gated by r_t. Do not delete
all recurrent-affine terms just because the previous state is zero.

Conversely, every element of `turn.weight_hh_l0` multiplies zero and is
structurally inactive. Omit this matrix from the variant's optimizer and saved
inventory rather than keeping an unused parameter and calling it trained.
Retain the input matrix and both bias vectors as active trainable parameters.
A small explicit one-step module can implement the equations without allocating
a dummy recurrent matrix. Word-level recurrent matrices are still present and
can be active within a completed message. Earlier ineligible turn outputs have
no direct loss; this is distinct from claiming every tensor element gets a
nonzero gradient in every batch.

### B. Separately trained mean-word replacement

Remove both word GRUs. Use a trainable embedding mean **including EOS**:

```text
m_t = (1 / L_t) sum_{j=1..L_t} E[x_tj]       # 64 dimensions
u_t = concat(m_t, m_t)                       # 128, not a learned projection
h_t = GRU(u_t, h_{t-1}); h_0 = 0
logit_t = H(h_t)
```

Duplicating the mean preserves the reference turn-GRU input width and the
classifier architecture. It does not recover two independently learned word
directions: the two halves are identical and corresponding input columns are
redundant. This capacity/conditioning change must be disclosed, not called a
perfect intervention on word order alone. Both halves' input weights may be
optimized; their redundant effects are not structurally zero.

An empty ordinary-token turn is `(EOS,)`, so m_t is E[EOS]. UNK is included
where encoding supplies it; PAD and padding are excluded. There is no division
by raw pre-truncation length, and no pooling of a whole conversation before
constructing early predictions. Token order is erased within the retained bag;
history order remains in the causal turn GRU.

### C. Separately trained fully order-erased encoder

Remove both recurrent levels. First use the same per-turn embedding mean as
variant B, then mean only the completed turns in the current prefix:

```text
s_0 = 0
s_t = s_{t-1} + m_t
v_t = s_t / t
logit_t = H(v_t)
```

No word-GRU, turn-GRU, auxiliary projection, count feature, latest-turn vector,
position embedding or source identity is retained. The active tensors are
only embedding and the common classifier. This is mathematically invariant
to permutations of retained tokens within a turn and to permutations of the
completed turns within a specified prefix. Float32/float64 summation order can
cause rounding differences; validate invariance with declared numerical
tolerances, not a claim of bitwise permutation invariance.

This is a **mean of turn means**, not a token-count-weighted global mean or
multinomial NB. Every turn receives equal representational weight irrespective
of its retained length. Repeating every identical turn leaves the mean
unchanged; a count-only predictor can behave differently. Those are deliberate
baseline properties and additional confounders in an architectural comparison.

For B and C, retention happens before pooling. Reordering an arbitrarily long
raw message can change which 128 tokens are retained; invariance is asserted
only when the retained multiset itself is unchanged. Never sort all raw tokens
before the head cap, because that silently changes the input policy.

## 3. Exact active inventories and initialization matching

Keep the main one-layer parameter names for retained arrays. A mode-specific
closed shape function must reject all missing or extra arrays.

| Mode | Active saved arrays | Stored scalar parameters, including the fixed PAD row |
| --- | --- | ---: |
| Main | embedding; 4 word arrays × 2 directions; 4 turn arrays; 4 classifier arrays | `64V + 89,281` |
| Current-only | main minus `turn.weight_hh_l0` | `64V + 76,993` |
| Mean-word | embedding; `turn.weight_ih_l0`, `turn.weight_hh_l0`, `turn.bias_ih_l0`, `turn.bias_hh_l0`; classifier | `64V + 39,361` |
| Fully order-erased | `embedding.weight`, `head.weight`, `head.bias`, `output.weight`, `output.bias` | `64V + 2,113` |

For clarity, each word direction has `(192,64)` input and recurrent matrices
and two `(192,)` biases. The full turn layer has `(192,128)` input,
`(192,64)` recurrent and two `(192,)` biases. Current-only retains the first
and both biases. The four classifier arrays total 2,113 scalars. At the maximum
10,003-row vocabulary the totals are 729,473 / 717,185 / 679,553 / 642,305.
PAD contributes 64 stored constants, is zero, is excluded from inputs and stays
unchanged. Report both stored totals and this fixed-row qualification. Other
rows may receive no gradient if absent from the selected training observations;
that is data-dependent inactivity, not grounds to claim they were updated.

Use the existing `INITIALIZATION_VERSION` and the main canonical draw order for
all seeds. Generate the original main initializer's **untrained** named tensors
once, select the mode's fixed subset, and discard excluded tensors before
creating optimizer state. This preserves exact common initial tensor bytes in
the same supported runtime. Record `canonical-main-subset-init.v1` plus initial
member hashes. Do not simply initialize a shorter inventory in order: omitting
draws would change later shared initial weights. Do not initialize from a fitted
main artifact or carry optimizer moments between variants.

The temporary full initial inventory is intentional and must be admitted even
for the small order-erased model. An implementation may refactor the existing
initializer into an explicit shared utility; it may not monkeypatch a global
forward function, alter main-model methods, or claim the absent matrices remain
trainable. This first experiment supports only the stated one-layer 64-wide
reference architecture; other dimensions/layers must reject until a separately
specified shape/equation version exists.

## 4. Causal batching and work accounting

Keep compact conversation catalogs and endpoint handles. Use the same local
seeded conversation ordering, encoded position counts, and admitted batch
schedule policy as the main trainer. The maximum eligible observation catalog,
not an expanded list of copied prefixes, supplies vocabulary and batch lengths.

- Current-only: encode each admitted completed turn once with the word GRU,
  invert word-length sorting, apply the independent zero-state gate to each
  turn vector, then gather eligible endpoints. Do not concatenate conversations
  into a recurrent stream. Earlier word encodings cannot influence later gates.
- Mean-word: mask valid token positions before summing, divide by that turn's
  encoded length, restore conversation/turn positions, then pack the causal
  turn GRU. Never average padded rows or divide by padded maximum length.
- Order-erased: compute each m_t once and use a prefix cumulative sum within
  each conversation. Gather `(s_t/t)` only at eligible endpoints. Denominators
  reset per conversation. There is no lookahead and no need to sort or expand
  complete conversations into multiple prefix copies.

Computing a batched convolution/pool over all available turns is not permission
to average them all for every earlier endpoint. The state at endpoint t may
depend only on turns 1..t, even though later eligible observations occupy the
same batch tensor. Do not cache trainable embedding/word outputs across optimizer
steps; recompute with the current parameters. Caches during frozen inference
have different semantics and must be bound to a fixed parameter/vocabulary ID.

With T real encoded positions, U unique admitted turns, d=h=g=64 and q=32,
the declared one-layer forward affine multiplication counts are:

```text
main:          6h(d+h)T + 3g(2h+g)U + q(g+1)U
current-only:  6h(d+h)T + 3g(2h)U   + q(g+1)U
mean-word:                  3g(2d+g)U + q(g+1)U
order-erased:                             q(d+1)U
```

Mean pooling additionally costs O(Td) additions and O(Ud) scalings. Prefix
averaging adds O(Ud) additions/scalings. Report these separately from affine
multiplications; the latter is not a total FLOP/instruction count. The all-epoch
training estimate remains `epochs * (3*train_forward + val_forward)` for the
variant, with pooling work separately admitted. It is not measured backward work.

Keep the same absolute source, token, parameter, per-inference, aggregate work
and memory caps as the candidate plan; do not spend the smaller model's savings
on extra epochs or wider layers. Derive padding-aware activation estimates for
the actual variant. Include its parameters, gradients, Adam moments, selected
copies and padding buffers, and also the temporary canonical initializer.
For example, reserve at least `12*P_main` bytes for simultaneous float32 source,
snapshot and projected initialization tensors, in addition to applicable native
overhead, and use the maximum of initialization and training-phase estimates.
The existing `48*P_active` training parameter allowance is a useful starting
term, not the entire estimate. Explicitly document what Python/allocator/native
workspace overhead is excluded; admission does not guarantee process RSS.

## 5. Frozen inference and artifact/API separation

Proposed implementation boundaries, not APIs already shipped:

```text
AblationVariant = current-turn.v1 | mean-word.v1 | order-erased.v1
ablation_parameter_shapes(variant, reference_architecture) -> closed name/shape map
FrozenAblationParameters.from_arrays(variant, reference_architecture, arrays, limits=...)
infer_ablation_turns(parameters, encoded_turns, limits=...) -> logits/probabilities/work
train_ablation_model(training, validation, vocabulary, variant=..., settings=...)
AblationEventForecaster.fit(training, validation, policy_validation=...)
AblationEventForecaster.predict / evaluate / save / load
```

Use explicit mode-specific modules or a closed internal dispatch argument, not
arbitrary callables read from artifacts. Share pure admission, vocabulary,
eligibility, threshold and metric utilities where appropriate. A shared trainer
can accept a trusted closed numerical mode internally, but its result type must
identify the variant and actual tensor inventory; it must not pretend to be a
`NeuralTrainingResult` with incompatible `FrozenNeuralParameters`.

Frozen NumPy implements the equations above using privately owned immutable
float32 bytes and float64 arithmetic, with finite/magnitude/shape checks and
stable sigmoid. Current-only omits recurrent matrix products; mean-word and
order-erased explicitly mask/count valid positions. Public inference does not
import Torch, change a caller's process settings, or execute artifact code.

Define a distinct format `turnscope.neural-ablation-artifact.v1`. Its closed
manifest binds the variant and one of these exact numerical identities:

- `current-turn.reset-after-zero-rzn.v1`
- `mean-word-eos-duplicate.reset-after-rzn.v1`
- `mean-word-eos.mean-prefix.v1`

Also bind the fixed reference architecture, derived active inventory, actual
initialization policy, vocabulary/UCD, retention/eligibility, all configuration
and resource limits, training/selection/policy partition and group digests,
actual support, history, initial/changed parameter hashes, selected epoch,
threshold, clipping, runtime/source/protocol hashes and private-data notice.
There are respectively 16, 9 or 5 numeric arrays for the three modes. Unknown
modes, inactive extra matrices, cross-mode inventories or altered mode strings
must reject, even after checksums are recomputed.

The existing `turnscope.neural-forecast-artifact.v1` loader remains unchanged and
rejects this new format. The ablation-only loader rejects a main artifact rather
than guessing a mode. There is no automatic relabelling/conversion of trained
main weights. A separate format is needed even for current-only, whose retained
arrays look familiar but whose recurrence is different.

Reuse the strict storage policy through explicitly shared helpers if refactoring
is warranted: bounded regular-file reads; canonical closed JSON with duplicate,
finite, depth and queued-node limits; stored ZIP only; closed bounded NPY headers;
little-endian C-order float32 only; no pickle, arbitrary code, optimizer state,
raw conversations or extraction-to-filesystem. Keep the 64 MiB bundle and
4 MiB manifest caps, full before-allocation inventory checking, exact hashes,
exclusive atomic publication and explicit post-publication cleanup warnings.
Validate training history and actual source support as for the main artifact.
Hashes are integrity checks, not provenance authentication or anonymization.

Fit each candidate privately, validate its plan/support, save/reload, and choose
its threshold through the frozen NumPy implementation on policy validation.
Record the exact ordering if threshold selection precedes final save, as in
the current main pipeline. No official-test prediction belongs in this fit stage.
Do not add resumable-optimizer or cross-machine bitwise reproducibility claims.

## 6. Prefix-safe frozen history-order intervention

This diagnostic uses each frozen **main** seed's learned parameters, vocabulary,
retention policy and already-selected threshold. It does not refit weights,
recalibrate on transformed validation/test data, or define a new deployment
artifact. It asks how that fixed scorer responds to a declared ordering change.

Freeze one label-independent ordering protocol across all three model seeds:

```text
format = turnscope.prefix-history-permutation.v1
seed = turnscope-history-order/2026-09-12
```

For each original eligible prefix ending at original turn t:

1. Start with only its retained encoded turns 1..t. Keep current turn t last.
   This preserves the newest-message content and gives a direct invariant
   control for the current-only model.
2. For each historical turn i<t, hash UTF-8 compact sorted-key JSON containing
   exactly `{format, seed, conversation_id, endpoint_turn_id, historical_turn_id}`.
   Sort the historical turns by `(hash bytes, original position)` and append t.
   Hash ties use the original position; no labels, source split, future length,
   later IDs/text, probabilities or model seed are inputs. Keep the one fixed
   permutation protocol rather than choosing among permutations by outcomes.
3. Run the reordered prefix from zero turn state and retain **only its final
   logit** as the diagnostic score for the original endpoint. Never use the
   intervened internal-position logits as new chronological observations.
4. Attach scores to the original immutable eligible handles and lead times;
   apply the main model's original threshold and existing metric denominators.
   Hash both original-prefix and transformed encoded inputs and the ordering
   protocol for private reproducibility. Do not fabricate an `ObservedPrefix`
   with false chronological timestamps or rewrite original conversation records.

For t=2 only one historical turn exists and the intervention is a no-op. Some
longer prefixes also happen to retain their order or have identical turn content.
Report total eligible prefixes, prefixes with at least two historical turns,
index-order changes and actual encoded-sequence changes separately. Keep all
eligible examples in primary metrics; a changed-only diagnostic has its own
explicit denominator and does not replace the primary cohort.

**Do not permute the maximum conversation once and reuse all its earlier
outputs.** For original `[a,b,c]`, reordering to `[c,a,b]` would make an output
ostensibly at original `[a,b]` depend on c. Even keeping the final turn of the
longest prefix fixed does not protect every earlier prefix. Ordering is local
to each admitted endpoint, so changes to turns after that endpoint cannot
alter either its permutation or score.

For efficient frozen scoring, compute each unique completed-turn word vector
once under immutable main parameters, then run a fresh turn recurrence for each
prefix-specific ordering. This requires an explicitly source/model/vocabulary-
bound private word-vector cache or pure decomposition helper; do not feed
unchecked caller vectors into a public scoring API or cache trainable vectors
between optimizer steps. A straightforward fully recomputed prefix path is a
useful independent oracle, not the complexity claimed for the cached path.

If endpoint lengths are t_k, cached forward work is one word-encoder pass plus
`sum_k(t_k)` turn steps and K final classifier calls. Thus turn work can be
O(U²) per conversation, not the main model's one O(U) pass. Pre-admit this
aggregate work and cache bytes before numerical scoring; keep at most one
conversation's vector cache if needed. Report actual charged counters. A budget
failure must not skip long prefixes or silently replace their shuffled score
with the original. The same declared absolute cap applies; no cap increase is
inferred from a disappointing result.

The intervention preserves each prefix's token/turn content but may destroy
chronological coherence. It is an out-of-distribution sensitivity diagnostic,
not a causal effect of turn order in human conversation. It is not the trained
fully order-erased baseline, and its result cannot substitute for that baseline.

## 7. Required validation and reporting before quality acceptance

Implementation gates, not results already obtained:

1. Hand scalar zero-state current-gate arithmetic, including nonzero `b_hn` and
   unequal reset gates; compare with a full GRU at h=0. Prove the omitted hh
   matrix has zero derivative and retained biases do not disappear.
2. Hand mean/EOS/UNK/empty-turn arithmetic, repeated tokens, variable lengths,
   masking, the explicit duplicated 128-vector, and mean-of-turn-means versus
   global-token-mean distinction. Mean-prefix scalar gradient coefficients are
   `1/(t*L_i)` for a token occurrence in turn i<=t before subsequent head factors.
3. Independent serial Torch versus packed/vectorized forward and all active
   parameter gradients; finite differences of separately written NumPy equations
   on tiny shapes at the primitive level. Do not test only parameter updates.
4. Common initial tensor hashes match the canonical main initializer for each
   declared seed; omitted arrays are absent, not optimizer entries with zero
   updates. PAD stays zero. Test every active block can influence a nondegenerate
   authored loss; do not require every vocabulary row to change on sparse data.
5. Current-only is invariant to earlier-turn changes but responds to a changed
   current turn. Mean-word is invariant to retained within-turn permutations
   but can respond to history order. Order-erased is invariant to both within-
   turn and within-prefix turn permutations up to explicit rounding tolerance.
   A same-bag authored order rule is learnable by the full hierarchy but is
   indistinguishable to the defined order-erased input; preserve any failed
   fixed-protocol attempt rather than search seeds.
6. Compare all eligible earlier outputs before/after changing later observations;
   test independent prefix encoding against all-endpoint batching. Include
   equal-time blocks, excluded group collisions, unequal prefix counts, final
   partial batches, no event-text vocabulary, and policy-only threshold selection.
7. Explicit permutation oracle on short ID tuples; modifying a future turn/ID
   cannot change an earlier permutation. Cached and fully recomputed intervention
   scores agree. Keeping current last leaves current-only unchanged. Verify
   no-op/content-identical counts, final-logit-only selection and original lead
   labels. Test work limit at the exact bound and plus one before inference.
8. Actual separately trained tiny fit -> frozen NumPy -> save/load -> threshold
   -> heldout workflow for each mode, with Torch imports blocked in deployment.
   Wrong mode, inactive extra arrays, shape/dtype corruption, UCD mismatch,
   nonfinite values and rechecksummed semantic mismatches reject. Failed refit
   keeps the prior model; failed receipt after model publication leaves a private
   candidate without claiming rollback or successful evaluation.

Before any new official-test scoring, freeze implementations, this specification,
all three candidates per variant, vocabulary/source/partition hashes, loss
histories and their independent policy thresholds. Report matched NB/prior
controls separately under their own already-declared token/threshold contracts.
Use conversation-weighted prefix Brier, clipped log loss, tied AUC, any-alert
confusion/BA/F1/FPR and first-alert lead time with its true-positive denominator.
Retain per-seed results, descriptive mean/range/standard deviation, failures,
token truncation/OOV coverage, changed parameter counts, selected epochs,
compute, memory-measurement scope and adverse outcomes. Do not select the best
seed or silently introduce a score-averaging ensemble.

Fewer parameters, redundant duplicated means, removal of recurrence, pooling
normalization, gradient norm changes and different selected epochs all affect
optimization and capacity. Shared inputs/settings/common initial tensors reduce
some differences; they do **not** causally isolate a single factor. Three seeds
are not a confidence interval, and a curated roughly balanced benchmark is not
natural attack prevalence. Broader pretrained models, external confirmatory
cohorts, fairness, calibration and whole-repository parity remain separate gaps.

Keep source records, identities, per-conversation outputs, vocabulary and weights
local until rights/privacy are separately reviewed. Publish only appropriately
reviewed aggregate evidence and original code; diagnostic failures may still
contain sensitive detail and hashes are not anonymization. This design includes
no dependency/model download, network inference, paid provider, raw-data
redistribution, or implicit change to the existing main forecaster.
