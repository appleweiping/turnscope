# Prefix-local frozen history-order diagnostic

`benchmarks/neural_history_intervention.py` implements the diagnostic in section 6
of [the frozen ablation design](neural-cga-ablation-design.md). It evaluates an
already fitted **main** `HierarchicalEventForecaster`; it does not train, select a
seed, recalibrate probabilities or choose a new threshold. This source helper is
not a new public `turnscope` model, artifact format or command-line interface.
Use a checkout or source distribution containing the helper, with the matching
installed package. Older releases are not implicitly upgraded by these docs.

This is separate from the independently trained current-turn, mean-word and
order-erased models. It asks how a particular frozen main model responds to a
specified perturbation of its inputs. Reordering can destroy conversational
coherence, so the result is an out-of-distribution sensitivity diagnostic, not a
causal effect of history order in human conversation.

## Fixed ordering and original endpoints

For each original eligible endpoint, take only its completed, retained token
sequences. Keep the current turn last. For every historical turn, compute SHA-256
over UTF-8 compact sorted-key JSON with exactly these fields:

```text
format: turnscope.prefix-history-permutation.v1
seed: turnscope-history-order/2026-09-12
conversation_id: original conversation ID
endpoint_turn_id: original endpoint ID
historical_turn_id: original historical turn ID
```

JSON uses `ensure_ascii=False`. Historical turns sort by digest bytes, then by
original zero-based position for a hash tie. The endpoint is appended unchanged.
There are no model seeds, labels, future IDs, future lengths, scores or source
split names in this ordering key. `prefix_history_order(conversation_id,
prefix_turn_ids)` exposes this pure ordering primitive with bounded, unique IDs.
At one or two completed turns it is necessarily a no-op.

Every prefix is ordered separately. Permuting the maximum conversation once and
using its intermediate outputs would leak later messages into earlier endpoints.
Instead, the helper restarts the turn recurrence from zero for each locally
reordered prefix and keeps **only its final output**. Original endpoint handles,
labels, chronological lead times and the main model's learned threshold remain
unchanged. Intermediate positions in the transformed sequence are not additional
observations and are never assigned new labels.

## Evaluation API

From a source tree where the `benchmarks` directory is importable:

```python
from benchmarks.neural_history_intervention import (
    HistoryInterventionLimits,
    evaluate_history_intervention,
)

# model: already fitted main HierarchicalEventForecaster
# heldout: prepared SequenceForecastDataset or original Conversation iterable
report = evaluate_history_intervention(
    model,
    heldout,
    limits=HistoryInterventionLimits(max_permutation_positions=2_000_000),
)
```

The helper captures one immutable model state for admission, parameters,
threshold and report identity. It rejects unfitted or non-main model objects,
overlap with any training/model-validation/policy-validation group, and datasets
with no eligible examples. It does not drop long, OOV, unchanged or inconvenient
cases when a budget or numerical check fails; the call fails instead.

All original eligible examples remain in the primary metrics: conversation-
weighted prefix Brier, clipped log loss, tied AUC, any-alert confusion/BA/F1/FPR,
and first-alert lead time with its original true-positive denominator. The helper
does not optimize a changed-only subset or reuse transformed intermediate logits.
It reports that weights and threshold were not refitted, calibration is not
claimed, and whole-repository parity is not established.

## Cache, work and memory limits

Within each conversation, every unique completed turn is word-encoded once under
the captured frozen parameters. Multi-layer bidirectional word outputs use the
same named numerical contract as ordinary main inference. This private cache is
not accepted from callers, retained between model states, or reused during
training. A separate fresh turn recurrence processes each endpoint's permutation.

With T actual encoded positions, U unique turns, K eligible endpoints and
S the sum of their prefix lengths, the charged affine work is:

```text
word = sum_layers T * 6h * (input_width + h)
turn = sum_layers S * 3g * (input_width + g)
head = K * q * (g + 1)
total = word + turn + head
```

Word input width is the embedding width for its first layer and `2h` thereafter.
Turn input width is `2h` for its first layer and `g` thereafter. These are declared
affine multiplication counts, not total FLOPs or measured CPU instructions.
The turn work can be quadratic in conversation length across many endpoints;
the word cache does not make the complete diagnostic linear.

Original data and per-conversation numerical limits still apply. Each complete
conversation diagnostic must fit the saved per-inference affine quota, and the
whole collection must fit the model's original aggregate inference quota. Before
constructing any permutation tuples or numerical arrays, the helper admits the
sum of all endpoint prefix lengths against `max_permutation_positions`: default
2,000,000, hard maximum 8,000,000. Values must be positive exact integers; boolean,
fractional and excessive limits reject.

Numerical cache admission adds twice `U * 2h * 8` bytes to the original numerical
workspace estimate, allowing the float64 cached word vectors and a copied order
view. Only one conversation's numerical cache is scored at a time. The report
retains the maximum workspace estimate. It is not process RSS: Python encoded
catalogs/order tuples, interpreter objects, allocator/BLAS caches and native
overheads are not exhaustively represented. Logical order positions have their
separate explicit bound.

## Aggregate bindings and no-op denominators

The report binds model, parameters, vocabulary, prepared partition, and fixed
ordering protocol digests. For each original canonical endpoint, it independently
accumulates one compact UTF-8 JSON item plus a literal LF into three SHA-256
inventories: original token sequence, transformed token sequence and permutation
indices. The prediction inventory hashes the compact JSON array of scores.
These are integrity bindings, not authentication or anonymization.

No raw text, conversation/turn IDs, vocabulary or per-conversation scores are
returned. A reproducible research run must additionally pin the input archive,
helper/runtime sources and dependency versions outside this report; a prepared
partition digest alone does not identify a downloadable source archive or attest
to code provenance. This helper does not publish files or contact providers.

Counts distinguish all eligible prefixes, prefixes with at least two historical
turns, changed index order, and changed **encoded content**. Repeated messages or
OOV-equivalent encodings can change index order without changing model input.
Unchanged examples remain in the metrics. Raw/retained/known token counts and
truncated-turn counts describe the unique encoded observation catalog, not an
inflated sum over copied prefixes.

## Independent implementation evidence and remaining limits

`tests/test_neural_history_intervention.py` contains independently authored
numerical fixtures, explicitly not trained-model evidence. Across all four
one/two word-layer and one/two turn-layer combinations, cached scores agree with
fully recomputed `infer_encoded_turns` on each independently permuted prefix.
Tests verify exact UTF-8 ordering and inventory hashes, unique-cache call counts,
fresh recurrence lengths, repeated-content counts, future-ID/text isolation,
unchanged threshold/metric denominators, state replacement during admission,
and exact resource bounds followed by one-less-than-required rejection before
numerical allocation. Invalid and overlapping cohorts are not successful zeros.

One additional optional test performs an actual one-epoch CPU fit of a tiny main
model on authored conversations with separate model/policy validation, then
checks the diagnostic against full prefix recomputation. It uses the already
available optional Torch dependency and downloads nothing. If Torch is absent,
that test is visibly skipped; the numerical fixtures are not renamed as training.

No official CGA test scoring, parameter search, three-seed ablation comparison or
real-data quality conclusion is established by these tests. CPU/platform floating
reductions may change low bits, so score agreement uses explicit tolerances;
prediction hashes are not a cross-platform bitwise reproducibility guarantee.
