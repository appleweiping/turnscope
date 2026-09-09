# Shared bidirectional expected context

`DualContextModel` learns two relationship-conditioned representations in one
shared latent coordinate system. It is separate from the existing SVD-plus-ridge
`ExpectedContextModel`; neither model changes the other's API or artifacts.
Training requires the existing optional `turnscope[context]` NumPy extra. Loading,
prediction, term statistics, context projection, and nearest-center assignment use
only the Python standard library. No language model or network service is used.

## Explicit training contract

```python
from turnscope.dual_context import ContextEdge, ContextRecord, DualContextModel

catalog = [
    ContextRecord("discussion", "question", "Where should we meet?"),
    ContextRecord("discussion", "answer", "Meet at the library."),
    ContextRecord("discussion", "followup", "Which library entrance?"),
]
forward = [
    ContextEdge("discussion", "question", "answer"),
    ContextEdge("discussion", "answer", "followup"),
]
backward = [ContextEdge(edge.conversation_id, edge.context_id, edge.source_id) for edge in forward]
model = DualContextModel(n_components=2, n_clusters=2, seed=17).fit(catalog, forward, backward)
print(model.predict("Where should we meet?").to_dict())
print(model.project_context("Meet by the library door.").to_dict())
warning = model.save("dual-context.json")  # Exclusive: refuses an existing output.
if warning is not None:
    print(warning)  # The complete model was published; private temp cleanup failed.
loaded = DualContextModel.load("dual-context.json")
assert loaded.digest == model.digest
```

This tiny invented example demonstrates mechanics, not predictive quality.
Records have a composite `(conversation_id, utterance_id)` identity, never a
delimiter-concatenated ID. Record and edge inventories are sorted before fitting.
Duplicate record identities, duplicate edges, missing endpoints, self references,
and cycles within either direction are errors. Each direction needs an edge.
Reversing the same acyclic relationships is supported; the directions need not be
reversals. The supplied labels `forward`/`backward` do not establish temporal
causality. The caller must supply and audit authentic relationships; the model
does not infer edges from adjacent rows, timestamps, or similar text.

Only training records enter feature selection, document frequency, column norms,
SVD, ranges, and clustering. Each catalog record is counted once even if many edges
reference it. Isolated training records also enter the training vocabulary and
normalization. Split whole conversations (and any related leakage groups) before
building this catalog. `predict(text)` and `project_context(text)` cannot refit or
extend any inventory. Out-of-vocabulary words do not change known-term weights.

## Numerical definition

The shared-space equations follow the expected-context construction in
[Zhang's dissertation, section 4.4 and Table 4.2](https://tisjune.github.io/papers/phd-thesis.pdf).
This implementation has its own tokenizer, numerical guards, graph contract,
clustering, missing-projection policy, and artifact format; it does not claim
bit-for-bit reproduction of ConvoKit or full-package parity.

1. Tokenize with TurnScope's Unicode word/apostrophe/hyphen regex, then `casefold`.
   There is no NFKC normalization, stemming, stop-word list, or pretrained model.
   Case folding can introduce combining marks. It happens after regex matching;
   therefore re-tokenizing a stored vocabulary token need not be idempotent.
2. Select terms meeting `min_document_frequency`, take at most `max_features` by
   descending training document frequency with lexical ties, then order columns
   lexically. Smooth IDF is `1 + log((N + 1)/(df + 1))`. Multiply raw term counts by
   IDF, L2-normalize each row, then L2-normalize each column across the full training
   catalog. Empty rows remain zero. Save these column norms for held-out text.
3. Let `C` contain the unique union of target-context rows from both directions.
   Compute one reduced SVD `C = U diag(s) V.T`. Keep only singular values above
   `max(1e-12, max(C.shape) * float64_epsilon * largest_singular_value)`.
   `drop_first=False` is the default; if explicitly true, remove the leading
   component before retaining at most `n_components`. No retained rank is an
   error, not a fabricated zero model.
4. For each direction, align source rows `A` and context rows `U_e` by its explicit
   edge list. Its term map is `W = A.T @ U_e @ diag(1/s)`. Terms are represented by
   `unit(W[t])`; query utterances by `unit(a @ W @ diag(1/s))`; context queries by
   `unit(c @ V @ diag(1/s))`. The second singular-value division for utterances is
   intentional. Context projections use the same `V` and `s` in both directions.

The largest-absolute feature loading in each retained basis column is made
positive, breaking exact loading ties by feature order. This fixes sign ambiguity,
not arbitrary rotations inside repeated singular-value subspaces. The same sorted
input is deterministic within a fixed numerical runtime; artifacts are not
promised bit-identical across BLAS versions/platforms or rank/truncation boundaries.

Tokenizer identity also pins `unicodedata.unidata_version`. Loading an artifact
under a different Unicode database is an explicit error, even when its vocabulary
is ASCII: newly assigned characters can change word boundaries. For example,
`"alpha\u1c89beta"` splits into two known words with older Unicode databases but is
one word with Unicode 16. Use the original compatible runtime for frozen inference,
or refit from the original training records and edges for the new runtime. Do not
edit the tokenizer version and recompute the checksum to bypass this requirement;
that would change the model's query semantics without recording a new fit.

## Range, orientation, and shift

For a term present in a source row, each edge contributes one observation,
regardless of its count or TF-IDF weight. Its range is the mean of cosine distances
between its normalized term map and the associated normalized retained `U` rows,
clipped to `[0, 1]`. A zero context projection is excluded rather than assigned a
fake distance. `support_edges` counts all term-present edges;
`projectable_edges` counts those with nonzero retained context projection.
Undefined term projections/ranges are `None`.

The query's range is its nonnegative, frozen TF-IDF-weighted average of defined
term ranges. `range_weight_coverage` reports the fraction of **known-vocabulary
weight** with a defined range; `support_weight_coverage` reports the fraction with
directional training-edge support. These are not fractions of original tokens;
`tokens` and `known_tokens` separately report token coverage. Unsupported terms
are not silently treated as zero-range terms.

`orientation = forward.range - backward.range` when both are defined.
`shift` is Euclidean distance between the two normalized query vectors when both
exist. Otherwise they are `None`. A direction's `reason` describes projection
availability: `ok`, `empty_text`, `out_of_vocabulary`, `no_direction_support`, or
`zero_projection`. These scores describe the learned relationship geometry, not
confidence, truth, psychological intent, or causal influence.

## Clustering and public results

Each direction clusters nonzero projected utterances from the training catalog,
with one sample per record ID. It uses Euclidean Lloyd updates with arithmetic-mean
centers, not spherical/renormalized centers. Initialization selects a seeded first
distinct vector and repeatedly selects the farthest remaining vector; a SHA-256
priority derived from `seed` and float coordinates breaks ties. This is
**farthest-first, not k-means++**. Assignment distance ties use the smallest center
index. Iterations are bounded by `max_kmeans_iterations`.

Identical projections still contribute multiple samples when their record IDs
differ. Requested clusters may exceed distinct projections; empty clusters can
also be removed. `training_summary()["clustering"]` records effective clusters,
sample/distinct counts, iterations, convergence, squared-distance objective, and
the degeneracy reason. Hitting the iteration limit is explicitly not convergence.
All-zero directional training projections produce an empty clustering. Centers
are frozen. Utterances, terms, and context queries are assigned to the same
direction's centers, with Euclidean distances; cluster numbers have no semantic
labels and are not comparable to another independently trained artifact.

`predict(text)` (also `transform`) returns an immutable `DualContextPrediction`,
with `.forward`, `.backward`, `.orientation`, `.shift`, and token counts. Each
direction includes its vector/range/coverage/reason and optional cluster ID and
distance. `project_context(text)` returns the shared context vector and assignment
to each direction's centers. `term_statistics()` returns fresh dictionaries with
term vectors, support/ranges, assignments, orientation, and shift. Their `.to_dict()`
forms use JSON arrays and `null` for unavailable values. `state` and `config` are
immutable fitted values; exported dictionaries can be edited without mutating them.

## Resource and artifact boundaries

Defaults: 30,000 catalog records; 60,000 edges per direction; 32 MiB total training
UTF-8 text; 4 million tokens; 100,000 vocabulary candidates; 256 selected features;
16 retained components; 8 clusters; 100 Lloyd iterations. A single text is limited
to 1 MiB UTF-8 and each nonempty identifier to 256 bytes. Each configuration field
also has a hard ceiling; invalid numbers and booleans used as integers are errors.

Before NumPy allocation, `max_dense_cells` (default 64 million float64 cells,
approximately 512 MB) bounds the conservative estimate
`4NP + 4CP + 2C min(C,P) + 2P min(C,P) + 8Pd + 4Nd + 4NK`, where `N` is catalog size,
`C` context count, `P` features, `d` the requested pre-drop dimension bound, and `K`
requested clusters. It allows multiple dense copies and SVD/Lloyd work arrays.
It is **not an RSS guarantee**: tokenization, input text, Python containers, allocator
overhead, BLAS implementation scratch, and threads are outside this accounting.
Deployment-level process limits remain the caller's responsibility.

Reduced SVD costs approximately `O(C P min(C,P))`; directional matrix products cost
`O(CPd)` plus explicit edge/source aggregation, and ranges use only observed term
occurrences rather than a vocabulary-by-edge distance matrix. Lloyd work is bounded
by `O(iterations * N * K * d)`. Frozen single-text projection costs `O(Pd + Kd)`
after tokenization, and term statistics `O(PKd)`. Artifact orthogonality validation
costs `O(Pd²)`. These are algorithmic bounds, not measured latency promises.

Artifacts use closed versioned JSON, never pickle. The entire canonical UTF-8
envelope is capped at 32 MiB; loading bounds file bytes before parsing, nesting at
32 levels, and structure at 2 million values. Shape, finite numeric ranges,
orthonormal basis, support/cluster/rank/accounting consistency, tokenizer, version,
and checksum are checked. Invalid or over-budget fits leave any previous fitted
state unchanged. Successful fit must already fit the exported envelope.

`save(path)` publishes exclusively using an owned temporary file and hard link.
An existing output raises `FileExistsError`. `overwrite=True` explicitly permits
atomic replacement of an unaliased regular file; symlinks, multiple hard links,
and nonregular targets are rejected. Filesystem atomicity assumes a cooperating
local filesystem and does not provide hostile-directory race isolation. A returned
cleanup warning means publication succeeded but an owned private temporary file
could not be removed; it does not mean the model failed to save.

Catalog/edge SHA-256 fingerprints include composite identities and training text,
but not raw catalog rows in the artifact. Vocabulary can still expose sensitive
words: treat artifacts as sensitive derived data. Checksums detect inconsistency,
not malicious alteration or authentic training provenance; rehashing cannot prove
the supplied data were used or that scores are accurate. Real held-out engineering
evaluation and dataset provenance are separate from these arithmetic tests.
