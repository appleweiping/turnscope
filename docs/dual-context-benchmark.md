# Dual-context reply retrieval: frozen research protocol

This development benchmark exercises a complete shared-space dual-context model
on an existing public conversation archive. It is a bounded **reply-context
retrieval experiment**, not a dialogue-act classifier, cluster-accuracy test, or
evidence of whole-ConvoKit parity. The implementation is original; its
expected-context mathematics is attributed to Justine Zhang's
[2021 dissertation, sections 4.4.1–4.4.4](https://tisjune.github.io/papers/phd-thesis.pdf)
and independently checked against ConvoKit's expected-context implementation at
`5dabba5ae034686185d9ff6612e2d57d694bff68`.

The adapter, miniature synthetic end-to-end tests and one complete frozen
official-fold experiment were run on 2026-09-09. The aggregate
[observed report](../benchmarks/results/cga-dual-context.json) is included without
discarding poor scores. These are development-branch capabilities, not a statement
about the published main branch or a released package.

## Observed result: lexical retrieval is stronger here

On this fixed task, the dual-context model underperforms lexical TF-IDF on all
four MRR measurements and all four Recall@10 measurements. Its test-backward MRR
is also lower than the shuffled-relation null. These results support neither a
quality-parity claim nor the claim that the learned representation is a better
retriever. Range/orientation/clustering mathematics passing unit tests is a
different claim from useful predictions.

Every row below has 256 selected queries and the same task-specific pool of
1,024 candidates before excluding each query's own identity. `Q` is the number of
scorable queries, not a denominator substitution: MRR and Recall always average
over all 256, including undefined queries as zero. Values are rounded here;
the JSON retains their original precision, all positive/candidate counts,
conditional MRR, and page/pair macro Recall@1/5/10.

| Fold | Direction | Method | MRR | R@1 | R@5 | R@10 | Q | Page macro MRR | Pair macro MRR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | Forward | Dual context | 0.011574 | 0.000000 | 0.009766 | 0.013672 | 252 | 0.010547 | 0.010667 |
| Validation | Forward | TF-IDF | 0.066235 | 0.025716 | 0.068685 | 0.096029 | 252 | 0.060553 | 0.059666 |
| Validation | Forward | Shuffled null | 0.009804 | 0.003906 | 0.003906 | 0.009766 | 252 | 0.008060 | 0.007891 |
| Validation | Forward | Training mean | 0.009853 | 0.003906 | 0.003906 | 0.009115 | 256 | 0.008001 | 0.007836 |
| Validation | Backward | Dual context | 0.018276 | 0.003906 | 0.015625 | 0.039062 | 251 | 0.015283 | 0.015674 |
| Validation | Backward | TF-IDF | 0.045845 | 0.023438 | 0.054688 | 0.082031 | 251 | 0.053130 | 0.050891 |
| Validation | Backward | Shuffled null | 0.007097 | 0.000000 | 0.003906 | 0.011719 | 251 | 0.006550 | 0.006452 |
| Validation | Backward | Training mean | 0.011436 | 0.003906 | 0.011719 | 0.015625 | 256 | 0.007561 | 0.009373 |
| Test | Forward | Dual context | 0.016277 | 0.003906 | 0.011719 | 0.023438 | 253 | 0.016665 | 0.015849 |
| Test | Forward | TF-IDF | 0.066452 | 0.024740 | 0.064779 | 0.107747 | 253 | 0.067679 | 0.065148 |
| Test | Forward | Shuffled null | 0.009414 | 0.003906 | 0.003906 | 0.007812 | 253 | 0.008615 | 0.008219 |
| Test | Forward | Training mean | 0.010539 | 0.001953 | 0.005859 | 0.009766 | 256 | 0.012379 | 0.011718 |
| Test | Backward | Dual context | 0.011360 | 0.000000 | 0.015625 | 0.027344 | 255 | 0.012165 | 0.012149 |
| Test | Backward | TF-IDF | 0.048033 | 0.027344 | 0.054688 | 0.070312 | 255 | 0.038890 | 0.038220 |
| Test | Backward | Shuffled null | 0.012489 | 0.003906 | 0.011719 | 0.023438 | 255 | 0.015131 | 0.014333 |
| Test | Backward | Training mean | 0.008147 | 0.000000 | 0.007812 | 0.019531 | 256 | 0.008139 | 0.007817 |

The four tasks contain 308/256/317/256 positive relations, respectively.
The represented page-group counts are 177/163/176/179 and pair-group counts are
184/176/190/191. These are controlled, relation-gold-inclusive candidate pools,
not retrieval over every held-out utterance; they do not establish performance
under a natural deployment candidate distribution. The same test source had
already been used for this repository's different forecasting benchmark. This
experiment neither tunes on these results nor claims globally untouched data.

The main forward clustering reached its 100-iteration cap with
`converged=false`; main backward converged in 56 iterations. Null forward
converged in 94; null backward reached 100 without convergence. All have 8
effective clusters, but no gold semantic labels exist, so none of these counts or
objectives is called accuracy. Iterations were not increased after seeing this
result.

Both models fit all 15,744 official nonheader training records and 9,804 edges per
direction, with 13,120 unique context records, 1,169,933 tokens, 256 selected
features, numerical rank 256 and 16 retained dimensions. The entire run took
204.273 seconds: source hashing/parsing 27.013, main fit 35.317, null fit 33.648,
baseline construction/training-context projection 16.389, and retrieval scoring
87.502. Remaining time includes external artifact verification and orchestration.
Fit Python-tracked allocation peaks were 244,301,921 and 239,398,003 bytes; these
are not native/process RSS. Runtime: Python 3.12.13, NumPy 2.5.3, Unicode database
15.0.0 on Windows 11.

The report is byte-identical to its first successfully published external
artifact (27,947 bytes). Its exact bindings are:

```text
report SHA256:       c6f2456df80fabff6c929fd14c00d790d6b23a604ea413fb6ec4cae218c422a7
main model SHA256:   031ef0751d6ecd1945064d379b89de8d1bef5af944c78ac5045b4276ca278f18
null model SHA256:   236847b7f83fa99b968f65f9024eda91d6cfd12ec1d4bac35bf489d7f11cd452
model input SHA256:  bfc169daa8a96b0cde0503017ba0fe7ec53dd628da425bdeb6f28c0fc5a6772a
protocol SHA256:     08b1648b202ef7b94e7eda7c2301e48bcc1c87a85042d3827322b8596a9a854b
benchmark SHA256:    a24d1f77e5fba1f2ea8ebcd52e68b14ee942ae79ee1b69270a516a82f17a9ceb
model tokenizer:     turnscope.regex-casefold.v1/ucd-15.0.0
```

All runtime modules, benchmark source and project configuration hashes matched
before/after this run; both fitted artifact digests were unchanged by evaluation.
The report contains every source hash, not only the abbreviated inventory above.
Later documentation/report publication does not alter those runtime bindings.

An earlier attempted run was stopped when parallel independent review discovered
that artifacts needed to bind the Unicode database version. It produced no
published report, and no scores from that attempt were used. After the compatibility
guard and cross-version tests passed, this run restarted with **the original
protocol**, not a different seed, dimension, cluster count, candidate pool or
metric. The negative results above are the retained completed result.

## Frozen before seeing retrieval scores

`BenchmarkProtocol` fixes seed `20260909`, 16 retained dimensions, explicitly
drops the leading singular component, uses minimum document frequency 3, at most
256 vocabulary terms, and 8 Euclidean Lloyd clusters with at most 100 iterations.
The CLI has no score-driven tuning options. Small authored unit fixtures use
smaller explicit configurations only to check integration.

All nonheader utterances in the official training fold fit vocabulary, IDF and
column norms once. Both declared directions share one context SVD per model;
forward edges mean parent → observed reply and backward edges reverse those
retained authentic edges. No adjacent-utterance or timestamp-based edges are
invented. One model fits actual training relations and, when structurally
available, another fits the fixed negative-control relations. Held-out text is
only projected through frozen parameters. Artifact roundtrips and unchanged
model digests are required.

The official validation and test folds are both reported without choosing a
configuration from either. The test fold was previously used by this repository's
different personal-attack forecasting experiment. It is therefore **not globally
unseen data**, and this is not an independent external replication.

## Source identity and relation audit

Use an already downloaded
[CGA-WIKI archive](https://convokit.cornell.edu/documentation/awry.html). The script
does not download anything, extract files, or read the large parsed-analysis
member. It hashes the entire ZIP before parsing and requires:

```text
84e2d1ac60a3269251b5e175e549fc65cec875fdc926f99172b5b70d3ca1b122
```

Limits include 128 MiB encoded ZIP, 4 MiB metadata, 256 MiB expanded utterance
stream, 16 MiB per JSONL row, 50,000 source utterances, and 1,000,000 UTF-8 bytes
per retained text. JSONL is separated by actual LF bytes, not Unicode line
separators inside strings. Duplicate JSON keys, invalid Unicode and numeric
overflow are rejected. Utterance IDs must be globally unique in this source;
model identities remain `(conversation_id, utterance_id)` pairs.

The pinned archive is **not strict JSON**: a recursive source audit found 11
literal `NaN` values, all precisely in the top-level `reply-to` field. Initial
strict parsing failed, correctly. An earlier permissive read-only tally counted
these together with dangling references; that preliminary count of 1,215 is
superseded by **1,204 dangling references plus 11 legacy NaN parents**.

The narrowly scoped source parser retains an explicit legacy-NaN marker. Each
such parent edge is quarantined, not changed to null, repaired, or used for
training/retrieval. Its valid text node remains eligible for the catalog and
other nodes' valid replies to it remain eligible edges. The report separately
counts retained nonheader nodes and accepted incoming replies involving these
nodes: all 11 have retained nonheader text, and 13 valid replies to them survive.
NaN anywhere else, nested NaN in `reply-to`, Infinity/-Infinity, and finite
numeric values masquerading as parent IDs are rejected. Public model/evaluation
JSON interfaces do not inherit this legacy exception.

Primary exclusion reasons are disjoint: legacy NaN, dangling, self reference,
cross-conversation reference, cyclic edge, header endpoint. Null parents are
counted separately. Only cycle-member edges are removed; a valid tail pointing
to a cycle member is not automatically discarded, and no replacement edge is
created. Nodes are not removed merely because their own parent is invalid.

The verified read-only inventory, before any model fit, is:

| Quantity | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Official conversations | 2,508 | 840 | 840 | 4,188 |
| Nonheader catalog nodes | 15,744 | 5,239 | 5,205 | 26,188 |
| Retained parent → reply edges | 9,804 | 3,242 | 3,215 | 16,261 |
| Legacy NaN parent edges | 3 | 5 | 3 | 11 |
| Dangling parent edges | 719 | 230 | 255 | 1,204 |
| Self edges | 40 | 18 | 26 | 84 |
| Header-endpoint edges | 5,160 | 1,734 | 1,703 | 8,597 |

There are 30,021 raw nodes, 3,833 headers and 3,864 null parents. The 26,157
nonnull parent fields reconcile exactly to the six exclusion classes plus
accepted edges; cross-conversation and cyclic-edge counts are both zero here.
Training text is 6,867,870 UTF-8 bytes; no official training record is truncated.
Reciprocal matched-pair membership and page groups are hard-checked for
train/validation/test isolation. Toxicity predictions, attack labels, parses,
timestamps, speaker IDs and page titles never become model features.

## Controlled candidate pools and honest denominators

For each held-out fold and direction, at most 256 eligible source queries are
selected by SHA-256 ordering of the seed, direction and composite ID. Each query
must have at least one retained true context. Every selected query's positive
contexts is included in the shared candidate pool; other nonheader fold nodes
fill the pool to at most 1,024 by a different deterministic SHA-256 order. If all
required positives alone exceed this cap, the benchmark fails instead of
dropping positives. The query itself is excluded from its own candidates.

**Candidate construction uses relation ground truth.** This deliberately
controlled pool is not full-corpus retrieval or a natural deployment candidate
distribution. Every model and baseline sees exactly the same frozen tasks,
candidate pools and tie policy. No candidate is selected according to scores.

Cosine score ties use ascending composite IDs. Multi-positive MRR is the
reciprocal rank of the first retrieved true context. Recall@1/5/10 is the number
of retrieved positives divided by *all* true contexts for the selected query.
An undefined/OOV candidate is unrankable but remains in this denominator. An
undefined/OOV query earns zero in the primary all-selected-query MRR and Recall;
query coverage and conditional-on-scorable MRR are reported separately. Even a
projectable query with no projectable positives remains a scored zero, not an
excluded example. Eligible-versus-selected counts and every scored/unscored
candidate/positive count are visible.

Per-page and per-matched-pair macro results first average queries within each
represented group, then weight represented groups equally. These can differ
from micro averages; they do not claim to cover unselected groups. There are no
human gold cluster labels. Cluster sizes, projection reasons, range, orientation
and shift are descriptive geometry only, with defined/total counts—not accuracy.

## Baselines and conditional negative control

- **Lexical TF-IDF cosine:** independently fitted on each training utterance once,
  smoothed IDF, the same frequency/feature caps, no column normalization. Its
  vocabulary and parameters remain frozen during held-out projection.
- **Training-context mean:** one vote per authentic training edge target
  occurrence, projected in the main model's context space and normalized. This
  intentionally ignores the query; forward and backward means can differ.
- **Relation-shuffled null:** all training nodes move to an explicitly synthetic
  scope; original composite IDs are opaque canonical JSON strings, avoiding
  delimiter collisions. Permuting the target multiset preserves each node's
  incoming and outgoing degrees. Backward edges reverse that same fake graph.

The null rejects self edges, duplicates and cycles using at most 128 deterministic
attempts. The seed is never replaced according to scores. If no legal graph is
found, its result is explicitly `unavailable` and other models still run. This is
structurally conditioned shuffling, **not a uniform sample of all mappings**.
Attempts, rejection reasons and unchanged-edge fraction are reported. Independent
decoded-ID/text and context-inventory hashes verify the same original texts and
shared context inputs; vocabulary, document frequencies, column norms and singular
values are also compared after fitting. No test-based negative control is chosen.
The real-source structural preflight accepted attempt 4 (two self-edge rejections,
one cycle rejection), with 1 of 9,804 original training edges unchanged. This
preflight examined structure, not model scores.

## Reproduce, without overwriting prior evidence

From an explicit development checkout with the existing NumPy context extra:

```powershell
.venv/Scripts/python.exe -m pytest --no-cov tests/test_dual_context_benchmark.py
.venv/Scripts/python.exe -m benchmarks.benchmark_dual_context `
  D:/Company/nlp-original-projects/alignment/datasets/cga-wiki/conversations-gone-awry-corpus.zip `
  --output D:/Company/nlp-original-projects/alignment/artifacts/dual-context-fresh.json
```

The output parent must already exist and the output path must be new. A complete
bounded aggregate JSON payload is flushed to a fresh helper-owned temporary file
and published by an exclusive hard link; an existing path, hardlink alias or
publication race cannot overwrite an artifact or the source archive. Filesystems
without hard-link support fail without a fallback overwrite. Failed evaluation
publishes no report. A stdout failure after publication reports that the artifact
was completed and published, rather than claiming it was absent. Exceptions are
reported by type, never by raw text or credentials.

The aggregate report binds the archive/member/input hashes, protocol/task hashes,
model digests, all runtime modules, benchmark source and project configuration
before/after the run, plus Python/NumPy/platform versions and timings.
Unicode database versions and each model's actual tokenizer identity are explicit;
loading a model under a different Unicode database version is rejected.
Per-fit `tracemalloc` covers Python-tracked allocations; it excludes parsing and
the external artifact roundtrip, but includes validation performed inside
`model.fit`; it is not process or native-library RSS. Source parsing, baseline
construction/training-context projection, retrieval scoring and full-run timings
are named separately. Exact floating-point
results may vary across BLAS/runtime platforms; bitwise cross-platform equality
is not promised.

Raw archive records, texts, source IDs, tokens, vocabulary and fitted parameters
are not published. Dataset accessibility is not an additional redistribution
license grant. A useful or poor result is retained as observed; no quality
threshold is manufactured, no negative result is dropped, and reply retrieval
alone cannot establish semantic dialogue-act quality, clustering validity,
neural-model parity or complete product alignment.
