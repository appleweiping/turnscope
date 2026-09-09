# Shared dual-context API and CLI workflow

These development commands complement the existing `context` ridge predictor;
they do not replace its API or artifact format. Install the development branch's
optional `turnscope[context]` extra for fitting. Loading, transforming, evaluating
and assigning frozen centers require only the standard library. No automatic
dependency/model download, external provider call or plugin discovery occurs.

Artifact tokenizer identity also binds the Python Unicode database version.
Loading into a runtime with a different Unicode version is rejected rather than
silently changing word boundaries or case folding. Use a compatible Unicode
runtime for inference, or refit from the original audited training inputs; editing
the tokenizer label and recomputing the checksum is not a valid migration.

## Complete original demonstration

From a checkout using the same installed branch:

```shell
python examples/dual_context_demo.py --output-dir ./new-dual-demo
```

The output directory must be new and its parent must exist. The script writes
small original JSONL inputs, then invokes four separate `python -m turnscope`
processes: fit, transform, terms and evaluate. Each later command reloads the
saved model. The demo checks that the model bytes did not change and that an
unknown query stays in the all-query retrieval denominator. It is an arithmetic
and integration fixture, not a semantic-quality benchmark. A failed demo can
leave its already-created directory and files for inspection; rerun into a new
directory rather than overwriting them.

## Strict input formats

A catalog JSONL line has exactly three string fields:

```json
{"conversation_id":"discussion","utterance_id":"question","text":"Where should we meet?"}
```

An edge line has exactly three string fields:

```json
{"conversation_id":"discussion","source_id":"question","context_id":"answer"}
```

IDs are conversation-scoped; do not concatenate them with a separator. Both
endpoints must be present in their declared catalogs. Training rejects duplicates,
missing endpoints, self references and per-direction cycles. Supplied forward
and backward edges can be reversals, but need not be. Neither row adjacency nor
similar text is interpreted as a relationship. For retrieval, relevance edges
run from the query catalog to the candidate catalog; they are caller-declared
positives, not inferred labels. Query/candidate overlap is allowed only with
identical text for the same composite ID, and a query never retrieves itself.

UTF-8, CRLF/LF records and a final line without newline are supported. Blank
lines, extra fields, non-string values, duplicate JSON keys, non-finite numbers,
unpaired Unicode surrogates and invalid encoding fail. The public CLI does not
inherit any source-specific legacy-data exception from a research adapter.

## Fit, save and reload

```shell
turnscope dual-context fit training.jsonl forward.jsonl backward.jsonl model.json --n-components 16 --n-clusters 8 --seed 17
turnscope dual-context transform model.json new-catalog.jsonl --output projections.json
turnscope dual-context terms model.json --output terms.json
```

`fit --help` exposes the model's configuration fields, including explicit
`--drop-first`, minimum document frequency, feature/token/text/edge/record limits,
dense-workspace estimate and Lloyd iteration limit. `--drop-first` defaults to
false; actual retained dimensions and cluster counts are reported. Low-rank
degeneracy and non-convergence are not hidden. See the
[model contract](dual-context.md) for numerical definitions and all hard ceilings.

The fit command emits only the digest and training/cluster summary, not raw
training text. The saved model contains vocabulary and derived values and must
still be treated as private data. A transform row contains IDs, forward/backward
predictions, shared context projection and frozen cluster assignments. It does
not repeat the source text. Term export does include learned vocabulary.

All model/report paths must be new and have existing parent directories. Outputs
cannot alias inputs, including an existing hard link or resolved path alias.
This CLI never requests the API's explicit `overwrite=True` option.

## Fixed-pool multi-positive evaluation

```shell
turnscope dual-context evaluate model.json queries.jsonl candidates.jsonl relevance.jsonl --direction forward --k 1 --k 5 --k 10 --output retrieval.json
```

Use `backward` to evaluate the other map. Repeated `--k` values must be unique,
strictly increasing integers between 1 and 100; the default is `(1, 5, 10)`.
Candidates are the caller's fixed pool, not an automatically discovered corpus.
For a research claim, freeze the pool and protocol before examining scores and
use the same tasks for all models and baselines.

The standard-library API is available without the CLI:

```python
from turnscope import ContextEdge, ContextRecord, DualContextModel, evaluate_dual_context

model = DualContextModel.load("model.json")
report = evaluate_dual_context(
    model,
    [ContextRecord("new", "q", "Where should we meet?")],
    [ContextRecord("new", "a", "Meet at the library.")],
    [ContextEdge("new", "q", "a")],
    direction="forward",
)
assert report["training_separation_verified"] is False
```

That assertion is intentional. The evaluator cannot prove training/held-out
separation from a model digest; the caller must split and audit conversations and
related groups before training. The model remains frozen during evaluation, but
frozen inference alone is not proof of a leakage-free experiment.

Ranking uses descending cosine in the shared space, then ascending composite ID
for exact floating-point ties. Self candidates are excluded. Unprojectable
candidates cannot be retrieved; positive ones remain in the recall denominator.
MRR is the reciprocal of the first retrieved positive's rank, or zero when none
is retrieved. Recall@k is retrieved positives in the first k divided by **all**
declared positives, including unprojectable ones. A k larger than the ranked pool
does not create extra hits.

`summary.all_gold_queries` averages all queries with at least one positive.
OOV/zero-projection queries contribute zero instead of disappearing.
`summary.scorable_gold_queries` separately conditions on a projectable query;
even if none of its candidates can be projected, that query remains with zero.
`query_coverage` is the fraction of gold queries with a projection. Queries without
positives are explicitly undefined and excluded from both gold-query averages,
not counted as successful negatives. If a denominator is zero its mean is `null`.
Per-query counts, reasons and top candidates expose these distinctions. Cluster
numbers are not dialogue-act ground truth, and these metrics do not score
clustering accuracy or prove causal conversational influence.

## Limits, failures and private output

Each CLI JSONL input has a 64 MiB total limit and a 1 MiB line limit, including
JSON syntax/escaping. Catalogs have at most 30,000 rows, and edge files at most
60,000; lower configured model limits also apply. A model configuration with a
higher API ceiling does not raise these CLI input ceilings. File growth during
reading is checked against the cumulative byte limit again. An oversized or
invalid later row cannot publish a partial model/report.

Public evaluation defaults are 1,024 queries, 4,096 candidates, 60,000 positive
relationships, 32 MiB combined query/candidate UTF-8 text and one million
query-candidate pairs. The pair budget is checked before projections; scoring
does not allocate a full score matrix. Each projection costs approximately
`O(Pd + Kd)` after tokenization; ranking adds `O(QCd + QC log C)`, where `Q/C` are
query/candidate counts. CLI report encoding has a separate 64 MiB cap, checked
while aggregating complete rows. These are work/encoding limits, not process-RSS
or hard wall-clock guarantees.

Reports are `private_data: true`: IDs, vocabulary, vectors and scores can be
sensitive even without source text. Errors use fixed diagnostics and do not echo
raw input or nested parser causes. File output is written completely to an owned
temporary file, flushed and published exclusively by hard link; an existing
destination is not overwritten, even in a publication race. Hostile directory
mutation, remote filesystems and secure erasure are outside this contract.

If publication succeeded but temporary-file cleanup failed, exit status remains
zero and a warning says a private temporary copy may remain. Inspect that copy;
failure to clean it up is not proof that nothing was published. After a model
was saved, failure to deliver its optional stdout summary likewise preserves
success with a warning. A transform/terms/evaluation stdout failure returns 2;
stdout itself is not atomic and may contain a prefix. Short writes are detected,
not assumed successful or automatically resent. After a real failed output
descriptor, shutdown flushing is disabled so Python does not replace the chosen
status with exit 120. Other controlled command failures return 2.
