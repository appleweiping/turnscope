# Fitted sparse conversation representations

`TfidfVectorizer` fits vocabulary and inverse document frequency (IDF) on an
explicit training corpus. Reuse that instance, or its saved artifact, to project
validation/test conversations. Projection never updates the vocabulary,
document counts, or IDF. Split data before fitting; the library cannot infer
whether a user-selected training corpus includes held-out records.

```python
from turnscope import TfidfVectorizer, SparseSimilarityIndex, iter_path

model = TfidfVectorizer(min_document_frequency=2, max_features=20000)
model.fit(iter_path("train.jsonl"))
model.save("tfidf.json")

restored = TfidfVectorizer.load("tfidf.json")
candidates = restored.transform_corpus(iter_path("train.jsonl"))
index = SparseSimilarityIndex(candidates)
for query in iter_path("heldout.jsonl"):
    vector = restored.transform_conversation(query)
    print(query.id, index.query(vector, limit=5))
```

## Arithmetic and normalization

Each conversation is one training document, regardless of its number of turns.
Document frequency counts a term at most once in that conversation. With `N`
training conversations and document frequency `df(t)`, the fitted weight is
`idf(t) = 1 + ln((1 + N) / (1 + df(t)))`.

The tokenizer matches Unicode word runs with internal ASCII apostrophes or
hyphens, then casefolds each match. For example, `Straße` becomes `strasse` and
`can't` stays one term. Tokenization applies independently to each utterance;
words do not merge across turn boundaries. It applies no stemming, stop-word
removal, language-specific segmentation, or Unicode composition normalization.
The artifact records the tokenizer contract version. Pin Python when exact
Unicode behavior across runtime versions matters.

`min_document_frequency` removes rare terms. `max_features` retains terms in
descending document frequency, breaking ties lexically. Training order does
not change the resulting artifact. Duplicate conversation IDs fail so repeated
documents cannot silently inflate the denominator. The fit operation consumes
an iterable once and commits its state only after success; a failed refit keeps
the previous model intact.

Projection aggregates token counts across all turns. Unknown/pruned tokens are
discarded before relative term frequency is calculated:
`weight(t) = count(t) / sum(known counts) * idf(t)`. Available normalization modes
are `none` (retain those weights), `l1` (divide by their absolute sum), and `l2`
(unit Euclidean norm, the conversation default). An empty or all-unknown
conversation yields an empty vector. A vocabulary emptied by filtering is a
valid fitted model. `transform_corpus` returns immutable, lexically ordered
conversation-ID mappings and rejects duplicate IDs.

The existing `transform(conversation)` API continues to return per-utterance
relative-TF times IDF weights with no additional normalization. Its return
shape and default arithmetic are unchanged. `fit_transform` retains that
per-utterance behavior. Use `transform_conversation` or `transform_corpus` for
the conversation representations described above.

For two training documents `red red blue` and `blue green`, the IDF of `blue`
is 1, while `red` and `green` each have IDF `1 + ln(3/2)`. A held-out document
`red unseen` maps to `{red: 1}` under L2 normalization, with no effect on the
fitted vocabulary or IDF.

## Similarities

`sparse_cosine(left, right)` calculates cosine without a dense vocabulary-sized
array. Zero vectors have similarity 0. `normalize_sparse` and cosine support
signed finite coordinates and scale before computing norms to avoid overflow
for large finite values. Invalid coordinate types, booleans, NaN and infinity
fail explicitly.

`SparseSimilarityIndex` snapshots and L2-normalizes its input vectors, then
builds term-to-document postings. Queries touch only overlapping terms.
Results have `id` and `score`, ordered by decreasing cosine and then lexical
ID. `minimum_score` is a strict threshold in `[0, 1]`; zero or negative matches
are omitted, and `exclude_id` can remove a self-match. This is exact lexical
similarity, with no dense embeddings, approximate index, or learned semantics.
Candidate and query vectors must use the same fitted vocabulary/IDF; the CLI
enforces this by loading one model for both.

## Portable artifacts and CLI

Artifacts contain only configuration, training document count, selected terms
and document frequencies, tokenizer version, and a SHA-256 checksum. They
contain no conversation IDs, raw text, or executable objects. Vocabulary terms
can still contain sensitive words, so treat model artifacts as derived data.
IDF is reconstructed from validated integer frequencies. Unsupported formats,
versions/tokenizers, duplicate fields or terms, inconsistent frequencies,
noncanonical ordering, and checksum mismatches are errors. A checksum detects
accidental corruption; it does not authenticate an artifact against intentional
replacement. Saving uses a temporary file in the same directory and atomic
replacement after successful serialization.

```bash
turnscope vectors fit train.jsonl tfidf.json --min-df 2 --max-features 20000
turnscope vectors transform tfidf.json heldout.jsonl --normalization l2 -o vectors.json
turnscope vectors query tfidf.json candidates.jsonl queries.jsonl --limit 5 -o matches.json
```

`query --exclude-self` omits candidates whose ID equals the query ID.
`--minimum-score 0.25` retains scores strictly greater than 0.25. Commands
return 0 on success and 2 on input/validation errors. Output paths that resolve
to any input path (including hard links) are rejected before writing. JSONL
training can be streamed; vector and result exports materialize their output
so malformed records cannot produce partially serialized JSON.

## Complexity and limits

For `T` training tokens, `V` distinct terms, and `N` conversations, fit takes
`O(T + V log V)` time and retains `O(V + N)` counters/IDs plus one conversation's
tokens. `max_features` bounds fitted state, not the transient vocabulary while
fitting. A single projection costs tokenization plus `O(K log K)` sorting for
its `K` nonzero terms. Corpus projection retains all resulting sparse rows.

Index construction costs normalization and sorting per input sparse row and
retains `O(Z)` postings for `Z` nonzero coordinates. A query visiting `E`
overlapping postings and `C` candidate documents takes
`O(Q log Q + E + C log k)` time for `Q` query coordinates and top-`k` selection,
with `O(E + Q + k)` working memory. Candidate products are accumulated with
`math.fsum` to reduce cancellation error. The current index is in-memory and
immutable; changing candidates requires rebuilding it. Dense matrices,
dimensionality reduction, speaker-level fitting, pretrained encoders, and
distributed storage are outside this workflow.
