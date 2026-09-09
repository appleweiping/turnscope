# Reply-aware lexical coordination

`reply_coordination` measures how a responding speaker's category usage changes
when the message being replied to uses that category. It follows actual
`reply_to` links, aggregates across conversations, and subtracts that responder's
usual category usage when replying to the same partner.

```python
from turnscope import iter_path, reply_coordination

scores = reply_coordination(
    iter_path("conversations.jsonl"),
    {"articles": ["a", "an", "the"]},
    speaker_field="speaker_id",
    min_replies=20,
    min_conditioned_replies=5,
)
for score in scores:
    print(score.source, score.target, score.category, score.score)
```

## Direction and exact denominator

`source` is the speaker of the parent message; `target` is the speaker who
replies. For each observed ordered speaker pair and lexical category, the
output retains these counts:

| Output | Meaning |
|---|---|
| `replies` | All target-to-source replies, whether or not either uses the category |
| `conditioned_replies` | Replies whose parent uses the category |
| `response_category_replies` | Replies whose response uses the category |
| `coordinated_replies` | Replies where both parent and response use the category |

`conditional_rate = coordinated_replies / conditioned_replies`.
`baseline_rate = response_category_replies / replies`.
`score = conditional_rate - baseline_rate`.

For example, a parent containing `the` receives two replies, one containing
`the` and one without it. A second parent from the same speaker lacks `the`
and receives one reply that also lacks it. The conditional rate is `1/2`, the
baseline is `1/3`, and coordination is `1/6`. Each child is a distinct sample,
so the first parent's category is counted twice in the conditioned denominator.
Replies by this responder to another partner do not affect this pair's baseline.

A responder who uses the category in every reply has a conditional rate of 1
and a baseline of 1, giving coordination 0. Negative values indicate that the
category is less common in responses to parents that use it. These are
descriptive frequency differences, not causal effects or significance tests.

## Edges, identities, and corpus scope

Speaker identity defaults to `Utterance.role`. Use `speaker_field` for a stable
metadata identity when distinct people share a role. String IDs are matched
exactly and merged across conversations. Missing, blank, or non-string speaker
metadata fails for every utterance, including roots. The caller must ensure
that equal speaker IDs refer to the same person across the supplied corpus.

Only explicit parent-to-reply links between different identities contribute.
Roots and same-speaker replies are excluded. Adjacency, timestamps, and sequence
positions do not create links. Reordering utterances or conversations preserves
all output rows and values. Reply IDs are scoped to each conversation; missing
parents, duplicate utterance IDs, cycles (including self-links), and duplicate
conversation IDs are errors. Forward references in input order are permitted.

The API consumes conversations once, validates each complete reply forest,
and accumulates sparse observed speaker pairs. It returns no rows for
unobserved pairs. Each observed pair produces one row per configured category,
ordered by source, target, then category. An empty valid corpus returns `()`.

## Support and lexicons

`min_replies >= 1`, `min_conditioned_replies >= 1`, and
`min_response_category_replies >= 0` apply independently to each pair/category.
The defaults are 1, 1, and 0. If any count is below its threshold, `score` is
`None` and `support_failures` identifies the unmet thresholds. Counts, baseline,
and any calculable conditional rate remain visible. With no parent category
occurrences, `conditional_rate` is also `None`; this is distinct from a measured
zero. Thresholds do not pool partners or discard unrelated categories.

Each lexicon entry must be a nonempty iterable of individual lexical tokens;
a string, malformed word, or empty category fails. Words are casefolded,
duplicates collapse, and overlapping categories are allowed. Matching uses the
same Unicode-word/apostrophe/hyphen tokenizer as the vector workflow. It applies
no stemming, wildcard matching, dependency parsing, or composition normalization.

With `categories=None`, the five small built-in dictionaries from
`default_coordination_categories()` are used. They are examples, not a validated
linguistic inventory. In particular, a literal `n't` does not match every
contracted word, and categories with the same names as external inventories
need not contain the same words. Results are not numerically interchangeable
with other tools that use different dictionaries, selectors, pooled speaker
statistics, or thresholds. Validate and document the lexicon for the language
and research question before interpreting a score.

## CLI and compatibility

```bash
turnscope reply-coordination conversations.jsonl --speaker-field speaker_id \
  --categories lexicon.json --min-replies 20 --min-conditioned-replies 5 \
  --min-response-category-replies 2 --output scores.json
```

The command emits a JSON array of score rows. `lexicon.json` must contain a
strict JSON object mapping names to term arrays; duplicate JSON fields fail.
Exit 0 means the calculation succeeded even if some scores lack support.
Exit 2 means configuration or input validation failed. Input/output path
collisions are rejected; validation errors leave an existing output untouched.

The older `linguistic_coordination`, `CoordinationScore`, `coordination` CLI,
and service operation named `coordination` retain their original contract:
adjacent turns by different roles, and a conditional response rate without
subtracting a baseline. They do not implement the reply-based quantity above.
Existing applications keep their numeric behavior; new reply-network analysis
should use this separate API or `reply-coordination` command.

## Complexity

For total input text length `T`, category membership matches `M`, utterance and
edge counts `U`/`E`, `P` observed ordered pairs, and `K` categories, annotation
and accumulation cost `O(T + M + U + E + A)`, where `A` is the total number of
parent/response category memberships processed across reply edges. Producing
the sorted output costs `O(P log P + P*K)`. No all-speaker-pairs cross product
is created. Working memory retains the lexicon, seen conversation IDs, sparse
pair/category counters, and annotations/forest for one conversation; the
returned tuple additionally retains `P*K` rows. Large branched conversations
remain materialized as one conversation, and the API does not estimate
confidence intervals or account for dependent replies.
