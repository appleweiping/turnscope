# Feature pipelines

`FeaturePipeline` gives TurnScope a small fit/transform interface for combining
stateful transformers (such as `TfidfVectorizer`) with pure conversation
features. Steps are named, ordered, and evaluated without mutating source
conversations.

```python
from turnscope import CallableTransformer, FeaturePipeline, TfidfVectorizer, conversation_features

pipeline = FeaturePipeline(
    (
        ("tfidf", TfidfVectorizer(min_document_frequency=2)),
        ("counts", CallableTransformer(conversation_features)),
    )
)
features = pipeline.fit_transform(conversations)
print(features[0].digest())
```

The pipeline requires an explicit fit before transformation and rejects
duplicate step names or duplicate conversation IDs. `FeatureRecord.digest()`
is a canonical cache key for JSON-serializable feature blocks.

For interpretable social-signal analysis, `linguistic_coordination()` accepts a
named mapping of function-word categories and returns one directional score for
each ordered speaker pair. A score is the fraction of adjacent target turns
that use a category when the source turn used it; `conditioned_turns` and
`coordinated_turns` make low-support results visible, while `None` means the
source never supplied evidence for that category.

`linguistic_diversity()` complements coordination with per-speaker lexical
profiles: token count, unique-token count, type-token ratio, and Shannon entropy.
It accepts the same optional metadata grouping field and returns zero-valued
metrics for an utterance group whose text contains no tokens.
