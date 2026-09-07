# Feature pipelines

`FeaturePipeline` gives TurnScope a small fit/transform interface for combining
stateful transformers (such as `TfidfVectorizer`) with pure conversation
features. Steps are named, ordered, and evaluated without mutating source
conversations.

```python
from turnscope import CallableTransformer, FeaturePipeline, TfidfVectorizer, conversation_features

pipeline = FeaturePipeline((
    ("tfidf", TfidfVectorizer(min_document_frequency=2)),
    ("counts", CallableTransformer(conversation_features)),
))
features = pipeline.fit_transform(conversations)
print(features[0].digest())
```

The pipeline requires an explicit fit before transformation and rejects
duplicate step names or duplicate conversation IDs. `FeatureRecord.digest()`
is a canonical cache key for JSON-serializable feature blocks.
