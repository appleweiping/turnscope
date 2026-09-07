# Conversation search

`ConversationSearchIndex` provides deterministic, in-memory BM25-style retrieval
for utterances. It indexes normalized tokens and posting frequencies, supports
conversation-level filtering and replacement, and returns stable ties by
conversation and utterance ID. The index stores no source text, so applications
can keep sensitive transcripts in their own storage.

```python
from turnscope import ConversationSearchIndex

index = ConversationSearchIndex(conversations)
for hit in index.query("voltage stability", limit=5):
    print(hit.conversation_id, hit.utterance_id, hit.score, hit.matched_terms)
```

Derived indexes can be cached without copying source text. The snapshot is
canonical JSON and `save` returns its SHA-256 digest; `load` revalidates the
schema and rebuilds posting lists before serving queries.

```python
digest = index.save("artifacts/search-index.json")
restored = ConversationSearchIndex.load("artifacts/search-index.json")
assert restored.query("voltage") == index.query("voltage")
```

This is a lexical diagnostic primitive, not a semantic embedding service. For
large corpora, persist the input corpus and rebuild the index with a pinned
tokenizer/version as part of the experiment manifest.
