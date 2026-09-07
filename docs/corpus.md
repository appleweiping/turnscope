# Persistent corpora

`CorpusStore` stores native TurnScope conversations in SQLite, without extra
dependencies. It preserves utterance order, metadata and duplicate utterance
IDs. Run the reliability auditor separately to evaluate graph validity.

```python
from turnscope import CorpusStore, iter_path

with CorpusStore("conversations.db") as corpus:
    corpus.put(iter_path("conversations.jsonl"))
    conversations, utterances = corpus.counts()
    page = corpus.ids(limit=100)
    first = corpus.get(page[0]) if page else None
```

Imports are atomic, including when the input iterator fails halfway through.
Existing conversation IDs cause an error unless `replace=True`; duplicate IDs
within a single import always cause an error. Replacements are also rolled back
if any later record fails. Batch memory holds one serialized conversation and a
set of imported conversation IDs; it does not materialize the complete corpus.
JSON input arrays are parsed in memory by the existing reader; use JSONL for
streamed import. A single very large conversation is still materialized.

## Command line

```console
turnscope corpus import conversations.db conversations.jsonl
turnscope corpus stats conversations.db
turnscope corpus list conversations.db --limit 100
turnscope corpus list conversations.db --limit 100 --after previous-last-id
turnscope corpus get conversations.db conversation-id
turnscope corpus export conversations.db exported.jsonl --limit 100
turnscope corpus export conversations.db resumed.jsonl --after previous-last-id
```

All results are JSON on standard output except the exported JSONL records;
export prints a final count object. Errors return exit status 2. Query commands
require an existing database. Export uses keyset ID ordering and an atomic
temporary file, so a failed export does not replace an existing destination.
Import refuses to use its input file as the database, including filesystem
aliases. No delete command is provided; explicit `corpus.delete(ids)` is
available through the Python API.

## Consistency and limits

Use a store on one thread and close it with a context manager. Each write commits
before returning. Other connections see committed changes. SQLite controls
writer contention and raises an error after its default lock timeout; this API
does not retry writes silently. Keyset pages use case-sensitive binary ID order,
not input order. Multiple page calls are not a snapshot across concurrent writes.

The database carries an application ID and schema version. Unsupported databases
are rejected without being migrated. This is local corpus storage, not a network
database or an untrusted-file security sandbox. Back up databases using SQLite's
backup facilities when another process may be writing; avoid copying an active
database without accounting for journal files.

This module does not implement a ConvoKit-compatible corpus model, speaker
vectors, fitted conversational transformers, or research-model evaluation.
Those capabilities remain separate gaps in whole-repository alignment.
