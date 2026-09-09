# TurnScope benchmarks

`benchmark_fixture.py` runs the parser, search index, and fitted feature
pipeline over the checked-in human-authored conversation fixture. It records
the fixture SHA-256, environment, elapsed time, and traced Python peak. This is
checked-in example evidence; it does not establish performance on a real
external corpus. Older result files retain their historical labels.

```bash
python benchmarks/benchmark_fixture.py
```

`benchmark_vectors.py` performs a self-retrieval smoke check on the checked-in
fixture (which contains only one conversation). It also generates
1,000 synthetic training conversations and 50 held-out queries, round-trips the
model, builds an inverted cosine index, and checks the first five queries
against exhaustive cosine enumeration. It records source hashes, corpus shape,
time, traced memory, and whether the model stayed frozen. Generation and the
exhaustive oracle are excluded from the timed work. The fixture is small,
human-authored example data; neither workload establishes real-world retrieval
accuracy or equivalence to a large external corpus.

```bash
python benchmarks/benchmark_vectors.py
```

For external-corpus evidence, `benchmark_cornell.py` reads a locally supplied
[Cornell Movie-Dialogs Corpus archive](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
and verifies its pinned SHA-256 before sampling. It selects the first 1,000
training conversations and 100 queries from deterministic, disjoint movie
partitions, so the same film cannot occur on both sides. It records the archive
and selection digests, movie/utterance counts, sparse retrieval timing/memory,
frozen parameter check, and agreement with exhaustive cosine on five queries.
The output contains numeric evidence and hashes, not dialogue or vocabulary.
The raw archive is neither checked in nor automatically downloaded; its bundled
README does not provide an explicit redistribution license.

```bash
python benchmarks/benchmark_cornell.py /local/cornell_movie_dialogs_corpus.zip \
  --output benchmarks/results/cornell-vectors.json
```

Archive parsing and data selection are outside the timed portion. Timestamps
are fixed placeholders because this source lacks message timestamps; vector
calculations do not use them. No reply links are synthesized from the source's
ordered lines, so this run supplies no reply-coordination evidence. The corpus
contains no retrieval relevance labels; agreement with exhaustive cosine
establishes algorithmic consistency, not usefulness or retrieval accuracy.
This is a bounded real-data subset, not a full-corpus scalability claim.
