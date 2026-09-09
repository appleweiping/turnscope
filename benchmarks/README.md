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
