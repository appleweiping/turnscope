# TurnScope benchmarks

`benchmark_fixture.py` runs the parser, search index, and fitted feature
pipeline over the checked-in human-authored conversation fixture. It records
the fixture SHA-256, environment, elapsed time, and traced Python peak. This is
fixture-real evidence, not a claim of equivalence to an external corpus.

```bash
python benchmarks/benchmark_fixture.py
```
