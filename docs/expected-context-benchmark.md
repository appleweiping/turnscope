# Movie-disjoint expected-context engineering evaluation

The checked-in [result](../benchmarks/results/expected-context-cornell.json) runs
the fitted [SVD-plus-ridge estimator](expected-context.md) on a locally supplied
official Cornell Movie-Dialogs archive. It is **not a gold annotation benchmark**
or a claim of response quality, authentic reply-tree accuracy, or parity with
another toolkit.

The corpus supplies ordered utterance IDs within dialogues. The experiment
explicitly selects `sequence-successor`: predicting the next listed utterance's
lexical latent representation. No `reply_to` links are fabricated. Movie IDs are
hashed into train/heldout folds before selecting the first 1,000 training and
100 heldout conversations; the two sides contain eight and two films with zero
movie overlap. This is a small deterministic subset, not a representative sample
of all cinema, speakers, or conversational domains.

The pinned archive SHA-256 is
`3bde8a571f615201bc2d2453e22878090719638592f774720eddec739de8c900`.
The loader validates referenced line IDs and film consistency. Parsing, source
URLs, selection hashes, source-module hashes, numerical environment, and model
digest are recorded in the result. The shared loader is
[`benchmark_cornell.py`](../benchmarks/benchmark_cornell.py); its SHA-256 is also
recorded. The archive's README does not provide an explicit redistribution license
grant. Raw dialogue, vocabulary, and fitted parameter files stay local; this
repository publishes only aggregate numbers and hashes, not source text or model
vocabulary. Source availability is not treated as a license grant.

## Results and interpretation

The fixed defaults (256 source/context features, 16 dimensions, ridge penalty 1)
were not tuned on the heldout set:

| Measure | Recorded run |
| --- | ---: |
| Training pairs | 2,419 |
| Heldout positional pairs | 246 |
| Training coordinate MSE | 0.012568 |
| Training-mean baseline, training MSE | 0.014179 |
| Heldout coordinate MSE | 0.012455 |
| Training-mean baseline, heldout MSE | 0.011405 |
| Heldout source token coverage | 65.22% |
| Heldout context token coverage | 67.09% |
| Heldout zero source / context vectors | 22 / 9 |

**The learned predictor is worse than the training-mean baseline on this heldout
subset.** Its lower training error is not evidence of generalization. The test
demonstrates a functioning fitted predictor and honest frozen heldout measurement,
not a successful research result. Small, shifted film samples and a narrow lexical
vocabulary limit interpretation; changing those choices requires a new evaluation
protocol and untouched test data, not selecting a favorable result from this fold.

Every heldout prediction is compared before/after JSON model roundtrip. The
artifact must remain byte-equivalent as a dictionary after evaluation. A separate
loop accumulates coordinate errors and baseline errors for all 246 pairs and
compares them with the public evaluation API. Independent numerical solution
oracles are in `tests/test_expected_context.py`; the benchmark's loss check does
not independently validate its underlying SVD or regression solver.

The recorded 5,275,024-cell preflight estimate covers major dense fitting arrays;
it is not an RSS cap. Timings and `tracemalloc` peaks are in the JSON, with fitting
and evaluation measured separately and parsing, JSON roundtrip, and oracle checks
excluded. Tracemalloc does not necessarily observe all native/BLAS allocation.
Timings vary by machine and instrumentation and are not a comparative speed claim.

## Reproduce

From a source checkout with the development or context extra installed:

```bash
python benchmarks/benchmark_expected_context.py /local/cornell_movie_dialogs_corpus.zip --output /local/expected-context-result.json
```

Only use an archive you are entitled to access. No download, paid API, pretrained
model, telemetry, or external upload is performed. The script checks the pinned
digest and refuses an output path that aliases the archive. Cross-platform BLAS
differences may change last digits and the artifact digest; see the estimator's
SVD sign/degeneracy limitations.
