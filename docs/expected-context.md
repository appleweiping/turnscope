# Fitted expected-context prediction

`ExpectedContextModel` learns which lexical contexts follow (or precede) an
utterance. It fits a low-rank context space and an **SVD-plus-ridge predictor**,
not just a mean of neighboring messages. The predictor is TurnScope's estimator;
it does not reproduce ConvoKit's exact expected-context estimator, clustering,
term-range diagnostics, or dual-context framework. No pretrained model is used.

From a checkout of `feat/whole-repository-alignment`, install
`python -m pip install -e '.[context]'` to fit models; the optional dependency
is NumPy. This API is currently unreleased. Loading, prediction, transformation, and evaluation use the standard
library, including when NumPy is not installed.

## Context means exactly the declared relation

- `reply` (default): one parent-to-reply example for each actual `reply_to` link.
- `predecessor`: the same links reversed, predicting the parent from its reply.
- `sequence-successor`: one example from each utterance to the next **input
  position**. This mode is explicitly positional; it does not create or claim
  ground-truth reply links. Timestamps do not reorder messages.

Replies can branch, and same-speaker edges are retained. Neither roles nor
speaker identities are learned features. Reply modes reject dangling parents
and cycles in the complete conversation, even if a malformed message would
otherwise provide no training example. All modes reject duplicate conversation
IDs and duplicate utterance IDs within a conversation. The positional mode
ignores reply metadata; use the auditor separately to check it. Training pair
sorting makes reply-mode fitting independent of input order within one numerical
environment. Positional contexts naturally change if input order changes.

## Numerical contract

For `N` training pairs, `P` source features, `Q` context features, and `M` distinct
context endpoints:

1. Fit **separate** source and context TF-IDF vocabularies on distinct endpoints.
   A parent with three replies counts as one source document when computing IDF
   but as three paired regression examples. Source/context vocabularies contain
   at most `max_features` entries each. Feature selection and tokenization follow
   [frozen vectors](vectors.md): Unicode word runs, internal apostrophes/hyphens,
   casefolding, and smoothed `log((documents + 1)/(df + 1)) + 1` IDF. This simple
   tokenizer is not a linguistic parser or multilingual segmentation model.
2. Form L2-normalized TF-IDF rows, including zero rows when an endpoint has no
   retained token. Let `X` be the `N × P` paired source matrix and `Y` the `M × Q`
   distinct context matrix. Repeated edges do not duplicate rows for context SVD.
3. Compute a thin SVD of `Y`. Retain `K = min(n_components, numerical_rank(Y))`
   right-singular columns `V`. Rank uses the threshold
   `max(M, Q) * binary64_epsilon * largest_singular_value`. Empty source/context
   vocabularies and rank-zero contexts are errors; constant or rank-deficient
   nonzero training sets remain valid.
4. Project the paired observed contexts to `Z = Y[pair_context_indices] V`.
   Center `X` and `Z` by their **pair-weighted** training means and solve
   `W = (X_centered.T X_centered + regularization I)^-1 X_centered.T Z_centered`.
   The intercept is unpenalized; `regularization` is a positive finite L2 penalty,
   not divided by `N`. Implementation uses a linear solve, not an explicit inverse.
5. Predict `(x - source_mean) W + context_mean`. Project an observed context as
   `y V`. These are signed latent coordinates, not probabilities or generated text.

NumPy fitting uses binary64. Each SVD column's first maximal-absolute loading is
made nonnegative. This fixes sign ambiguity, **not rotations among equal singular
values**. Floating-point roundoff, BLAS/LAPACK versions, and degenerate subspace
bases can change coordinates or artifact bytes across platforms. Pin the numerical
environment and save the model; do not assume cross-platform byte equality.

## Training and heldout separation

```python
from turnscope import ExpectedContextModel
from turnscope.io import load_path

model = ExpectedContextModel(n_components=16, regularization=1.0)
model.fit(load_path("training.jsonl"))
model.save("context-model.json")

frozen = ExpectedContextModel.load("context-model.json")
prediction = frozen.predict("Could you explain the result?")
print(prediction.vector, prediction.coverage)
print(frozen.evaluate(load_path("heldout.jsonl")))
```

`fit` replaces state only after successful validation and fitting. A failed refit
preserves the prior state. No heldout operation updates vocabulary, IDF, basis,
means, or coefficients. `transform(conversation)` predicts every utterance
independently; it does not inspect any reply text. `evaluate` requires actual
observed contexts according to the model's declared relation, which is frozen
in its artifact. Selecting hyperparameters from heldout scores would make that
data validation data, not an untouched final test set.

`ContextPrediction` reports total and known token counts and their ratio. Unknown
tokens have no weight; all-unknown and empty sources map to the **fitted
intercept**, which need not equal the training context mean. All-unknown observed
contexts project to zero. Evaluation includes these examples and reports zero
source/context counts. Low lexical coverage can make a low latent MSE misleading.

Evaluation reports mean squared error averaged across **all pairs and retained
coordinates**, plus the same loss for the fixed pair-weighted training-context
mean. The baseline is never refitted on heldout contexts. These losses assess the
defined lexical representation, not factual accuracy, response quality, causal
influence, or gold annotation accuracy. Changing rank or vocabulary changes the
metric's space, so scores from different model configurations are not directly
comparable.

## CLI and artifacts

```bash
turnscope context fit training.jsonl context-model.json --components 16 --regularization 1
turnscope context transform context-model.json heldout.jsonl --output predictions.json
turnscope context evaluate context-model.json heldout.jsonl --output metrics.json
```

Add `--relation predecessor` or `--relation sequence-successor` to `fit` only when
that context definition is intended. Outputs cannot alias the model or source
input. Transform outputs IDs, vectors, and coverage, but do not repeat raw text.
They are materialized before writing, so malformed later records do not produce
a partial report. The report size is proportional to all selected utterances;
use the Python API to process one conversation at a time for larger inputs.

Versioned JSON stores both frozen TF-IDF models, configuration, SVD basis,
regression coefficients, training summaries, and an integrity checksum. Model
saves are atomic and reject artifacts exceeding the loader's exact 32 MiB byte
limit before modifying the target. Loading validates closed schemas,
duplicate JSON keys, finite numbers, matrix dimensions, orthonormal basis columns,
canonical signs, singular-value ordering, normalized means, and workspace limits.
The checksum detects accidental changes, **not authenticity**; it is not a
signature or a substitute for trusting the model producer. Artifacts include
learned vocabulary and parameters: treat them as potentially sensitive training
derivatives, even though raw messages and endpoint IDs are not stored.

## Bounds and cost

Defaults: 256 source/context features each, 16 requested dimensions, 5,000 pairs,
and 16,000,000 estimated dense float64 cells. Hard configuration limits are 1,024
features, 256 dimensions, 20,000 pairs, and 64,000,000 cells. TF-IDF frequency
counting precedes feature pruning, so `max_features` does **not** bound the raw
vocabulary encountered or input-text memory. Pair counting is bounded, but a
single parsed conversation or huge message can still consume substantial memory.

Before dense allocation, the preflight estimate is
`4 N P + 4 M Q + 3 P² + 3 N K_requested + 2 Q K_requested`, where the requested
rank is first limited by `M` and `Q`. It accounts for major input, centered,
SVD, regression, and solver work arrays. Multiply by eight for an approximate
float64 allocation budget. It is **not a process-memory guarantee**: Python
objects, token maps, allocator overhead, and backend workspaces are additional.

Dense fitting costs roughly `O(M Q min(M,Q) + N P² + P³ + N P K)` after
tokenization. Stored coefficients and basis cost `O((P+Q)K + P+Q)` besides the
vocabularies. A prediction costs `O(text_tokens + P K)`; projection costs
`O(text_tokens + Q K)`. This bounded dense estimator is not appropriate for a
million-word vocabulary or unbounded corpus fit. Numerical failures suggest
increasing regularization, reducing feature/rank limits, or revisiting input.

See [the movie-disjoint heldout run](expected-context-benchmark.md) for engineering
evidence and its explicitly positional, non-gold limitations.
