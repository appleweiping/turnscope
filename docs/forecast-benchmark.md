# Human-annotated CGA-WIKI future-event evaluation

The [recorded result](../benchmarks/results/cga-wiki-forecast.json) evaluates the
[prefix-event workflow](forecasting.md) against supplied human personal-attack
annotations from the official Conversations Gone Awry Wikipedia corpus. This is
an authentic future-event target, not a rule-generated label or an ordering-only
proxy. It remains a lexical baseline experiment, not a neural-model reproduction,
population-calibrated risk estimate, or claim of whole-toolkit equivalence.

## Source and split audit

Official [dataset documentation](https://convokit.cornell.edu/documentation/awry.html)
describes the human per-comment annotation, conversation labels, page IDs, matched
pair IDs, and original train/validation/test splits. The locally supplied ZIP has
SHA-256 `84e2d1ac60a3269251b5e175e549fc65cec875fdc926f99172b5b70d3ca1b122`.
It contains 4,188 conversations and 30,021 utterances. The loader:

- Checks the complete archive digest and all utterance identities.
- Reads source text and supplied boolean human event labels, discarding parsed
  representations and toxicity-model scores rather than using them as features.
- Verifies that each conversation label agrees with its non-header utterance
  annotations and that matched pairs are reciprocal.
- Verifies zero cross-fold page or matched-pair groups: 1,816 page groups and
  2,094 matched-pair groups. No new split is chosen for better scores.
- Sorts by timestamp then ID; equal-time blocks are never split into artificial
  past/future prediction boundaries. It filters 3,833 section headers.

The original folds contain 2,508 training, 840 validation, and 840 test
conversations. After requiring two observed non-header turns, a strictly later
observation, and no prior/simultaneous event, three training conversations and
one test conversation supply no eligible prefix. The final test therefore covers
839 conversations, with 420 positive and 419 negative outcomes. Exclusions are
reported rather than counted as negative predictions.

The archive has no included README or license document establishing a new
redistribution grant. Public availability is not treated as unrestricted
redistribution permission. Raw source, text, and fitted vocabulary/parameters
remain outside this repository; only aggregate metrics and hashes are published.
Follow source terms and underlying content rights for your own use. This repo's
MIT code license does not relicense the dataset.

## Fixed protocol and results

The configuration was fixed before final-test scoring: `alpha=1`, 10,000 features,
minimum two observed turns, and all eligible prefixes with each conversation
sharing total weight one. Train determines vocabulary, likelihoods, and prior.
Only validation chooses the any-alert threshold. Test never influences model or
threshold selection. Parameters are serialized and loaded before test evaluation,
and the frozen artifact is checked unchanged afterwards.

| Measure | Lexical forecaster | Fixed training-prior baseline |
| --- | ---: | ---: |
| Test any-alert balanced accuracy | 0.56035 | 0.50000 |
| Conversation-weighted prefix ROC-AUC | 0.57075 | 0.50000 |
| Conversation-weighted prefix Brier score | 0.37861 | 0.25000 |
| Conversation-weighted prefix log loss | 2.38138 | 0.69315 |

Training uses 2,505 conversations / 10,489 prefixes. Validation uses 840 / 3,501
and selects threshold `0.9909555311574961`, with balanced accuracy `0.57619`.
Final test uses 839 / 3,455. Its confusion counts are TP 179, TN 291, FP 128,
FN 241. For true-positive conversations, the first alert is on average 3.41
non-header turns before the event, counting the event turn itself.

**The model does not outperform the baseline on probability quality.** Its worse
Brier score and substantially worse log loss show overconfidence despite modestly
better ranking and thresholded balanced accuracy. These adverse results are
retained, not optimized away by tuning against the test fold. There are no
confidence intervals or external-domain replication here, so the ranking gain
is not evidence of statistically established superiority.

The corpus is a curated, roughly balanced case-control sample rather than the
natural base rate of attacks on Wikipedia. Negative outcomes describe only the
recorded future and may be censored after collection. Timestamp ordering is used,
not a reconstructed reply tree. These limits preclude claims about calibrated
population risk, intervention benefit, causal influence, or fairness.

Independent tests verify weighted NB likelihood arithmetic, prefix exclusion,
future-text and annotation perturbation, group leakage rejection, weighted AUC
against pairwise enumeration, threshold ties against exact arithmetic, and
malformed artifact rejection. The benchmark itself validates dataset contracts
and the frozen heldout workflow; it is not a second independent implementation
of the model.

## Reproduce without model downloads

From the feature-bearing source checkout:

```bash
python benchmarks/benchmark_forecast.py /local/conversations-gone-awry-corpus.zip --output /local/cga-result.json
```

No paid calls, pretrained model, GPU, automatic download, or external upload is
used. The script verifies the pinned archive and refuses an output alias of it.
Fit timing includes validation policy selection; archive parsing and artifact
serialization are excluded. Recorded `tracemalloc` peaks are about 100 MB for
fit and 26 MB for evaluation and are not process RSS guarantees. The result
records source/script/environment hashes; timings vary with instrumentation and
hardware and are not a comparative performance claim.
