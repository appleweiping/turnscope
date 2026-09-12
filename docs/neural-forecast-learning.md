# Authored turn-order learning regression

This is a narrowly scoped functional-learning test of the hierarchical GRU,
not a natural-language or Conversations Gone Awry quality benchmark. It goes
beyond checking changed parameter bytes: a trained model must distinguish two
declared input orders which are indistinguishable to unigram-only features.

## Protocol fixed before observing outcomes

The positive completed-turn sequence is `alpha`, `beta`, `anchor`; the negative
sequence is `beta`, `alpha`, `anchor`. Each turn contains one ordinary token.
The same final turn prevents a classifier that reads only the latest message
from solving this task. Both source sequences have a fourth turn, `future`;
only the positive fourth turn carries an event label. With `min_turns=3`, each
conversation supplies exactly one eligible prefix ending at `anchor`.
Neither the future text nor its label is an inference feature.

There are 8 training, 4 model-validation, 4 threshold-policy-validation and
6 final held-out conversations. Every partition is balanced, with distinct
conversation IDs and explicit group IDs. The utterance IDs, roles, timestamps,
lengths, token bags and final-turn text do not identify the class. Vocabulary
fitting sees only training observations; all three ordinary tokens have
conversation document frequency 8. `future` must be absent.

The two text patterns intentionally recur across the partitions. Therefore
distinct IDs/groups do **not** make this a test of unseen text patterns or
linguistic generalization. It is a deterministic capacity-and-learning
regression for a simple specified rule.

The single fixed protocol is:

- Initialization/shuffle seed `20260912`, with no seed search.
- Embedding width 8, bidirectional word hidden width 8, causal turn hidden
  width 8, one word layer and one turn layer.
- One ordinary token per turn, explicit reject-on-long-turn policy; EOS is
  still appended by the frozen vocabulary encoder.
- CPU Adam learning rate `0.03`, gradient clipping `5`, full batch size 8.
- At most 80 epochs, patience 20, earliest strictly best model-validation
  checkpoint. Separate policy-validation selects the alert threshold.
- Before inspecting outcomes, acceptance was fixed at every positive held-out
  probability at least `0.75`, every negative at most `0.25`, a minimum score
  gap of `0.5`, held-out balanced accuracy and ROC AUC `1`, and Brier score at
  most `0.0625`.

Default remaining vocabulary, data, numerical and training resource caps are
unchanged. No pretrained weights, model downloads, external providers, paid
calls or real-dataset training are involved. The fixture trains once per test
module. Optional training tests skip when Torch is unavailable; the independent
causality and unigram-control test still runs.

## Controls and interpretation

For every observation the global bag is exactly `{alpha:1, beta:1, anchor:1}`.
Any deterministic model depending only on that bag assigns a constant score;
on the balanced labels either constant alert policy has balanced accuracy
`0.5`. A probability-`0.5` baseline has Brier `0.25`, log loss `ln(2)` and
tie-aware ROC AUC `0.5`. These are arithmetic controls, not a comparison with
a tuned neural or external reference model.

A paired intervention swaps only the first two turn texts while retaining the
same conversation identity, utterance identities, timestamps and final turn.
The learned score and alert must change in the declared direction. Finally,
the selected trained parameters are loaded into an unpadded serial Torch
composition and compared with frozen NumPy inference on both orders. This
checks inference portability of this learned example, not all numerical
inputs or cross-platform bitwise equality.

## Recorded first attempt

The unchanged protocol passed all four tests in 122.30 seconds on the existing
native Python 3.12.0 CPU environment, with PyTorch `2.10.0+cpu` and NumPy
`1.26.4`. All 80 epochs ran; epoch 80 was selected. The separate policy set
selected threshold `0.9941457669199947`. On the six held-out conversation IDs:

| Quantity | Observed result |
| --- | ---: |
| Minimum positive probability | 0.9941457669199947 |
| Maximum negative probability | 0.006222354251178646 |
| True positive / true negative | 3 / 3 |
| False positive / false negative | 0 / 0 |
| Balanced accuracy / ROC AUC | 1 / 1 |
| Conversation-weighted prefix Brier | 0.00003649486869109478 |
| Conversation-weighted prefix log loss | 0.0060566150277264795 |

No failed attempts, changed seeds or revised optimization settings preceded
this result. The vocabulary/control test separately passed without Torch;
the other three tests were explicitly skipped in that dependency-free run.
These are observations from the stated runtime, not guarantees that every
future numerical environment will produce the same probabilities or timings.

The test prints the one attempt's numerical environment, checkpoint, threshold
and final metrics before asserting its acceptance criteria. Failure must not
be silently hidden by searching seeds or widening thresholds. The original
two-pattern rule and repeated small samples cannot establish real event
forecasting utility, calibration, fairness, CRAFT parity or deployment safety.
