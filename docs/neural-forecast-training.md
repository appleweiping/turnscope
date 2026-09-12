# CPU training of an ordered hierarchical forecast candidate

`train_sequence_model` trains a private hierarchical GRU candidate from
[ordered causal datasets](neural-forecast-data.md). It is an optional numerical
layer, not a pretrained model service or a complete forecasting application.
It makes no external requests, downloads no weights, compiles no generated
code, and does not load Torch checkpoints or pickle. The existing lexical
forecaster is unchanged.

## Inputs and model

The caller supplies separate `SequenceForecastDataset` training and
model-validation partitions, a `SequenceVocabulary` fitted exclusively on
training observations, and a `NeuralArchitecture`. Both partitions must have
eligible examples from both classes. Any overlap between their hashed source
conversation/group inventories is rejected, including groups retained only
from excluded conversations. No automatic split, resampling or truncation is
performed by training.

The vocabulary is recomputed from the training partition using the configured
`min_document_frequency` and `max_features`, and the vocabulary's pinned
`max_turn_tokens` and `long_turn_policy`. The supplied vocabulary must match
exactly, including tokens, document frequencies, document count and tokenizer
identity. Passing a vocabulary fitted on validation cannot silently introduce
validation-only terms. Inference uses this same frozen vocabulary.

Each complete observed turn is encoded by an embedding and bidirectional word
GRU; its top-layer forward and backward final states are concatenated. A
unidirectional turn GRU consumes these turn representations in source order.
A linear projection to `turn_hidden // 2`, tanh, and a one-unit linear output
produce one logit per complete turn. Training gathers logits only at the
eligible prefix endpoints. Bidirectional word encoding can see the whole
**completed message**, not later messages. Turn recurrence is causal.

The numerical convention is the explicit reset-after r/z/n GRU specified by
`NUMERICAL_VERSION` in `neural_forecast_math.py`. Vocabulary size includes PAD0,
UNK1 and EOS2. PAD is introduced only for packed batching and its embedding
row starts and remains zero. All embedding, word-direction/layer, turn-layer,
projection and output parameters are trainable. This is an original declared
architecture, not a claim to reproduce CRAFT's trained weights, estimator,
preprocessing, pretrained vocabulary or reference scores.

```python
from turnscope.neural_forecast_data import SequenceForecastDataset
from turnscope.neural_forecast_math import NeuralArchitecture, infer_encoded_turns
from turnscope.neural_forecast_train import NeuralTrainingConfig, train_sequence_model
from turnscope.neural_token_data import encode_observed_prefix, fit_sequence_vocabulary


def fit_candidate(training: SequenceForecastDataset, validation: SequenceForecastDataset):
    settings = NeuralTrainingConfig(
        max_features=10_000,
        epochs=8,
        patience=3,
        batch_conversations=16,
        seed=17,
    )
    vocabulary = fit_sequence_vocabulary(
        training,
        min_document_frequency=settings.min_document_frequency,
        max_features=settings.max_features,
        max_turn_tokens=128,
        long_turn_policy="head",
    )
    architecture = NeuralArchitecture(vocabulary.size)
    candidate = train_sequence_model(
        training, validation, vocabulary, architecture, config=settings
    )
    encoded = encode_observed_prefix(validation.observations[0], vocabulary)
    prediction = infer_encoded_turns(candidate.parameters, encoded.turns)
    return vocabulary, candidate, prediction
```

This example does not select an alert threshold or evaluate an untouched test
set. The source observation, supervision and full-source audit digests remain
separate as described in the data contract. Model-selection validation is not
an independent final test.

## Objective, complete batches and checkpoint selection

For a conversation with K eligible prefixes, its loss is the mean of K binary
cross-entropies with logits. A batch loss is the mean of these conversation
losses. Thus a long conversation with ten eligible prefixes does not receive
ten times the objective weight of a conversation with one. Unselected earlier
turn logits are not separately supervised, although gradients can flow through
their recurrent context.

CPU Adam uses the declared learning rate, betas `(0.9, 0.999)`, epsilon `1e-8`,
zero weight decay and non-fused/non-foreach updates. Gradient norm clipping uses
the declared `gradient_clip`. Training rejects non-finite loss/gradient or
non-finite/over-limit resulting parameters instead of returning a zero score.
Every epoch uses a local seeded shuffled conversation order. All conversations
are included; the final partial batch is not dropped. Batches can also split
at the configured sum of real token positions, with no split inside a
conversation. An individually over-budget conversation is rejected.

As in ordinary minibatch optimization, each optimizer step uses its own batch
mean, including a smaller final batch. This is not a claim that sequential
Adam updates equal one full-dataset gradient update. For reporting, batch
losses are weighted by their conversation counts, so a final one-conversation
batch does not have the reporting weight of a full batch. Reported training
losses are gathered before their respective updates at changing parameter
states; they are not a second full pass at the final checkpoint.

Model-validation uses the same per-conversation objective without gradients.
The earliest **strictly lowest** validation loss selects the checkpoint. A
tied or worse loss increments the stale counter; `patience` consecutive
non-improvements stop the run. Returned parameters are copied from the selected
epoch, not simply the last candidate state. No test scores participate.

## Initialization, dependencies and failure isolation

Torch is imported only after typed contracts, split/vocabulary checks,
encoding and resource pre-admission. Training requires an already available
CPU-capable PyTorch and NumPy environment; the data modules and training-module
import do not themselves require Torch. The frozen numerical inference path
uses NumPy without Torch. This implementation does not install dependencies.

Layer construction uses Torch's meta device; parameters are allocated on CPU
and uniformly initialized by an explicitly local CPU `torch.Generator`. The
per-layer bound is `1 / sqrt(layer_dimension)`, as identified by
`INITIALIZATION_VERSION`, and PAD's row is then zeroed. Python shuffling also
uses a local `random.Random`. Training does not reseed the global Python,
NumPy or Torch RNG, set a global default dtype, change the configured thread
count or choose a global deterministic-kernel mode. The actual Torch/NumPy
versions and Torch thread count are reported.

The function creates its own module, optimizer and candidate snapshots. It
does not accept or mutate an existing caller model, change the supplied
datasets/vocabulary, or publish files. Exceptions return no partial trained
candidate. This is local failure isolation, not rollback of external resources
or process-global effects of arbitrary third-party hooks.

Same-seed reproducibility is scoped to compatible runtime, numerical-library
and execution settings; CPU kernels and library upgrades may change floating
point results. The report identifies versions and initialization but does not
promise cross-platform bitwise equality or statistical equivalence.

## Work and allocation admission

`NeuralTrainingConfig` pins vocabulary selection, epochs, patience,
conversation batch size, real token positions per batch, estimated workspace,
estimated total affine multiplications, optimizer settings and seed. Types are
strict; booleans are not integers, and huge/non-finite numeric values are
rejected. Separate `SequenceLimits` and `NeuralNumericLimits` still apply to
source data, each encoded sequence and the closed parameter inventory.

All possible seeded epoch batch schedules are planned before loading Torch.
The workspace estimate includes parameter/optimizer/gradient storage and
padding-dependent word/turn activation allowances. Validation batches are
included. The aggregate work estimate reserves every configured epoch using
`3 × training forward affine work + validation forward affine work` per epoch.
The factor three approximates forward-plus-backward work; it is not a measured
FLOP count. Early stopping may execute fewer epochs, while the report still
records the originally admitted upper estimate. Neither patience nor an
expected early stop is used to evade resource admission.

`max_batch_token_positions` counts real encoded positions, including EOS, not
the padding rectangle. Padding is separately included in the workspace
estimate. Source text and Python tuple/list metadata, Torch import/runtime
overhead, allocator caching and vendor-specific temporary buffers are not
fully bounded by this estimate. These limits are not an OS memory sandbox or
a promise of process RSS, and they are not wall-clock limits.

## Returned evidence and present limits

`NeuralTrainingResult` contains frozen float32 parameters, the exact training
config, immutable epoch history, selected epoch, initial per-parameter hashes,
names of tensors whose selected bytes changed, partition/vocabulary digests,
conversation/prefix counts, allocation/work estimates and numerical environment
versions. Changed tensor bytes demonstrate updates, not useful representations
or generalization. Array access remains backed by immutable byte storage.

Each `NeuralEpoch` reports the training and validation losses, optimizer-step
count, and maximum observed unclipped gradient norm. A few synthetic examples
can verify these mechanics but cannot establish CRAFT parity, authentic event
forecasting quality, fairness, calibration or deployed reliability.

The dedicated training tests use small authored conversations with independently
computed loss weights and gradients, actual two-level/two-layer parameter
updates, partial batches, a controlled early-stop trace with exact selected
checkpoint byte comparison, frozen NumPy inference agreement, and invalid
inputs rejected before dependency import. Actual numerical tests skip when
Torch is unavailable; pre-admission tests remain runnable without Torch.
These fixtures are not a real CGA benchmark. Portable artifact packaging,
threshold selection, final held-out metrics and a complete public forecasting
workflow are separate integration responsibilities.
