# Separately trained neural controls

`turnscope.neural_ablation_train` fits three explicit alternative encoders on
the same immutable causal datasets as the hierarchical forecaster. This is an
optional CPU trainer and a frozen numerical result, not yet an alert-policy
wrapper, persistence format, CLI, CGA result, or claim of whole-project parity.
The matched real-data experiment is specified separately in
[the frozen ablation design](neural-cga-ablation-design.md). Its seed, data and
optimization choices must not be changed in response to eventual test scores.

## API and dependencies

```python
from turnscope.neural_ablation_train import (
    AblationTrainingLimits,
    train_ablation_model,
)
from turnscope.neural_forecast_math import NeuralArchitecture
from turnscope.neural_forecast_train import NeuralTrainingConfig
from turnscope.neural_token_data import fit_sequence_vocabulary

# training and validation are independently prepared SequenceForecastDataset
# values. They include group pins even for source conversations with no examples.
vocabulary = fit_sequence_vocabulary(training)
result = train_ablation_model(
    training,
    validation,
    vocabulary,
    NeuralArchitecture(vocabulary.size),
    variant="mean-word.v1",
    config=NeuralTrainingConfig(seed=17),
    training_limits=AblationTrainingLimits(
        max_total_pooling_operations=100_000_000_000,
    ),
)
```

Training requires the existing optional NumPy/PyTorch CPU dependencies; importing
this module and rejecting invalid input do not import Torch or NumPy. No weights,
dependencies, source datasets, models or providers are downloaded. Use a checkout
or distribution containing these new modules; this document does not assert that
an older release has them.

`train_ablation_model(training, validation, vocabulary, architecture, *, variant,
config=None, numeric_limits=None, data_limits=None, pooling_limits=None,
training_limits=None)` returns immutable `AblationTrainingResult`. It does not
modify a caller's dataset or vocabulary, select a decision threshold, write a
file, resume an optimizer or expose a main-model checkpoint.

The initial version supports exactly embedding/word-hidden/turn-hidden widths
64 and one layer at each reference recurrent level. Vocabulary size includes
PAD, UNK and EOS. Unsupported shapes and unknown versions fail explicitly; small
test vocabulary/sequence sizes are supported without inventing a smaller model.

## Forward meaning and active parameters

| Variant | Learned representation | Active arrays |
| --- | --- | ---: |
| `current-turn.v1` | Bidirectional word GRU, independent zero-state turn gate | 16 |
| `mean-word.v1` | EOS-inclusive mean embedding duplicated to 128, causal turn GRU | 9 |
| `order-erased.v1` | EOS-inclusive mean per turn, mean of completed turns | 5 |

All modes retain the common 64-to-32 `tanh` classifier and scalar output. The
full original inventory has 17 arrays. Current-only removes only
`turn.weight_hh_l0`: multiplying a zero prior state is inactive, whereas the two
turn bias vectors still affect reset/update gates and the gated candidate. The
mean-word model removes both word recurrent directions; fully order-erased
removes both recurrent levels. Excluded arrays are absent from the optimizer and
returned tensors, rather than present with misleading zero-update reports.

For mean pooling, `embedding_bag` receives one flattened sequence of real token
IDs and exact turn offsets, `mode="mean"`, and `include_last_offset=True`.
Every completed turn includes its final EOS; UNK participates, PAD never enters
the input, and an empty ordinary-token message means `E[EOS]`. There are no padded
rows in these segmented means. The mean-word turn sequence is packed by length
and unsorted back to its original conversation. Current-only uses packed word
sequences and restores their original turn order before the independent gates.

The erased representation at endpoint t is
`(m_1 + ... + m_t) / t`, never a whole-conversation mean reused at earlier
endpoints. It is a mean of turn means, not a token-weighted global mean. All
completed-turn outputs are calculated once per batch; loss gathers the original
eligible endpoint handles. Future/event/header text is excluded by the unchanged
data contracts. No prefix catalog is expanded into quadratic copies.

Retained-token permutations leave the two means unchanged mathematically.
Fully erased prefix means also ignore the order of completed turns. Floating
reductions need tolerances, not a bitwise invariance claim. Reordering raw text
before the frozen head-token cap can change the retained bag and is a different
operation. Current-only ignores earlier turns but still learns word order inside
the current completed message.

## Matched initialization and optimization

`make_torch_ablation` constructs the canonical **untrained** main initializer
using its local CPU generator. It records all initial member hashes, then
projects the named active subset and releases excluded parameters before Adam
is created. It never initializes from a trained main model. Common initial
bytes match the main initializer in the same runtime for each seed; skipped
random draws would not have this property.

The result binds `canonical-main-subset-init.v1` and the existing main
`INITIALIZATION_VERSION`. Main and active initial hashes are stored separately,
and selected-byte comparisons determine `changed_parameter_names`. PAD is a
stored canonical positive-zero row and remains unchanged. A nonzero gradient
somewhere in a tensor is not a claim that every element or unused vocabulary row
was trained on that dataset.

The existing `NeuralTrainingConfig` controls all common settings. Defaults remain
8 epochs, patience 3, batch size 16, at most 8,192 real encoded positions per
batch, learning rate 0.001, gradient clip 5, and seed 0 for the general API. The
real-data protocol separately fixes seeds 17, 101 and 202. This API does not pick
a seed from model results.

Adam is CPU float32, beta `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay,
non-fused and non-foreach. No dropout is added. Loss is the mean of each
conversation's mean eligible-prefix BCE; conversations with different numbers
of endpoints have equal weight. Final partial batches are retained. The reported
epoch loss weights a batch by its conversation count; sequential Adam updates
are not equivalent to one full-dataset gradient.

Validation uses the same weighting. The earliest strictly lowest validation
BCE selects a private cloned checkpoint. A tie is not an improvement; the
trainer stops after `patience` non-improving epochs or the configured epoch cap.
It returns the selected checkpoint, not the last candidate's parameters.

The result records typed configuration, mode, all initial hashes, complete
epoch history, selected epoch, actual conversation/prefix support, partition and
vocabulary digests, estimated work, maximum workspace, runtime versions and
observed Torch thread count. Its constructor validates closed tensor identity,
hash inventories, selected-byte change lists, bounded numeric fields, consecutive
history, earliest-best selection and stopping consistency. These integrity
checks cannot authenticate a caller's claimed dataset or prove training occurred.

## Admission, work and failure behavior

Before Torch import or parameter allocation the trainer checks typed values,
fixed architecture, group disjointness, training-only vocabulary (including DF,
retention and UCD identity), both-class support, source/encoding budgets, every
possible seeded epoch batch schedule, and validation batches. The vocabulary is
re-fitted from training for an exact comparison; it is never extended on heldout
data. A rejected source or budget is an exception, not score zero or permission
to silently truncate/drop additional inputs.

The full canonical initializer must also satisfy `numeric_limits.max_parameters`,
even if a variant's active subset would be smaller. Initialization admits at
least `12 * P_main` bytes. Training estimates start with `48 * P_active` bytes
for parameters, gradients, moments and selected/candidate copies, plus
variant-specific word/turn padding and activation buffers and integer
token/offset/length buffers. The maximum of initialization and all training/
validation batch estimates must fit `config.max_workspace_bytes`.
The actual active initial parameters are checked for finite values and the
configured magnitude limit before any forward pass or optimizer construction;
the same checks apply after each update. A tighter magnitude cap may therefore
reject the specified canonical initializer rather than silently rescale it.

Per-conversation numeric and pooling admission uses the frozen numerical
module. Affine multiplication estimates and pooling are separate. For T real
positions and U turns, word means charge `(T-U)*64` additions and `U*64`
scalings. Erased prefix means add `(U-1)*64` additions and `U*64` scalings,
including the explicit first division by one. Current-only charges no mean
pooling. These are logical arithmetic estimates, not measured native-kernel
instructions or total FLOPs.

All possible epochs charge `epochs * (3*training_forward + validation_forward)`
for each counter. The factor three estimates forward and backward work, and the
full configured estimate is retained even if early stopping executes fewer
epochs. Affine work is bounded by the unchanged training config. The new
`AblationTrainingLimits.max_total_pooling_operations` bounds the **sum** of these
addition/scaling estimates, defaults to 100 billion, and has a hard maximum of
10 trillion. Per-inference `AblationPoolingLimits` remains a separate, smaller
quota. Boundary tests cover exact admission and one less than required.

Neither workspace arithmetic nor native package quotas guarantee process RSS.
Python objects, interpreter/allocator overhead, Torch/BLAS caches and native
implementation-specific temporary workspaces are not exhaustively counted.
Failures after private optimizer steps return no candidate and preserve caller
inputs, but consumed CPU time is not rolled back. The trainer never changes
existing Python/NumPy/Torch RNG state, default dtype, thread count or deterministic
flags. Experiment drivers may separately scope/restore an explicit process policy.

## Evidence and limits

`tests/test_neural_ablation_train.py` uses authored unequal-length examples, not
source gold outputs as its sole oracle. Tests compare packed/segmented production
forward and every active gradient against an independently written, unpadded
serial reset-after recurrence. An independent two-step clipped-Adam calculation
covers a final partial batch. Other checks cover three seeds' common initial
hashes, actual three-mode optimization, frozen NumPy selected-weight agreement,
causality/order invariances, earliest-best versus last weights, strict results,
pre-import failures, source preservation and global process settings.

These tests establish implementation contracts on small examples. They do not
establish real-CGA discrimination, calibration, robustness, cross-platform
bitwise training or a complete ablation product. Separate policy thresholding,
safe mode-specific artifact publication/reload and the matched heldout experiment
must be integrated and verified before reporting those capabilities. The different
capacities, redundant duplicated means and altered optimization make these
comparisons informative controls, not perfect causal isolation of word/turn order.
