# Fixed neural ablation arithmetic

`turnscope.neural_ablation_math` implements the three forward functions specified
in [the frozen CGA ablation design](neural-cga-ablation-design.md). This is a
numerical primitive: it does not fit a model, select a threshold, read an artifact,
or establish benchmark quality. The existing main forecaster and artifact format
are unchanged.

## Closed variants

Only a `NeuralArchitecture` with embedding/word/turn widths 64 and one layer at
each recurrent level is supported. Vocabulary size includes PAD 0, UNK 1 and EOS
2. Smaller sequence/vocabulary fixtures are supported; alternate widths, layers,
mode names or undeclared arrays are rejected, not silently approximated.

| Variant | Completed-turn representation and context | Active arrays | Stored parameters |
| --- | --- | ---: | ---: |
| `current-turn.v1` | Original bidirectional word GRU; a separate zero-state turn gate for each completed message | 16 | `64V + 76993` |
| `mean-word.v1` | Mean of retained embeddings including EOS, duplicated to 128 dimensions; original causal turn GRU | 9 | `64V + 39361` |
| `order-erased.v1` | Same 64-dimensional message mean; cumulative mean of completed-message means | 5 | `64V + 2113` |

The current-only gate omits the inactive `turn.weight_hh_l0` matrix, but retains
both bias vectors. In reset-after GRU arithmetic, the recurrent candidate bias
is multiplied by the reset gate even when the previous state is zero. Removing
that bias would implement a different comparison. The classifier remains the
same 64-to-32 tanh layer and scalar output in all modes.

EOS-only messages use the EOS embedding. UNK and repeated retained tokens count
normally; PAD cannot appear in input. Pooling divides by each message's actual
encoded length, not a padded or pre-truncation length. Order-erased context is a
mean of message means, **not** a token-count-weighted global mean. Mean-word's
duplicated halves are redundant, not independently learned directions. These
are capacity and optimization differences, not a perfectly isolated causal
experiment on a single feature.

## API and numerical identity

```python
from turnscope.neural_ablation_math import (
    AblationPoolingLimits,
    FrozenAblationParameters,
    ablation_parameter_shapes,
    admit_ablation_turns,
    infer_ablation_turns,
)
from turnscope.neural_forecast_math import NeuralArchitecture, NeuralNumericLimits

architecture = NeuralArchitecture(vocabulary_size=10003)
shapes = ablation_parameter_shapes("mean-word.v1", architecture)
# `arrays` must supply exactly these names/shapes as plain, C-contiguous,
# little-endian float32 NumPy arrays. This snippet does not initialize or train.
```

`FrozenAblationParameters.from_arrays(variant, architecture, arrays, limits=...,
pooling_limits=...)` checks actual bounded mapping iteration, not a caller's
potentially misleading `keys()` override. It fetches each value once, validates
the entire shape/dtype/contiguity inventory before any parameter copying, then
rechecks and snapshots those same array objects into immutable bytes. Unsynchronized
concurrent mutation of caller arrays is not supported. Returned array views cannot
be made writable. Finite values,
parameter magnitude and active storage counts are checked; the stored PAD row
must have canonical positive-zero bytes. Negative-zero PAD bytes are rejected
by this experimental format even though they have the same numerical value.

The parameter digest binds the variant, its exact numerical version, reference
architecture, ordered active member shapes and byte hashes. Resource policies
are separate configuration, not part of that parameter identity. A digest does
not prove training, source rights or authenticity. This type must not be passed
to the main-model artifact loader or labelled as main-model parameters.

`infer_ablation_turns(parameters, turns, limits=..., pooling_limits=...)` accepts
only a nonempty bounded tuple of nonempty token tuples. Each message ends in
exactly one EOS. It returns immutable logits, uncalibrated sigmoid scores,
message vectors, completed-message states and `AblationInferenceWork`.
Thresholding and eligible endpoints belong to the higher-level workflow.

Inference uses float64 arithmetic over stored float32 values. Importing the
module and inspecting shapes needs neither NumPy nor Torch. Arithmetic needs
the optional NumPy dependency but never imports Torch. It does not mutate
process RNG, thread or dtype settings. Production primitive tests exercise this
Torch-free path; they are not installed-artifact acceptance for a whole trainer.

## Admission and work counters

Input, shape, token, active-parameter, affine-work and workspace admission happens
before numerical arrays are allocated. Default `NeuralNumericLimits` retain the
main model's absolute caps. A stricter parameter-magnitude override rescans actual
stored values only after cheap admission; a lower declared original bound alone
is not evidence that a value violates the new limit.

For U completed messages and T actual encoded positions including EOS, the
reported affine multiplication counts are:

```text
current-turn: 6*64*(64+64)*T + 3*64*128*U + 32*65*U
mean-word:                   3*64*192*U + 32*65*U
order-erased:                            32*65*U
```

These exclude elementwise gates and pooling and are not total FLOPs. The mean
implementations explicitly sum each message from its first embedding then divide
by its real length: `64*(T-U)` additions and `64*U` scalings. Order-erased adds
`64*(U-1)` prefix additions and `64*U` prefix scalings, including the explicit
first division by one. Current-only has zero pooling operations.

`AblationPoolingLimits(max_operations=100_000_000)` separately bounds the sum of
reported pooling additions and scalings per inference; its hard maximum is one
billion. Boolean, fractional, zero and excessive limits reject. It does not
silently spend a smaller model's affine savings on a larger token budget.
Training requires its own all-epoch pooling and initialization admission.

The workspace estimate includes float32 storage, float64 conversion/temporaries,
encoded embeddings, returned vectors/states and gate buffers. It does not bound
Python object overhead, native allocator/BLAS caches or process RSS. Training
must additionally admit the full canonical main initializer before projecting
its untrained parameters into a variant, plus gradients, optimizer state,
selected snapshots and padding-aware activations.

## Mathematical checks and remaining acceptance

Authored tests compare all states and logits with independently written scalar
equations, check the current-only gate against the full GRU at zero state,
exercise nonzero reset-gated biases, and distinguish message means from a global
token mean. They test exact quota boundaries before allocation, malformed
inventories, immutable snapshots and lazy imports. Earlier outputs agree with
independently recomputed prefixes when later messages change.

Current-only ignores earlier content but responds to current content. Mean-word
is invariant to retained within-message permutations but can respond to message
order. Order-erased is invariant to both retained permutations at a fixed prefix,
up to explicitly tested floating-point tolerance; these are not claims of bitwise
permutation invariance for arbitrary values. Retention must happen before pooling:
reordering a long raw message may change its retained head and is outside that
invariance contract.

Separately trained variants, independent gradient checks, fit/save/load deployment,
three fixed seeds, official-test comparison, and the prefix-safe frozen-model
history intervention remain separate acceptance work. Passing numerical tests
alone does not complete that experiment or whole-repository alignment.
