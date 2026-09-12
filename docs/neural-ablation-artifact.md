# Separate neural-control inference artifacts

`turnscope.neural-ablation-artifact.v1` stores one explicitly named control:
`current-turn.v1`, `mean-word.v1`, or `order-erased.v1`. It never changes the
meaning of a main-model artifact. The main loader rejects all three control
formats, and the control loader rejects the main format. Recomputing checksums
does not bypass the closed mode, inventory, or training-state checks.

This is inference state, **not optimizer resume**. There are no optimizer
moments, executable callbacks, generic object deserialization, automatic model
downloads, or network requests. Loading and prediction need the existing NumPy
dependency; neither imports Torch. The optional native tests fit only small
authored examples, not CGA, and do not measure research-model quality.

## API and private data

```python
from turnscope.neural_ablation_artifact import (
    load_ablation_forecaster,
    save_ablation_forecaster,
)
from turnscope.neural_forecast_artifact import NeuralArtifactLimits

# fitted is an AblationEventForecaster fitted with three separate partitions.
saved = save_ablation_forecaster(fitted, "private-control.tsa")
restored = load_ablation_forecaster(
    saved.path,
    limits=NeuralArtifactLimits(max_file_bytes=32 * 1024 * 1024),
)
assert restored.digest == fitted.digest
```

Saving returns the shared frozen `NeuralSaveResult(path, sha256, bytes_written,
cleanup_warning)`. The SHA-256 identifies the complete file bytes. Default
publication is exclusive: an existing output is an error. Replacement requires
`overwrite=True`. Callers must inspect a possible cleanup warning even when the
artifact was successfully published.

Vocabulary, parameter bytes, training/validation group hashes and training
metadata are private model data. Hashing does not anonymize them. The format
does not include raw source conversations, but that does not make the model
safe to publish; weights and vocabulary may reveal training information. This
implementation adds no automatic public upload or rights/privacy clearance.

## Closed archive and semantic restoration

The eight top-level manifest keys are `format`, `numerical_version`,
`training_version`, `initialization_version`, `model_digest`, `state`, `tensors`,
and `sha256`. `state` has only the complete model `summary`, vocabulary, and
sorted unique training/model-validation/policy-validation group inventories.
The manifest hash covers canonical UTF-8 JSON without its own `sha256` field.

The mode and fixed one-layer, 64-wide reference architecture determine every
active tensor name, shape, and order:

| Control | Numeric members | Numerical identity |
| --- | ---: | --- |
| Current turn | 16 | `current-turn.reset-after-zero-rzn.v1` |
| Mean word | 9 | `mean-word-eos-duplicate.reset-after-rzn.v1` |
| Order erased | 5 | `mean-word-eos.mean-prefix.v1` |

Each tensor descriptor binds its name, member name, little-endian float32 type,
shape, raw byte length, raw data SHA-256, and complete NPY member SHA-256. The
reader derives the expected shape itself and compares the entire canonical NPY
1.0 header byte-for-byte. It does not call `numpy.load`, `torch.load`, pickle,
AST evaluation, or an arbitrary header interpreter. All descriptors, headers
and hashes are checked before copying any numeric member. Parameters must be
finite and within both saved numerical and caller admission limits. The PAD
embedding's 64 stored float32 values must have canonical positive-zero bytes.

Restoration creates new typed immutable configuration, vocabulary, parameters,
training result and model state. It checks:

- Supported tokenizer/Unicode identity, token retention, vocabulary support and
  minimum document frequency; vocabulary size determines the architecture.
- Exactly three distinct fitting partition digests and disjoint group sets,
  including policy validation. The main model's optional two-way reuse is not
  accepted here, even if the supplied checksum is internally consistent.
- Consecutive epochs, nonnegative finite losses/gradient norms, bounded optimizer
  steps, declared early stopping, and the earliest strictly best validation
  epoch. The threshold and balanced-accuracy declaration are bounded, and the
  fixed decision/clipping/calibration declarations must match the model.
- The full canonical untrained main initializer's declared hash inventory and
  the control's active subset agree. Changed tensor names are derived from the
  declared initial hashes versus actual selected tensor bytes, not trusted as a
  free-form list. No initial tensor is regenerated with Torch during loading.
- Active parameter counts and the full canonical main initializer's fit under
  the saved numerical limit, initialization and optimizer workspace lower bounds,
  configured-epoch affine/pooling estimates, minimum source support, and policy
  inference quotas. Minimum work includes at least `prefixes + conversations *
  (min_turns - 1)` unique completed turns, not just `min_turns` per conversation.
  Configured affine and pooling caps retain their separate supported ranges.
  These metadata checks cannot reconstruct every original
  training input length or prove that declared training was executed.
- The entire restored summary and state digest match their canonical manifest
  values, not merely a subset of convenient fields.

Saving runs this same restoration validation **before creating a temporary
output**. Inconsistent manually assembled states cannot be published merely
because their fields have the expected Python class names. Hash-pair ordering
and duplicates are checked before summary construction could collapse them into
JSON objects.

## Bounded storage, publication and failure behavior

The implementation reuses the main artifact's private container primitives;
the public formats and semantic restoration paths remain separate. The shared
hard limits are 64 MiB per file, 4 MiB per manifest, 200,000 JSON nodes, depth 16,
16,000,000 active parameters, and at most 32 ZIP members. This format further
requires exactly 17, 10 or 6 total members, including `manifest.json`. Lower
`NeuralArtifactLimits` are admission checks: they reject, never clamp parameters
or silently replace the saved model's numerical/resource policy or identity.

Only the fixed canonical uncompressed ZIP layout is accepted. Duplicate or
unsafe names, extra/inactive arrays, symlinks, encryption, compression, hidden
members, overlapping offsets, alternate headers, trailing bytes, CRC mismatch,
and extra fields reject. No member is extracted to a filesystem path. Strict
JSON byte, depth, queued-node, integer, finite-value, Unicode and duplicate-key
checks occur before restoration; the manifest must use the exact canonical
encoding. These are bounded allocation/admission policies, not a hard process
RSS or CPU-time sandbox. Reading retains a bounded archive buffer and copying
validated arrays temporarily retains additional owned memory.

Input paths are bounded regular files, not symlinks. `lstat`, opened-descriptor
`fstat`, post-read `fstat`, and final path identity checks detect supported
replacement/change races. Output validation uses the corresponding regular-file
checks, and the prepared temporary file is flushed and `fsync`ed before an
exclusive hard-link publication or explicit atomic replacement. Existing hard
links to a replaced output retain their old bytes. A changed output observed
between validation and replacement rejects; an exclusive create racing another
writer does not overwrite that writer's path.

These operations assume a trusted local directory. They do not provide a
cross-process compare-and-swap lock against a hostile writer in the final
check/replace interval, pin all ancestor directories, fsync the directory, or
claim recovery after every power failure. The same trust limitation applies to
readers faced with an attacker able to rewrite data and restore filesystem
identity metadata. Cryptographic hashes are integrity checks, not signatures.

If publication succeeds but unlinking the private temporary link fails, saving
returns the successful result plus a fixed cleanup warning. It does not claim
that the completed artifact was rolled back. If publication fails, the original
failure is preserved; a second cleanup failure may leave an owned private
temporary file. Callers should keep failure directories private and inspect
them rather than automatically retrying expensive training.

## Provenance refinement before research scoring

The frozen [control design](neural-cga-ablation-design.md) requested source and
protocol SHA binding as part of artifact evidence. The standalone model API
never reads a benchmark source archive or fit driver, so it cannot truthfully
derive those hashes. This implementation binds numerical/training/initialization
versions, declared dependency versions, configuration, fitting partition/group
digests, vocabulary and actual parameter bytes. It does **not** add unverified
source/protocol metadata or pretend a declaration proves training occurred.

For a benchmark candidate, an external candidate-fit receipt must bind the exact
artifact-file SHA-256 and model digest to the actual runtime source hashes,
source archive, frozen protocol, support audit and execution evidence. Artifact
and receipt form the required evidence pair before test scoring. Neither the
standalone artifact nor its self-consistent partition hashes authenticate that
receipt, prove dataset separation, or establish whole-repository parity.
