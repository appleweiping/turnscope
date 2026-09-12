# Neural inference artifacts

This development API saves a `HierarchicalEventForecaster` for **inference**, not
optimizer resumption. Loading requires NumPy but never imports Torch, calls
`torch.load`/`numpy.load`, unpickles objects, evaluates a header, extracts archive
members, or fetches model weights. The training implementation remains separate.

```python
from turnscope.neural_forecast_artifact import (
    NeuralArtifactLimits,
    load_neural_forecaster,
    save_neural_forecaster,
)

# model is an already fitted HierarchicalEventForecaster.
saved = save_neural_forecaster(model, "private-model.tsn")
restored = load_neural_forecaster(
    "private-model.tsn",
    limits=NeuralArtifactLimits(max_file_bytes=32 * 1024 * 1024),
)
assert restored.digest == model.digest
```

`NeuralSaveResult` contains the absolute `path`, whole-file `sha256`,
`bytes_written`, and optional `cleanup_warning`. `NeuralArtifactError` denotes an
invalid, internally inconsistent, unsupported, or over-budget artifact. Ordinary
filesystem errors remain filesystem exceptions. A valid file is not necessarily
within a caller's tighter limits: limits reject, never clamp parameters or change
the saved inference policy/model digest.

## Closed format and admission

Format `turnscope.neural-forecast-artifact.v1` is a deliberately restricted ZIP
profile, not an arbitrary ZIP/NumPy import facility:

- `manifest.json` first, followed by `tensors/<logical-name>.npy` in the exact
  architecture-derived parameter order. One/two word layers and one/two turn
  layers imply at most 29 tensor members; no unknown arrays are allowed.
- Only uncompressed `ZIP_STORED`, fixed 1980-01-01 timestamp, regular-file mode
  `0600`, and fixed version/flag fields. No duplicate names, links, traversal,
  encryption, data descriptors, ZIP64, comments, extras, leading/trailing data,
  overlapping members, hidden local entries, or directory/member disagreement.
  Repacking with an ordinary ZIP utility can therefore make a file unsupported.
- Each tensor uses exactly the writer-generated NPY 1.0 header: little-endian
  float32, C order, derived shape, deterministic 64-byte header alignment. The
  loader compares that header literally, rather than parsing Python syntax.
  Exact payload length, member CRC, raw tensor SHA-256, full NPY member SHA-256,
  and aggregate numerical parameter identity are verified.
- The manifest is canonical compact sorted UTF-8 JSON. Duplicate fields,
  unknown fields, nonfinite numbers, surrogate code points, and integers outside
  signed 63-bit magnitude are rejected. A lexical node/depth pass precedes JSON
  decoding; pending tree nodes and encoded bytes are admitted before canonical
  encoding, including escape and multibyte expansion.

Default/hard admission ceilings are 64 MiB for the complete file, 4 MiB for the
complete manifest, 32 ZIP members, 200,000 manifest nodes, depth 16, 16,000,000
float32 parameters, and parameter magnitude `1e6`. All configurable limits can
only be reduced. Architecture and the persisted `NeuralNumericLimits` may impose
stricter bounds; the saved default numerical parameter budget is 4,000,000.
Vocabulary/group inventories must also satisfy saved data/training limits. These
are encoded-data, array, and operation admission limits, **not a hard process RSS
limit**: the file, immutable tensor copies, JSON objects, and NumPy temporaries
can coexist in memory. No archive decompression takes place.

## Restored semantics and what the hashes mean

The manifest stores the exact vocabulary and Unicode tokenizer version, model
configuration, eligibility/data/numerical policies, training configuration,
epoch history, earliest minimum-validation-loss selection, initial/selected
parameter identities, changed-parameter inventory, support counts, work/workspace
estimates, dependency declarations, partition/group digest sets, alert threshold,
and whether policy selection reused model validation. It pins the numerical,
training, and initialization version identifiers. A different Unicode database
tokenizer identity is rejected; do not edit a version string and rehash a file to
pretend it is compatible. Recorded Torch/NumPy training versions are provenance
declarations, not a demand to import that Torch installation during inference.

The loader rebuilds typed objects and checks relationships, including vocabulary
size/document frequencies/token policy, parameter shape/count, disjoint group
declarations, exact validation-reuse support, sequential epochs, early stopping,
earliest-best tie handling, optimizer-step bounds, and changed-array hashes. The
bounds also reject impossible support counts against the source-turn limit and
work estimates below an array-free, EOS-only minimum for the declared support. The
rebuilt complete state must reproduce its declared summary and `model_digest`.
Saving a manually assembled inconsistent state uses this same validator before
publication. The deployment threshold is retained exactly; it is not refitted on
load or on evaluation data. Inference artifacts contain no Adam moments, RNG
cursor, epoch cursor, or future-training resume promise.

Three distinct hashes serve different purposes: the file SHA covers ZIP bytes;
the canonical manifest `sha256` covers the manifest excluding that field; and
`model_digest` identifies the restored model state. The parameter identity also
binds the numerical architecture and every ordered tensor. These checks detect
accidental corruption and internally inconsistent declarations. They are **not
authentication, a signature, or proof that training actually happened**. An
author can create another internally consistent model with different weights
and recompute all hashes. Partition/group digests are declarations with integrity,
not independent proof of training provenance, group disjointness in an unseen
source corpus, or an honestly measured threshold. Verify a trusted whole-file
hash through an independent channel when origin matters.

The artifact contains vocabulary and learned parameters, which can reveal
training information. Treat the entire file, source/group identities, summaries,
and temporary files as private. Do not publish real-data artifacts merely because
they contain no raw dialogue. Repository tests use explicitly authored tiny
states; successful serialization is not model-quality evidence.

## Filesystem and failure behavior

Reads check `lstat` before opening, require a bounded regular file, use
`O_NOFOLLOW` where supported, compare opened-file identity, and recheck identity,
size, and modification time after the bounded read. Nothing is extracted.
Directories, FIFOs, and final-component symlinks are rejected. This is a local
trusted-filesystem boundary, not isolation against a malicious filesystem or
privileged writer racing path resolution. Parent-directory aliases are not a
sandbox; choose a private, trusted output directory.

Saving validates the model before creating a temporary sibling, writes and
flushes/fsyncs the complete file, then atomically publishes it. Default publication
is exclusive (`os.link`); an existing name or a racing creation is not overwritten.
`overwrite=True` explicitly permits `os.replace`, with final-component symlink,
regular-file, and observed identity checks. Replacement changes the directory
entry, not the contents of an old hard-linked inode. There is no fallback to
nonexclusive writing when the filesystem lacks hard-link support.

Failure before publication leaves an existing target unchanged. Cleanup is best
effort; a private temporary sibling can remain if deletion itself fails. If
publication succeeded but temporary cleanup failed, the API returns a successful
`NeuralSaveResult` with a fixed warning instead of falsely reporting that no
artifact was saved. Explicit replacement normally consumes the temporary name.
The API does not claim crash-durable directory metadata on every filesystem;
atomic visible publication is distinct from power-loss durability. It does not
automatically retry writes or coordinate concurrent model training.

## Verification scope

Portable authored tests cover every supported layer combination, exact restored
state and numerical output, byte-identical re-save, Torch-blocked fresh-process
loading, typed/tampered manifests (with both manifest and model hashes recomputed),
NPY header/object/nonfinite rejection, archive metadata/framing corruption,
pre-decoder/pre-encoder budget ordering, stricter caller limits, exclusive and
replacement races, source-model inconsistencies, and postpublication cleanup
warnings. Symlink creation tests explicitly skip where the host lacks permission;
that is not evidence that symlink behavior ran on that host. Actual trained-model
workflow acceptance is reported separately from authored artifact tests.
