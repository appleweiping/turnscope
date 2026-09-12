# Fixed CGA-WIKI neural data protocol

`benchmarks/neural_cga_data.py` is a read-only local source adapter and partition
protocol. It does not train a model, fit a vocabulary, download data, execute
parses, extract archives, or evaluate model quality. It is a distinct, stricter
loader than the earlier lexical forecasting benchmark.

Source attribution: Cornell/ConvoKit's [Conversations Gone Awry
dataset](https://convokit.cornell.edu/documentation/awry.html). The existing local
`conversations-gone-awry-corpus.zip` is pinned to SHA-256
`84e2d1ac60a3269251b5e175e549fc65cec875fdc926f99172b5b70d3ca1b122`.
Keep the archive, raw IDs/text, vocabulary, and fitted weights private. Source
availability is not a new redistribution grant. This protocol defines our local
experimental split; it does not claim compatibility or equivalence with another
project's full training/evaluation workflow.

## Source fidelity and bounded parsing

The loader hashes one bounded, regular, read-only file handle, checks that pin,
then reads only `conversations.json` and `utterances.jsonl` under the source's
`conversations-gone-awry-corpus/` prefix. Source identity, size, and modification
time are checked around the read. Nothing is fetched or extracted. A final-path
symlink is rejected; this is not a sandbox against a hostile filesystem racing
parent-directory resolution.

The pinned inventory contains 4,188 conversation metadata objects and 30,021
utterances. Metadata is 1,053,900 bytes; the utterance member is 186,487,825 bytes.
The separate roughly 168 MB `info.parsed.jsonl` member, vector file, speakers, and
other unused members are **not expanded**. Do not add a 256 MiB cap over all
declared expanded ZIP members: that would wrongly reject this legitimate archive.
The selected utterance stream has its own 256 MiB ceiling instead.

Default/hard ceilings (callers can lower them) are 128 MiB encoded archive,
64 members, 64 KiB ZIP central directory, 4 MiB metadata, 256 MiB selected
utterance stream, 4 MiB per JSONL row, 10,000 conversations, 50,000 utterances,
1,000,000 JSON nodes per parsed value, depth 32, and 64 fields per non-root object.
The metadata root instead uses the conversation limit. IDs are at most 1,024
UTF-8 bytes; feature text is at most 1 MiB per utterance and 32 MiB in aggregate.
The largest observed source row has 340,145 JSON nodes and depth 8; its inert
parse annotation is not a model feature.

ZIP member-count/directory admission precedes `ZipFile` construction. Required
members must be stored or deflated regular entries and fit declared expanded
bounds; streamed row/total bytes are independently counted. Duplicate names,
encrypted entries, links, and unsafe member names are rejected. These are bounded
encoded/expanded-input checks, not a hard process-RSS guarantee. Parsed JSON trees
and copied native conversation records occupy additional memory.

UTF-8 decoding is strict. A lexical node/depth/object-field pass precedes JSON
decoding; duplicate keys, nonfinite values, integer overflow, invalid Unicode,
and unknown/missing top-level schema fields fail. The sole historical exception
is literal `NaN` at the utterance object's **top-level `reply-to` field**. It is
quarantined as a private sentinel, not turned into a valid edge or general JSON
compatibility policy. `Infinity`, float overflow, and `NaN` anywhere else fail.
The current source contains 11 such legacy parent values.

Every utterance retains its exact text and human boolean
`comment_has_personal_attack` / `is_section_header` annotations. Conversation
labels must agree with the presence of a human attack on a non-header utterance.
Metadata labels are not repaired. Toxicity and embedded `parsed` data must still
be well-formed bounded source JSON but are not used as features. Original speaker
IDs and page titles are also not model features.

Reply relationships are irrelevant to this sequence task. Missing, self, cyclic,
or cross-conversation parent IDs therefore do **not** remove an utterance or
change its label. The adapter does not infer adjacency as a replacement reply
graph. Output `Utterance.reply_to` is unused; all sequence ordering is explicit
ascending finite UTC timestamp, then exact utterance ID. Equal-time blocks remain
subject to the sequence API's no-split prediction-boundary rule.

## Split fixed before eligibility or training

Official `train` (2,508 conversations) and `test` (840 conversations) stay intact.
Only the official `val` inventory (840 conversations) is subdivided:

1. Verify reciprocal, non-self matched pairs and official train/val/test labels.
   A pair or page cannot cross official folds.
2. Across **all** official validation conversations, construct connected
   components under “same page OR reciprocal pair.” This includes every
   conversation that will later be excluded for lacking an eligible prefix.
   An excluded conversation can still connect two otherwise separate page/pair
   groups; removing it first would permit leakage.
3. For each component, sort its exact conversation IDs and serialize the object
   `{"seed": SEED, "conversation_ids": sorted_ids}` as compact sorted-key,
   unescaped-Unicode UTF-8 JSON. `SEED` is fixed to
   `turnscope-neural-v1/2026-09-12`.
4. Convert its SHA-256 hexadecimal digest to an integer and take `% 5`.
   Buckets **0/1** go to policy/threshold validation; **2/3/4** go to
   epoch-selection validation. All component members follow the same assignment.

There is no label-dependent retry, seed search, class rebalance, row sampling, or
post-score partition adjustment. Component allocation is approximately a 40/60
rule, not a guarantee of exact proportions or class counts. If a future declared
input produces unusable class support, the correct response is a disclosed
failure—not silently trying another seed. The entire protocol declaration and
assignment/component inventories are hashed; the aggregate report contains no
raw component membership lists.

## API and causal preparation

```python
# Load the benchmark module from its repository path in a local benchmark script.
data = load_neural_cga(local_archive_path)
training = data.training
epoch_validation = data.validation
policy_validation = data.policy_validation
heldout_test = data.test
aggregate_source_audit = data.audit_dict()
aggregate_prepared_audit = data.prepared_audit()
```

`NeuralCgaPartitions` provides tuple containers of existing `Conversation`
records. Their native metadata behavior is unchanged; callers must not mutate
records during a benchmark. `.audit` and `.audit_dict()` return fresh aggregate
copies. `partition_neural_cga(metadata, records)` is also available for authored
fixtures; it explicitly marks archive origin as **unverified**. Only the pinned
file loader establishes the claimed local archive/member identity.

`prepared_audit(policy=None, limits=None)` calls only
`prepare_sequence_forecasts` and returns its support, group counts, and immutable
partition digests. It asserts pairwise group-set disjointness after preparation,
including group identities retained from excluded conversations. It does not
train or fit a vocabulary. Preparation excludes headers, event/future text, and
the final censored negative turn; a default eligible prefix has at least two
completed turns and strictly precedes the first human event. The preserved raw
record and the causal observed prefix are deliberately different objects.

The aggregate audit records official and derived conversation/class counts,
component bucket counts, source/member hashes, exact prepared support and input
digests. Such ingestion and causal-isolation evidence is **not neural accuracy**.
The final test was previously used by this repository for other task-specific
benchmarks; do not describe it as a universally untouched source dataset. This
new protocol does not use its labels to choose a seed or model hyperparameters.

The fixed source produced 381 validation components (largest: 10 conversations),
with 314 policy-validation and 526 epoch-validation conversations. Default causal
preparation produced the following support; excluded conversations still retain
their grouping influence:

| Partition | Source conversations | Eligible conversations | Eligible positive / negative | Prefixes |
| --- | ---: | ---: | ---: | ---: |
| Training | 2,508 | 2,505 | 1,253 / 1,252 | 10,489 |
| Epoch validation | 526 | 526 | 263 / 263 | 2,243 |
| Policy validation | 314 | 314 | 157 / 157 | 1,258 |
| Final test | 840 | 839 | 420 / 419 | 3,455 |

The exact partition protocol SHA-256 is
`add425152b542f81ecd625bee23d3aa090e7bbc524f17f779b6bd1db1a635327`;
the private validation assignment's aggregate SHA-256 is
`758ea4578e437d3bb8391ecc98728424c4700b0cf44b3fe8ce447795c622afd8`.
The selected utterance member SHA-256 is
`f9305896cb923d761b0f92ef60c25c91feebdd9384de373f153df6b3f5bda5d3`.
These are ingestion/partition observations, not evidence of a trained model.

## Authored adversarial oracles

The independent fixtures check the exact component hash arithmetic, an excluded
bridging conversation, label/row-order invariance, official fold isolation,
reciprocity and label failures, legacy-NaN location, duplicate JSON keys, full
native encoded-size bounds, pre-decoder field/node/depth limits, UTC tie ordering,
and feature-vocabulary isolation. A local ZIP spy confirms only the two required
members are read even when unused expanded bytes exceed the selected-member
budget. No test downloads a corpus or trains a neural model.
