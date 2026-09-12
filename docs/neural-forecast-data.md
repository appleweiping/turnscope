# Ordered causal sequence data

This is the data foundation for an ordered neural forecasting workflow. It does
not train a network, load pretrained weights, select a decision threshold, or
measure predictive accuracy. The existing lexical `PrefixEventForecaster` and
its bag-of-words contracts are unchanged. The new modules use the standard
library only.

## Preparation and inference views are different

`prepare_sequence_forecasts(conversations, policy=..., limits=...)` requires
explicit boolean events on every non-header source turn and non-empty explicit
forecast-group IDs on every source conversation. It constructs compact
supervision for the **first future event**, never already-observed events:

- Every example contains at least `SequencePolicy.min_turns` observations
  (default 2).
- Every retained endpoint is the end of a complete equal-timestamp block. No
  prediction is made between simultaneous turns. Input order within a block is
  retained, not interpreted as an independently verified temporal order.
- Positive prefixes end strictly before the first event's timestamp. Event
  text and all later text are absent from the retained observation catalog.
- Negative prefixes must have a later observed turn. The terminal negative
  turn, or terminal equal-time block, is not an eligible prediction point.
- Explicit `skip_field` headers do not contribute observations or lead times.
  Missing skip flags mean `False`; provided flags must be actual booleans.
- A positive `lead_turns` is the first event's index minus the inclusive prefix
  endpoint in the non-header sequence. It is not elapsed seconds.

For non-header turns `[a, b, c, event, after]`, the eligible prefixes under the
default policy are `[a, b]` with lead 2 and `[a, b, c]` with lead 1. For negative
`[a, b, c, last]`, the eligible prefixes are `[a, b]` and `[a, b, c]`. If `c` and
`last` have equal timestamps, only `[a, b]` is eligible.

`observed_prefix(conversation, policy=..., limits=...)` is the deployment
adapter. The caller supplies **only the completed observations available now**.
It applies the explicit header mask but never reads event labels, forecast
groups, roles, reply IDs, token counts, or unrelated metadata. It cannot infer
whether the caller accidentally included future text. An `ObservedPrefix`
contains only a conversation ID and immutable `(id, timestamp, text)` turns.
Timestamps must be timezone-aware and nondecreasing and are normalized to UTC;
utterance IDs must be distinct even on skipped source headers.

## Compact, immutable inventory

`SequenceForecastDataset.observations` has one `ObservedPrefix` per eligible
conversation, stopping at that conversation's **last eligible endpoint**. The
catalog is sorted by conversation ID; turn and token order are never sorted.
`examples` contains `SequenceForecastExample(conversation_index, endpoint,
label, lead_turns)` handles. Endpoints are inclusive and zero-based.

Repeated prefixes do not duplicate growing text or token lists. Use
`dataset.prefix(example)` to materialize a particular prefix on demand; it
validates membership by binary search and shares immutable turn objects. A
training loop can instead walk each conversation once and gather output at the
listed endpoints. `dataset.weights` assigns `1 / number_of_prefixes` to each
example, so every eligible conversation has total weight 1. The raw prefix
count is not a count of independent conversations.

`group_digests` includes **all** supplied conversations and groups, including
those excluded for too few pre-event observations. Conversation and group
names use separate SHA-256 domains compatible with the lexical forecasting
workflow. A training/validation/test coordinator must reject intersecting
group sets before fitting or evaluating; this data module does not combine
or silently repartition caller-provided datasets.

`SequenceAudit` reports the admitted full source, headers, exclusions,
eligible class counts, compact observation counts and raw eligible token
count. Source text/identity bytes include future turns and excluded
conversations because they were still admitted and checked. No source text,
group names, event metadata or labels are retained inside an inference view.

`dataset.observation_digest` binds just the canonical ordered observation
catalog. `dataset.digest` also binds example labels, lead times, hashed groups,
and the full-source audit. Changing future text can change audit byte counts
and the latter digest without changing the observation digest, vocabulary, or
encoded model input. These hashes are consistency checks, not authentication
of the original labels, timestamps, source, or statistical independence.

## Frozen vocabulary and integer encoding

```python
from datetime import datetime, timedelta, timezone

from turnscope.models import Conversation, Utterance
from turnscope.neural_forecast_data import observed_prefix, prepare_sequence_forecasts
from turnscope.neural_token_data import encode_observed_prefix, fit_sequence_vocabulary

base = datetime(2026, 1, 1, tzinfo=timezone.utc)
source = Conversation(
    "training-conversation",
    [
        Utterance(
            str(index),
            "participant",
            text,
            base + timedelta(seconds=index),
            metadata={"event": index == 2},
        )
        for index, text in enumerate(["Please explain", "I disagree", "future event"])
    ],
    metadata={"forecast_groups": ["discussion-17"]},
)
training = prepare_sequence_forecasts([source])
vocabulary = fit_sequence_vocabulary(
    training, max_features=10_000, max_turn_tokens=128, long_turn_policy="head"
)
query = Conversation("observed-now", source.utterances[:2])
encoded = encode_observed_prefix(observed_prefix(query), vocabulary)
assert encoded.vocabulary_size == vocabulary.size == len(vocabulary.tokens) + 3
assert all(turn[-1] == 2 for turn in encoded.turns)
```

The tokenizer matches the existing lexical regex
`[\w]+(?:['-][\w]+)*` with Unicode word characters, then casefolds each match.
It does not remove accents, normalize Unicode forms, stem, or use a learned
tokenizer. Combining characters can split source matches; casefold itself can
introduce combining characters (for example U+0130). Both fitting and encoding
use exactly this procedure.

`TOKENIZER_VERSION` includes the runtime Unicode database version. Vocabulary
deserialization rejects a different tokenizer/UCD version. For example the
classification of U+1C89 differs between Unicode 15 and 16, so the same raw text
could otherwise tokenize differently under an unchanged vocabulary. Do not
relabel a saved version to force acceptance. Refit and revalidate deliberately
on the target tokenizer when migrating Unicode versions.

`SequenceVocabulary.tokens` contains **ordinary tokens only**, sorted
lexicographically. Ordinary token `tokens[i]` has ID `i + 3`:

| ID | Meaning | Stored in an encoded turn? |
| --- | --- | --- |
| 0 | PAD | No; batching belongs to the numerical layer. |
| 1 | UNK | Yes, for an ordinary token absent from the training vocabulary. |
| 2 | EOS | Exactly once, last, including for empty text. |
| 3 onward | Frozen ordinary vocabulary | Yes, in original order. |

Document frequency counts each retained ordinary token at most once per
eligible **conversation**, not per turn or expanding prefix. Selection uses
descending DF and lexical tie breaking, after `min_document_frequency`.
`max_features` limits selected ordinary tokens, not the three special rows.
An eligible EOS-only corpus can have an empty ordinary vocabulary; zero
eligible conversations cannot fit one.

`max_turn_tokens` (default 128, hard maximum 4096) counts ordinary tokens;
EOS adds one more integer per turn. The vocabulary permanently pins one of:

- `head`: retain the first N ordinary tokens of each turn. Only retained
  tokens contribute DF and vocabulary candidates. All raw eligible tokens
  are still scanned, counted, and checked for maximum token byte length.
- `reject`: raise `SequenceLimitError` when any turn has more than N tokens.

Encoding cannot override this policy. It applies the same rule to held-out
observations, does not update the vocabulary, and reports raw, retained,
known-token and truncated-turn counts. These are lexical accounting values,
not confidence or model accuracy. An empty turn is `(2,)`; an unknown
one-word turn is `(1, 2)`.

## Resource admission and serialization

`SequenceLimits` validates strict positive integers, rejecting booleans,
non-finite/fractional values and values above the hard ceilings. Invalid
options fail before consuming the source iterable. Defaults include 10,000
source conversations, 100,000 total source turns, 64 observed turns per
eligible conversation, 30,000 prefix handles, 32 MiB of full-source text and
identity UTF-8 bytes, 4 MiB per source turn, 2 million raw eligible tokens,
100,000 distinct retained candidate tokens, and 1,024 UTF-8 bytes per token.
The corresponding absolute ceilings are 100,000 conversations, 1 million
source turns, 4,096 observed turns, 100,000 handles, 128 MiB source bytes,
8 MiB per turn, 10 million raw tokens, 500,000 candidates and 4,096 token bytes.
Identity and group budgets are also explicit fields of `SequenceLimits`.

There is no automatic conversation sampling or turn dropping to meet these
limits. Source-byte admission includes conversation IDs, turn IDs and every
declared group occurrence, plus all source text, including skipped/future
text. Raw token work admission applies to the compact eligible observation
catalog, not excluded/future text and not quadratic expanded prefixes.
Candidate admission happens before frequency filtering and feature selection;
tokens explicitly omitted by the pinned head policy are not candidates.

`to_dict`/`from_dict` use closed versioned JSON-shaped envelopes for datasets,
vocabularies and encoded prefixes, with strict typed immutable nested values.
Observation, policy and audit records use exact closed fields. Duplicate
handles/IDs/tokens, invalid timestamps/Unicode, malformed shapes, impossible
lead accounting, mismatched counts, PAD-in-sequence and misplaced EOS are
rejected. Raw aggregate inventories are checked before constructing expanded
typed catalogs. These functions accept already-parsed objects: callers loading
an untrusted JSON file or HTTP body must separately bound its bytes and parse
depth before parsing. There is no pickle or automatic file loading here.

Serialized datasets preserve the admitted compact features and audit, not the
omitted original source. Deserialization verifies internal consistency but
cannot prove the first-event labels, complete eligibility, or source audit
against source text which is intentionally absent. Keep separately pinned
source provenance and the preparation policy in any training artifact.

Preparation requires O(source text + raw eligible token work + conversation
sorting + prefix handles) time. Each validation or fitting phase can rescan
bounded text; this is not a single-pass tokenizer claim. Vocabulary fitting
adds O(candidate count log candidate count) selection work and O(candidate
count) counting space. Encoding preserves sequence order with linear raw
token work and at most N + 1 output integers per turn. The main dataset stores
one turn catalog plus compact handles, rather than O(prefix count × history)
copied features. Explicit requests to materialize or encode every prefix
can still repeat history work, so a trainer should gather causal outputs from
one conversation pass.

Byte budgets measure source strings, not total Python object memory, escaped
JSON size, tokenizer temporary strings, tensor activations, or process RSS.
No numerical workspace budget is claimed by this layer. The standard JSON
encoder hashes in fragments, but may allocate an escaped individual string
fragment before hashing it. Caps are admission contracts, not an OS memory
sandbox.

## Verification and limits

The authored unit cases include an exhaustive independent short-sequence
eligibility formula (also compared with the unchanged lexical workflow),
simultaneous-time/header controls, full-source budget boundaries, group pins
for excluded conversations, future-text mutations, exact hand-computed DF,
order-sensitive token IDs, empty/OOV sequences, frozen roundtrips, malformed
payloads and pre-copy resource probes. None are a gold-label model evaluation.
The module neither claims CRAFT estimator parity nor a complete neural
forecasting product; training, portable parameters, calibrated validation and
held-out engineering/quality evaluation are separate layers.
