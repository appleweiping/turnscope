from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from turnscope.forecast_data import prepare_forecast_examples
from turnscope.models import Conversation, Utterance
from turnscope.neural_forecast_data import (
    TOKENIZER_VERSION,
    ObservedPrefix,
    ObservedTurn,
    SequenceAudit,
    SequenceDataError,
    SequenceForecastDataset,
    SequenceForecastExample,
    SequenceLimitError,
    SequenceLimits,
    SequencePolicy,
    observed_prefix,
    prepare_sequence_forecasts,
)
from turnscope.neural_token_data import (
    EOS_ID,
    PAD_ID,
    UNK_ID,
    EncodedPrefix,
    SequenceVocabulary,
    encode_observed_prefix,
    fit_sequence_vocabulary,
)
from turnscope.transformers import _tokens

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def conversation(identifier="c", texts=("a", "b", "future"), *, event=2, times=None):
    stamps = range(len(texts)) if times is None else times
    return Conversation(
        identifier,
        [
            Utterance(
                str(index),
                "speaker",
                text,
                BASE + timedelta(seconds=stamp),
                metadata={"event": index == event},
            )
            for index, (text, stamp) in enumerate(zip(texts, stamps, strict=True))
        ],
        metadata={"forecast_groups": ["group:" + identifier]},
    )


def dataset():
    return prepare_sequence_forecasts(
        [
            conversation("p", ("red red", "blue", "red", "secret"), event=3),
            conversation("n", ("blue", "green", "other"), event=None),
        ]
    )


def digest(kind, identifier):
    return hashlib.sha256((kind + "\0" + identifier).encode()).hexdigest()


def test_hand_eligibility_catalog_weights_and_roundtrip():
    value = dataset()
    assert [prefix.conversation_id for prefix in value.observations] == ["n", "p"]
    assert [tuple(turn.text for turn in prefix.turns) for prefix in value.observations] == [
        ("blue", "green"),
        ("red red", "blue", "red"),
    ]
    assert [example.to_dict() for example in value.examples] == [
        {"conversation_index": 0, "endpoint": 1, "label": False, "lead_turns": None},
        {"conversation_index": 1, "endpoint": 1, "label": True, "lead_turns": 2},
        {"conversation_index": 1, "endpoint": 2, "label": True, "lead_turns": 1},
    ]
    assert value.weights == (1, 0.5, 0.5)
    assert value.audit.observed_turns == 5
    assert value.audit.source_turns == 7
    assert value.audit.raw_tokens == 6
    assert value.audit.positive_conversations == value.audit.negative_conversations == 1
    assert len(value.prefix(value.examples[1]).turns) == 2
    clone = SequenceForecastDataset.from_dict(json.loads(json.dumps(value.to_dict())))
    assert clone == value and clone.digest == value.digest
    assert ObservedPrefix.from_dict(value.observations[0].to_dict()) == value.observations[0]
    assert SequencePolicy.from_dict(SequencePolicy().to_dict()) == SequencePolicy()
    assert SequenceLimits.from_dict(SequenceLimits().to_dict()) == SequenceLimits()
    assert SequenceAudit.from_dict(value.audit.to_dict()) == value.audit


@pytest.mark.parametrize("length", range(1, 9))
def test_all_short_future_positions_against_independent_formula_and_legacy(length):
    for event in [None, *range(length)]:
        source = conversation(texts=tuple(str(index) for index in range(length)), event=event)
        value = prepare_sequence_forecasts([source])
        # Inclusive endpoints j must contain >=2 turns and precede a future turn.
        final = length - 1 if event is None else event
        expected = [
            (j, event is not None, None if event is None else event - j) for j in range(1, final)
        ]
        actual = [(row.endpoint, row.label, row.lead_turns) for row in value.examples]
        assert actual == expected
        legacy = prepare_forecast_examples([source])
        assert [
            (row.prefix.turns - 1, row.label, row.lead_turns) for row in legacy.examples
        ] == expected
        assert value.audit.eligible_conversations == bool(expected)
        assert value.audit.excluded_conversations == (not expected)
        assert digest("conversation", "c") in value.group_digests
        assert digest("group", "group:c") in value.group_digests


@pytest.mark.parametrize(
    "event,times,endpoints",
    [
        (3, [0, 1, 1, 2], [2]),
        (2, [0, 1, 1], []),
        (4, [0, 1, 2, 2, 2], [1]),
        (None, [0, 1, 2, 2], [1]),
        (None, [0, 0, 0], []),
    ],
)
def test_equal_time_blocks_are_atomic(event, times, endpoints):
    source = conversation(texts=tuple("a" for _ in times), event=event, times=times)
    value = prepare_sequence_forecasts([source])
    assert [row.endpoint for row in value.examples] == endpoints
    assert (
        len(value.observations[0].turns) == endpoints[-1] + 1
        if endpoints
        else not value.observations
    )


def test_headers_ignored_for_labels_and_leads_but_admitted_and_never_copied():
    source = conversation(texts=("HEADER", "a", "b", "HEADER2", "event"), event=4)
    for index in [0, 3]:
        source.utterances[index].metadata.clear()
        source.utterances[index].metadata["is_section_header"] = True
    value = prepare_sequence_forecasts([source])
    assert tuple(turn.text for turn in value.observations[0].turns) == ("a", "b")
    assert value.examples[0].lead_turns == 1
    assert value.audit.header_turns == 2 and value.audit.source_turns == 5
    assert value.audit.source_text_bytes == len("HEADERabHEADER2event")
    assert fit_sequence_vocabulary(value).tokens == ("a", "b")


def test_frozen_feature_identity_excludes_future_metadata_outcomes_and_group_pins():
    source = conversation(texts=("red", "blue", "attack", "after"), event=2)
    before = prepare_sequence_forecasts([source])
    altered = Conversation(
        source.id,
        [
            *source.utterances[:2],
            replace(source.utterances[2], text="a much longer future attack"),
            replace(source.utterances[3], text="another future word"),
        ],
        source.metadata,
    )
    altered.metadata["forecast_groups"].append("extra-group")
    after = prepare_sequence_forecasts([altered])
    assert after.observation_digest == before.observation_digest
    assert after.digest != before.digest  # truthful whole-source counters and leakage pins
    assert fit_sequence_vocabulary(after) == fit_sequence_vocabulary(before)
    # Deployment views read only the explicit observation mask, not labels/groups.
    raw = Conversation("query", source.utterances[:2], {"forecast_groups": object()})
    first = observed_prefix(raw)
    raw.utterances[0].metadata["event"] = {"secret": "anything"}
    raw.utterances[1].metadata.clear()
    assert observed_prefix(raw) == first
    assert set(first.to_dict()) == {"conversation_id", "turns"}
    assert all(set(turn.to_dict()) == {"id", "timestamp", "text"} for turn in first.turns)


def test_input_order_canonical_conversations_but_observed_turn_and_token_order_preserved():
    sources = [conversation("z"), conversation("a")]
    first = prepare_sequence_forecasts(sources)
    second = prepare_sequence_forecasts(reversed(sources))
    assert first == second and first.digest == second.digest
    left = observed_prefix(conversation(texts=("red blue", "green"), event=None))
    right = observed_prefix(conversation(texts=("blue red", "green"), event=None))
    vocab = fit_sequence_vocabulary(
        prepare_sequence_forecasts([conversation(texts=("red blue", "green", "last"), event=None)])
    )
    assert Counter(left.turns[0].text.split()) == Counter(right.turns[0].text.split())
    assert encode_observed_prefix(left, vocab).turns != encode_observed_prefix(right, vocab).turns
    assert left.digest != right.digest


def test_caller_and_export_mutations_cannot_change_snapshots():
    source = conversation()
    value = prepare_sequence_forecasts([source])
    original = value.digest
    source.metadata["forecast_groups"].append("later")
    source.utterances[0].metadata["event"] = True
    exported = value.to_dict()
    exported["observations"][0]["turns"][0]["text"] = "changed"
    exported["audit"]["source_turns"] = 999
    assert value.digest == original
    with pytest.raises(FrozenInstanceError):
        value.observations[0].turns[0].text = "changed"


@pytest.mark.parametrize("value", [True, 0, -1, 10**400, 1.0, None, "2"])
def test_strict_limits_and_policies(value):
    with pytest.raises(SequenceDataError):
        SequenceLimits(max_tokens=value)
    with pytest.raises(SequenceDataError):
        SequencePolicy(min_turns=value)


def test_invalid_options_fail_before_consuming_source():
    def forbidden():
        raise AssertionError("source consumed")
        yield

    with pytest.raises(SequenceDataError):
        prepare_sequence_forecasts(forbidden(), limits={})
    with pytest.raises(SequenceDataError):
        prepare_sequence_forecasts(forbidden(), policy={})
    with pytest.raises(SequenceLimitError):
        prepare_sequence_forecasts(forbidden(), limits=SequenceLimits(max_observed_turns=1))
    with pytest.raises(SequenceDataError):
        SequencePolicy(event_field="same", skip_field="same")


@pytest.mark.parametrize(
    "groups", [None, [], (), "x", ["x", "x"], [True], [None], [""], ["\ud800"]]
)
def test_group_validation_even_for_excluded_conversations(groups):
    source = conversation(event=0)
    source.metadata["forecast_groups"] = groups
    with pytest.raises(SequenceDataError):
        prepare_sequence_forecasts([source])


@pytest.mark.parametrize("key,value", [("event", 1), ("event", None), ("is_section_header", 1)])
def test_strict_supervision_and_masks(key, value):
    source = conversation()
    source.utterances[0].metadata[key] = value
    with pytest.raises(SequenceDataError):
        prepare_sequence_forecasts([source])


def test_source_identity_timestamp_and_unicode_validation():
    with pytest.raises(SequenceDataError, match="unique"):
        prepare_sequence_forecasts([conversation(), conversation()])
    source = conversation()
    duplicate = replace(source.utterances[1], id="0")
    with pytest.raises(SequenceDataError, match="unique"):
        prepare_sequence_forecasts(
            [Conversation("c", [source.utterances[0], duplicate], source.metadata)]
        )
    with pytest.raises(SequenceDataError, match="nondecreasing"):
        prepare_sequence_forecasts([conversation(times=[1, 0, 2])])
    with pytest.raises(SequenceDataError, match="Unicode"):
        prepare_sequence_forecasts([conversation(texts=("a", "b", "\ud800"))])
    with pytest.raises(SequenceDataError, match="timezone"):
        ObservedTurn("x", datetime(2026, 1, 1), "a")
    with pytest.raises(SequenceDataError):
        ObservedTurn("x", "not-a-timestamp", "a")
    turn = ObservedTurn("x", BASE.astimezone(timezone(timedelta(hours=3))), "a")
    assert turn.timestamp == BASE
    with pytest.raises(SequenceDataError):
        ObservedTurn.from_dict({"id": "x", "text": "a", "timestamp": "not-a-date"})


def test_exact_source_and_work_budgets_include_excluded_and_future():
    source = conversation()
    value = prepare_sequence_forecasts([source])
    exact = value.audit.source_text_bytes + value.audit.source_identity_bytes
    assert (
        prepare_sequence_forecasts([source], limits=SequenceLimits(max_source_bytes=exact)) == value
    )
    with pytest.raises(SequenceLimitError):
        prepare_sequence_forecasts([source], limits=SequenceLimits(max_source_bytes=exact - 1))
    for limits in [
        SequenceLimits(max_source_turns=2),
        SequenceLimits(max_turn_bytes=5),
        SequenceLimits(max_tokens=1),
        SequenceLimits(max_groups=1),
    ]:
        with pytest.raises(SequenceLimitError):
            prepare_sequence_forecasts([source], limits=limits)
    with pytest.raises(SequenceLimitError):
        prepare_sequence_forecasts([conversation(event=0)], limits=SequenceLimits(max_turn_bytes=5))
    with pytest.raises(SequenceLimitError):
        prepare_sequence_forecasts(
            [source, conversation("d")], limits=SequenceLimits(max_conversations=1)
        )
    with pytest.raises(SequenceLimitError):
        prepare_sequence_forecasts(
            [conversation(texts=("a", "b", "c", "d"), event=3)],
            limits=SequenceLimits(max_prefixes=1),
        )


def test_observed_turn_limit_rejects_instead_of_silent_drops():
    source = conversation(texts=("a", "b", "c", "d"), event=3)
    with pytest.raises(SequenceLimitError, match="observed-turn"):
        prepare_sequence_forecasts([source], limits=SequenceLimits(max_observed_turns=2))
    with pytest.raises(SequenceLimitError, match="observed-turn"):
        observed_prefix(source, limits=SequenceLimits(max_observed_turns=2))
    with pytest.raises(SequenceDataError, match="min_turns"):
        observed_prefix(conversation(texts=("a",), event=None))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(extra=True),
        lambda row: row.update(format="future"),
        lambda row: row["examples"][0].update(endpoint=True),
        lambda row: row["examples"][0].update(label=1),
        lambda row: row["examples"][0].update(lead_turns=1),
        lambda row: row["examples"][1].update(lead_turns=0),
        lambda row: row["examples"][1].update(lead_turns=3),
        lambda row: row["examples"][1].update(conversation_index=99),
        lambda row: row["examples"][1].update(endpoint=10),
        lambda row: row["examples"].reverse(),
        lambda row: row["examples"].pop(),
        lambda row: row["observations"].reverse(),
        lambda row: row["group_digests"].append(row["group_digests"][0]),
        lambda row: row["group_digests"].clear(),
        lambda row: row["audit"].update(observed_turns=99),
        lambda row: row["audit"].update(raw_tokens=99),
        lambda row: row["audit"].update(source_turns=True),
    ],
)
def test_closed_dataset_corruption_rejected(mutate):
    data = dataset().to_dict()
    mutate(data)
    with pytest.raises(SequenceDataError):
        SequenceForecastDataset.from_dict(data)


def test_direct_dataset_guards_ties_unused_observations_and_fake_members():
    value = dataset()
    changed = replace(
        value.observations[1],
        turns=(
            value.observations[1].turns[0],
            replace(value.observations[1].turns[1], timestamp=BASE + timedelta(seconds=2)),
            value.observations[1].turns[2],
        ),
    )
    with pytest.raises(SequenceDataError, match="equal-time"):
        replace(value, observations=(value.observations[0], changed))
    with pytest.raises(SequenceDataError, match="every observation"):
        replace(value, examples=(value.examples[0],))
    for foreign in [
        None,
        SequenceForecastExample(1, 1, True, 3),
        SequenceForecastExample(0, 0, False, None),
    ]:
        with pytest.raises(SequenceDataError, match="member"):
            value.prefix(foreign)


def test_raw_aggregate_admission_precedes_typed_copy(monkeypatch):
    import turnscope.neural_forecast_data as module

    value = dataset().to_dict()
    value["audit"].update(source_text_bytes=10, observed_text_bytes=10, source_identity_bytes=0)
    monkeypatch.setitem(module._HARD, "max_source_bytes", 10)

    def forbidden(*args, **kwargs):
        raise AssertionError("typed turn copies before raw admission")

    monkeypatch.setattr(ObservedTurn, "from_dict", forbidden)
    with pytest.raises(SequenceLimitError, match="source-byte"):
        SequenceForecastDataset.from_dict(value)


def test_direct_aggregate_turn_admission_precedes_token_scan(monkeypatch):
    import turnscope.neural_forecast_data as module

    value = dataset()

    def forbidden(*args, **kwargs):
        raise AssertionError("tokenization before aggregate turn admission")

    monkeypatch.setattr(module, "_iter_tokens", forbidden)
    with pytest.raises(SequenceLimitError, match="source inventory"):
        value.validate_limits(SequenceLimits(max_source_turns=4))
    with pytest.raises(SequenceDataError, match="audit"):
        replace(value, audit=replace(value.audit, observed_turns=4))


def test_frozen_vocabulary_independent_document_frequencies_and_policy():
    value = dataset()
    vocabulary = fit_sequence_vocabulary(value)
    assert vocabulary.tokens == ("blue", "green", "red")
    assert vocabulary.document_frequencies == (2, 1, 1)
    assert vocabulary.documents == 2 and vocabulary.size == 6
    assert vocabulary.token_id("blue") == 3 and vocabulary.token_id("unknown") == 1
    # Frequency, then lexical tie-break: blue wins; green wins tie over red.
    smaller = fit_sequence_vocabulary(value, max_features=2)
    assert smaller.tokens == ("blue", "green")
    assert fit_sequence_vocabulary(value, min_document_frequency=2).tokens == ("blue",)
    assert SequenceVocabulary.from_dict(vocabulary.to_dict()) == vocabulary
    assert SequenceVocabulary.from_dict(vocabulary.to_dict()).digest == vocabulary.digest
    totals = defaultdict(float)
    for example, weight in zip(value.examples, value.weights, strict=True):
        totals[example.conversation_index] += weight
    assert list(totals.values()) == [1, 1]


def test_head_policy_trains_only_retained_words_but_scans_all_raw_tokens():
    source = conversation(texts=("alpha tail tail", "beta tail", "future"))
    value = prepare_sequence_forecasts([source])
    vocabulary = fit_sequence_vocabulary(value, max_turn_tokens=1)
    assert vocabulary.tokens == ("alpha", "beta")
    assert vocabulary.document_frequencies == (1, 1)
    encoded = encode_observed_prefix(value.observations[0], vocabulary)
    assert encoded.turns == ((3, EOS_ID), (4, EOS_ID))
    assert (
        encoded.raw_tokens,
        encoded.retained_tokens,
        encoded.known_tokens,
        encoded.truncated_turns,
    ) == (5, 2, 2, 2)
    assert vocabulary.max_turn_tokens == 1 and vocabulary.long_turn_policy == "head"
    assert encoded.vocabulary_size == vocabulary.size
    assert EncodedPrefix.from_dict(encoded.to_dict()) == encoded
    with pytest.raises(SequenceLimitError, match="token budget"):
        encode_observed_prefix(
            value.observations[0], vocabulary, limits=SequenceLimits(max_tokens=4)
        )
    with pytest.raises(SequenceLimitError, match="pinned token"):
        fit_sequence_vocabulary(value, max_turn_tokens=1, long_turn_policy="reject")
    fitted_reject = fit_sequence_vocabulary(dataset(), max_turn_tokens=2, long_turn_policy="reject")
    with pytest.raises(SequenceLimitError, match="pinned token"):
        encode_observed_prefix(value.observations[0], fitted_reject)


def test_tokenizer_compatibility_exact_unicode_and_eos_unk():
    text = "ONE one-one can't _x 42 Straße \u0130 e\u0301 中文 \U0001f600"
    source = conversation(texts=(text, "", "future"))
    value = prepare_sequence_forecasts([source])
    vocabulary = fit_sequence_vocabulary(value)
    expected = tuple(sorted(set(_tokens(text))))
    assert vocabulary.tokens == expected
    encoded = encode_observed_prefix(value.observations[0], vocabulary)
    assert encoded.turns[0] == (*(vocabulary.token_id(token) for token in _tokens(text)), EOS_ID)
    assert encoded.turns[1] == (EOS_ID,)
    assert PAD_ID not in encoded.turns[0]
    unknown = observed_prefix(conversation(texts=("notinvocab", ""), event=None))
    assert encode_observed_prefix(unknown, vocabulary).turns == ((UNK_ID, EOS_ID), (EOS_ID,))
    assert SequenceVocabulary.from_dict(vocabulary.to_dict()) == vocabulary


def test_empty_text_vocabulary_is_explicit_eos_only_not_failed_fit():
    value = prepare_sequence_forecasts([conversation(texts=("", "", "last"), event=None)])
    vocabulary = fit_sequence_vocabulary(value)
    assert vocabulary.tokens == () and vocabulary.size == 3 and vocabulary.documents == 1
    assert encode_observed_prefix(value.observations[0], vocabulary).turns == ((2,), (2,))
    with pytest.raises(SequenceDataError, match="eligible"):
        fit_sequence_vocabulary(prepare_sequence_forecasts([]))


def test_encoding_heldout_cannot_mutate_training_vocabulary():
    vocabulary = fit_sequence_vocabulary(dataset())
    before = vocabulary.to_dict()
    query = observed_prefix(conversation(texts=("heldout red", "blue new"), event=None))
    encoded = encode_observed_prefix(query, vocabulary)
    assert encoded.known_tokens == 2 and encoded.raw_tokens == 4
    assert vocabulary.to_dict() == before and "heldout" not in vocabulary.tokens
    assert encoded.source_digest == query.digest
    with pytest.raises(TypeError):
        vocabulary._indices["heldout"] = 99
    with pytest.raises(FrozenInstanceError):
        vocabulary.tokens = ("changed",)


def test_candidate_and_long_token_budgets_apply_before_feature_pruning():
    value = dataset()
    with pytest.raises(SequenceLimitError, match="candidate vocabulary"):
        fit_sequence_vocabulary(
            value, max_features=1, limits=SequenceLimits(max_candidate_tokens=2)
        )
    source = conversation(texts=("a longlonglong", "b", "last"), event=None)
    with pytest.raises(SequenceLimitError, match="token"):
        prepare_sequence_forecasts([source], limits=SequenceLimits(max_token_bytes=4))
    with pytest.raises(SequenceLimitError):
        fit_sequence_vocabulary(value, limits=SequenceLimits(max_tokens=5))
    with pytest.raises(SequenceLimitError):
        fit_sequence_vocabulary(value).validate_limits(SequenceLimits(max_token_bytes=3))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(format="future"),
        lambda value: value.update(extra=True),
        lambda value: value.update(tokenizer_version=TOKENIZER_VERSION + "-different"),
        lambda value: value.update(documents=True),
        lambda value: value.update(documents=0),
        lambda value: value.update(max_turn_tokens=True),
        lambda value: value.update(max_turn_tokens=10**400),
        lambda value: value.update(long_turn_policy="tail"),
        lambda value: value["tokens"].reverse(),
        lambda value: value["tokens"].append("x"),
        lambda value: value["tokens"].__setitem__(0, "Blue"),
        lambda value: value["tokens"].__setitem__(0, "!"),
        lambda value: value["document_frequencies"].__setitem__(0, True),
        lambda value: value["document_frequencies"].__setitem__(0, 3),
    ],
)
def test_closed_vocabulary_corruption(mutate):
    value = fit_sequence_vocabulary(dataset()).to_dict()
    mutate(value)
    with pytest.raises(SequenceDataError):
        SequenceVocabulary.from_dict(value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(format="future"),
        lambda row: row.update(source_digest="not-a-digest"),
        lambda row: row.update(vocabulary_size=True),
        lambda row: row.update(raw_tokens=0),
        lambda row: row.update(known_tokens=999),
        lambda row: row.update(truncated_turns=1),
        lambda row: row["turns"][0].__setitem__(0, PAD_ID),
        lambda row: row["turns"][0].__setitem__(0, EOS_ID),
        lambda row: row["turns"][0].__setitem__(0, True),
        lambda row: row["turns"][0].__setitem__(0, 999),
        lambda row: row["turns"][0].__setitem__(-1, UNK_ID),
        lambda row: row.update(turns=[]),
    ],
)
def test_closed_encoding_corruption(mutate):
    value = dataset()
    row = encode_observed_prefix(value.observations[0], fit_sequence_vocabulary(value)).to_dict()
    mutate(row)
    with pytest.raises(SequenceDataError):
        EncodedPrefix.from_dict(row)


def test_sequence_contracts_import_without_numpy_or_torch(monkeypatch):
    import builtins
    import importlib

    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "torch"}:
            raise AssertionError("numerical dependency imported by data layer")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    # Reloading would replace class identities used by other tests. Compile and
    # execute under a fresh package-qualified alias instead, preserving globals.
    for name in ("turnscope.neural_forecast_data", "turnscope.neural_token_data"):
        module = importlib.import_module(name)
        namespace = {"__name__": name, "__package__": "turnscope"}
        exec(compile(__import__("inspect").getsource(module), module.__file__, "exec"), namespace)


def test_unicode_byte_not_character_admission_and_numerical_source_type_errors():
    source = conversation(texts=("éé", "b", "last"))
    with pytest.raises(SequenceLimitError, match="byte limit"):
        prepare_sequence_forecasts([source], limits=SequenceLimits(max_turn_bytes=3))
    for invalid in [None, True, 1.5]:
        with pytest.raises(SequenceDataError, match="iterable"):
            prepare_sequence_forecasts(invalid)
    with pytest.raises(SequenceDataError, match="Conversation"):
        prepare_sequence_forecasts([{"id": "not typed"}])
    with pytest.raises(SequenceDataError, match="Unicode"):
        ObservedTurn(10, BASE, "a")
    with pytest.raises(SequenceDataError, match="ISO-8601"):
        ObservedTurn.from_dict({"id": "a", "text": "a", "timestamp": 10})


def test_aggregate_token_and_repeated_group_occurrence_budgets():
    sources = [conversation("one"), conversation("two")]
    with pytest.raises(SequenceLimitError, match="token budget"):
        prepare_sequence_forecasts(sources, limits=SequenceLimits(max_tokens=3))
    assert (
        prepare_sequence_forecasts(sources, limits=SequenceLimits(max_tokens=4)).audit.raw_tokens
        == 4
    )
    source = conversation()
    source.metadata["forecast_groups"] = ["a", "b"]
    with pytest.raises(SequenceLimitError, match="group budget"):
        prepare_sequence_forecasts([source], limits=SequenceLimits(max_groups=1))


@pytest.mark.parametrize(
    "turns",
    [
        (),
        [],
        ("untyped",),
        (ObservedTurn("a", BASE, "a"), ObservedTurn("a", BASE, "b")),
        (ObservedTurn("a", BASE + timedelta(seconds=1), "a"), ObservedTurn("b", BASE, "b")),
    ],
)
def test_public_prefix_constructor_rejects_nonimmutable_or_incoherent_turns(turns):
    with pytest.raises(SequenceDataError):
        ObservedPrefix("source", turns)


def test_native_prefix_caller_limits_and_raw_aggregate_turn_admission(monkeypatch):
    import turnscope.neural_forecast_data as module

    prefix = dataset().observations[1]
    with pytest.raises(SequenceLimitError, match="observed-turn"):
        prefix.validate_limits(SequenceLimits(max_observed_turns=2))
    with pytest.raises(SequenceLimitError, match="source-byte"):
        prefix.validate_limits(SequenceLimits(max_source_bytes=5))
    raw = prefix.to_dict()
    monkeypatch.setitem(module._HARD, "max_source_turns", 2)
    with pytest.raises(SequenceLimitError, match="turn budget"):
        ObservedPrefix.from_dict(raw)


@pytest.mark.parametrize(
    "change",
    [
        {"observations": []},
        {"examples": []},
        {"group_digests": set()},
        {"group_digests": frozenset({"not-sha256"})},
        {"audit": {}},
        {"observations": ("untyped",)},
        {"examples": ("untyped",)},
    ],
)
def test_native_dataset_cannot_bypass_closed_immutable_inventory(change):
    with pytest.raises(SequenceDataError):
        replace(dataset(), **change)
