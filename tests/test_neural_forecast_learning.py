"""One predeclared authored order-learning experiment, not language generalization.

The two input patterns recur across disjoint conversation/group IDs. No seed,
architecture, optimizer setting or success threshold is searched by these tests.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone
from itertools import combinations

import pytest

from turnscope.models import Conversation, Utterance
from turnscope.neural_forecast import HierarchicalEventForecaster, NeuralForecastConfig
from turnscope.neural_forecast_data import (
    ObservedPrefix,
    ObservedTurn,
    SequencePolicy,
    prepare_sequence_forecasts,
)
from turnscope.neural_forecast_math import infer_encoded_turns
from turnscope.neural_forecast_train import NeuralTrainingConfig, make_torch_hierarchy
from turnscope.neural_token_data import encode_observed_prefix, fit_sequence_vocabulary

SEED = 20260912
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
POLICY = SequencePolicy(min_turns=3)
CONFIG = NeuralForecastConfig(
    embedding_dim=8,
    word_hidden=8,
    turn_hidden=8,
    word_layers=1,
    turn_layers=1,
    max_turn_tokens=1,
    long_turn_policy="reject",
)
TRAINING = NeuralTrainingConfig(
    epochs=80,
    patience=20,
    batch_conversations=8,
    learning_rate=0.03,
    gradient_clip=5,
    seed=SEED,
)
PARTITIONS = (("train", 8), ("validation", 4), ("policy", 4), ("heldout", 6))


def source_partition(name, count):
    """Same unigram bag and last turn; labels depend only on the first two turns."""
    rows = []
    for index in range(count):
        positive = index % 2 == 0
        ordered = ("alpha", "beta") if positive else ("beta", "alpha")
        texts = (*ordered, "anchor", "future")
        rows.append(
            Conversation(
                f"{name}-{index}",
                tuple(
                    Utterance(
                        str(position),
                        "participant",
                        text,
                        BASE + timedelta(seconds=position),
                        metadata={"event": positive and position == 3},
                    )
                    for position, text in enumerate(texts)
                ),
                metadata={"forecast_groups": [f"group:{name}:{index}"]},
            )
        )
    return tuple(rows)


def prepared_partitions():
    return {
        name: prepare_sequence_forecasts(source_partition(name, count), policy=POLICY)
        for name, count in PARTITIONS
    }


def test_order_task_is_balanced_causal_and_indistinguishable_to_unigram_features():
    datasets = prepared_partitions()
    for left, right in combinations(datasets.values(), 2):
        assert left.group_digests.isdisjoint(right.group_digests)
        assert {row.conversation_id for row in left.observations}.isdisjoint(
            row.conversation_id for row in right.observations
        )
    for name, count in PARTITIONS:
        data = datasets[name]
        assert len(data.observations) == len(data.examples) == count
        assert data.audit.positive_conversations == data.audit.negative_conversations == count // 2
        assert data.weights == (1.0,) * count
        for example in data.examples:
            prefix = data.prefix(example)
            assert example.endpoint == 2
            assert example.lead_turns == (1 if example.label else None)
            assert tuple(turn.id for turn in prefix.turns) == ("0", "1", "2")
            assert prefix.turns[-1].text == "anchor"
            assert Counter(turn.text for turn in prefix.turns) == {
                "alpha": 1,
                "beta": 1,
                "anchor": 1,
            }
            assert (prefix.turns[0].text == "alpha") is example.label
        # Any deterministic unigram-only model gives the same score to every
        # example. On this balanced task every possible constant alert has BA .5.
        labels = [example.label for example in data.examples]
        for alert in (False, True):
            tpr = sum(alert and label for label in labels) / (count // 2)
            tnr = sum(not alert and not label for label in labels) / (count // 2)
            assert (tpr + tnr) / 2 == 0.5
        assert sum((0.5 - label) ** 2 for label in labels) / count == 0.25
        assert -sum(math.log(0.5) for _ in labels) / count == pytest.approx(math.log(2))
    vocabulary = fit_sequence_vocabulary(
        datasets["train"], max_turn_tokens=1, long_turn_policy="reject"
    )
    assert vocabulary.tokens == ("alpha", "anchor", "beta")
    assert vocabulary.document_frequencies == (8, 8, 8)
    assert "future" not in vocabulary.tokens


@pytest.fixture(scope="module")
def learned_order_model():
    torch = pytest.importorskip("torch", reason="optional CPU training dependency is not installed")
    datasets = prepared_partitions()
    identities = {name: data.digest for name, data in datasets.items()}
    model = HierarchicalEventForecaster(config=CONFIG, training_config=TRAINING, policy=POLICY).fit(
        datasets["train"], datasets["validation"], policy_validation=datasets["policy"]
    )
    assert {name: data.digest for name, data in datasets.items()} == identities
    return torch, model, datasets


def test_fixed_seed_learns_order_on_identity_disjoint_heldout(learned_order_model):
    torch, model, datasets = learned_order_model
    heldout = datasets["heldout"]
    predictions = model.transform(heldout.observations)
    positives = [
        value.probability
        for value, example in zip(predictions, heldout.examples, strict=True)
        if example.label
    ]
    negatives = [
        value.probability
        for value, example in zip(predictions, heldout.examples, strict=True)
        if not example.label
    ]
    metrics = model.evaluate(heldout)["metrics"]
    # Print the sole declared attempt before acceptance assertions: a failed
    # protocol remains visible and must not be concealed by trying more seeds.
    print(
        json.dumps(
            {
                "kind": "authored-two-pattern-order-learning",
                "seed": SEED,
                "torch": str(torch.__version__),
                "numpy": model.state.training.numpy_version,
                "epochs_executed": len(model.state.training.history),
                "selected_epoch": model.state.training.selected_epoch,
                "heldout_positive_min": min(positives),
                "heldout_negative_max": max(negatives),
                "threshold": model.state.threshold,
                "metrics": metrics,
            },
            sort_keys=True,
        )
    )
    assert model.state.training.config == TRAINING
    assert not model.state.policy_reuses_model_validation
    assert min(positives) >= 0.75
    assert max(negatives) <= 0.25
    assert min(positives) - max(negatives) >= 0.5
    assert metrics["conversations"] == metrics["prefixes"] == 6
    assert metrics["any_alert"]["balanced_accuracy"] == 1.0
    assert metrics["conversation_weighted_prefix_roc_auc"] == 1.0
    assert metrics["conversation_weighted_prefix_brier"] <= 0.0625
    assert metrics["any_alert"]["tp"] == metrics["any_alert"]["tn"] == 3
    assert metrics["any_alert"]["fp"] == metrics["any_alert"]["fn"] == 0


def test_swapping_only_earlier_turn_text_flips_learned_score(learned_order_model):
    _torch, model, datasets = learned_order_model
    original = datasets["heldout"].observations[0]
    swapped = ObservedPrefix(
        original.conversation_id,
        tuple(
            ObservedTurn(turn.id, turn.timestamp, text)
            for turn, text in zip(original.turns, ("beta", "alpha", "anchor"), strict=True)
        ),
    )
    assert original.turns[-1] == swapped.turns[-1]
    assert [turn.id for turn in original.turns] == [turn.id for turn in swapped.turns]
    assert [turn.timestamp for turn in original.turns] == [turn.timestamp for turn in swapped.turns]
    assert Counter(turn.text for turn in original.turns) == Counter(
        turn.text for turn in swapped.turns
    )
    positive = model.predict(original)
    negative = model.predict(swapped)
    assert positive.probability - negative.probability >= 0.5
    assert positive.alert and not negative.alert


def test_learned_torch_serial_and_frozen_numpy_agree_for_both_orders(learned_order_model):
    torch, model, datasets = learned_order_model
    parameters = model.state.parameters
    module = make_torch_hierarchy(torch, parameters.architecture, seed=SEED)
    module.load_state_dict(
        {name: torch.from_numpy(value.copy()) for name, value in parameters.arrays().items()}
    )
    module.eval()
    for prefix in datasets["heldout"].observations[:2]:
        encoded = encode_observed_prefix(prefix, model.state.vocabulary)
        frozen = infer_encoded_turns(parameters, encoded.turns)
        # Independent unpadded serial composition, not the trainer's packed-
        # batch helper. It checks the actual selected learned tensor values.
        with torch.no_grad():
            vectors = []
            for turn in encoded.turns:
                tokens = torch.tensor(turn, dtype=torch.long).reshape(-1, 1)
                _, hidden = module.word(module.embedding(tokens))
                vectors.append(torch.cat((hidden[-2, 0], hidden[-1, 0])))
            states, _ = module.turn(torch.stack(vectors).unsqueeze(1))
            logits = module.output(torch.tanh(module.head(states))).reshape(-1)
            probabilities = torch.sigmoid(logits)
        assert frozen.logits == pytest.approx(logits.tolist(), abs=2e-5, rel=2e-6)
        assert frozen.probabilities == pytest.approx(probabilities.tolist(), abs=2e-6)
        assert model.predict(prefix).probability == frozen.probabilities[-1]
