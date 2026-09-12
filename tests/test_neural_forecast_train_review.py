"""Optional real-Torch independent serial-gradient and optimizer review oracles.

No learned-model quality claim: all inputs are authored tiny conversations.
Only this optional module skips when Torch is unavailable.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from turnscope import neural_forecast_train as training
from turnscope.models import Conversation, Utterance
from turnscope.neural_forecast_data import prepare_sequence_forecasts
from turnscope.neural_forecast_math import NeuralArchitecture
from turnscope.neural_token_data import encode_observed_prefix, fit_sequence_vocabulary

torch = pytest.importorskip("torch")


def serial_logits(module, turns):
    """Direct unpadded Torch calls, independently of either sorting implementation."""
    vectors = []
    for turn in turns:
        ids = torch.tensor(turn, dtype=torch.long, device="cpu")
        _, last = module.word(module.embedding(ids).unsqueeze(1))
        vectors.append(torch.cat((last[-2, 0], last[-1, 0])))
    contexts, _ = module.turn(torch.stack(vectors).unsqueeze(1))
    return module.output(torch.tanh(module.head(contexts[:, 0]))).flatten()


def serial_mean_loss(module, records):
    per_conversation = []
    for record in records:
        scores = serial_logits(module, record.turns)[list(record.endpoints)]
        # Each conversation has unit weight regardless of its eligible-prefix count.
        target = torch.full_like(scores, float(record.label))
        per_conversation.append(
            torch.nn.functional.binary_cross_entropy_with_logits(scores, target)
        )
    return torch.stack(per_conversation).mean()


def unequal_records():
    return (
        training._EncodedConversation(((3, 2), (4, 5, 6, 2)), (0, 1), True),
        training._EncodedConversation(((7, 4, 2),), (0,), False),
        training._EncodedConversation(((5, 2), (3, 4, 5, 6, 7, 2), (2,)), (1, 2), True),
    )


def fitting_data(prefix):
    conversations = []
    for index, (turns, positive) in enumerate(((2, True), (3, False), (4, True))):
        messages = []
        for position in range(turns + 1):
            messages.append(
                Utterance(
                    f"m{position}",
                    "unused",
                    ("alpha beta", "beta gamma", "gamma alpha")[position % 3],
                    datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=position),
                    metadata={"event": positive and position == turns},
                )
            )
        conversations.append(
            Conversation(f"{prefix}-{index}", messages, {"forecast_groups": [f"{prefix}-{index}"]})
        )
    return prepare_sequence_forecasts(conversations)


def encode_records(dataset, vocabulary):
    return tuple(
        training._EncodedConversation(
            encode_observed_prefix(observation, vocabulary).turns,
            tuple(row.endpoint for row in dataset.examples if row.conversation_index == index),
            next(row.label for row in dataset.examples if row.conversation_index == index),
        )
        for index, observation in enumerate(dataset.observations)
    )


@pytest.mark.parametrize("word_layers,turn_layers", ((1, 1), (1, 2), (2, 1), (2, 2)))
def test_packed_inverse_orders_and_every_parameter_gradient_match_serial_oracle(
    word_layers, turn_layers
):
    architecture = NeuralArchitecture(8, 3, 2, 4, word_layers, turn_layers)
    packed = training.make_torch_hierarchy(torch, architecture, seed=23)
    serial = training.make_torch_hierarchy(torch, architecture, seed=23)
    records = unequal_records()
    actual = training.forward_encoded_batch(torch, packed, records)
    for record, scores in zip(records, actual, strict=True):
        torch.testing.assert_close(
            scores, serial_logits(serial, record.turns), rtol=1e-5, atol=1e-7
        )
    batch_loss = training._batch_loss(torch, packed, records)
    expected_loss = serial_mean_loss(serial, records)
    torch.testing.assert_close(batch_loss, expected_loss, rtol=1e-6, atol=1e-7)
    batch_loss.backward()
    expected_loss.backward()
    for (name, actual_parameter), (expected_name, expected_parameter) in zip(
        packed.named_parameters(), serial.named_parameters(), strict=True
    ):
        assert name == expected_name
        assert actual_parameter.grad is not None
        assert expected_parameter.grad is not None
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            rtol=1e-4,
            atol=1e-7,
            msg=lambda message, tensor_name=name: tensor_name + ": " + message,
        )
    assert torch.count_nonzero(packed.embedding.weight.grad[0]).item() == 0


def test_actual_training_preserves_global_rng_dtype_and_thread_configuration():
    train = fitting_data("training")
    validation = fitting_data("validation")
    vocabulary = fit_sequence_vocabulary(train)
    architecture = NeuralArchitecture(vocabulary.size, 3, 2, 4, 2, 2)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # Probe a non-default caller setting.
    before = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        torch.get_default_dtype(),
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        torch.are_deterministic_algorithms_enabled(),
        torch.get_deterministic_debug_mode(),
    )
    source_digests = train.digest, validation.digest, vocabulary.digest
    try:
        result = training.train_sequence_model(
            train,
            validation,
            vocabulary,
            architecture,
            config=training.NeuralTrainingConfig(epochs=1, batch_conversations=2, seed=79),
        )
        assert random.getstate() == before[0]
        current_numpy = np.random.get_state()
        assert current_numpy[0] == before[1][0]
        np.testing.assert_array_equal(current_numpy[1], before[1][1])
        assert current_numpy[2:] == before[1][2:]
        assert torch.equal(torch.get_rng_state(), before[2])
        assert (
            torch.get_default_dtype(),
            torch.get_num_threads(),
            torch.get_num_interop_threads(),
            torch.are_deterministic_algorithms_enabled(),
            torch.get_deterministic_debug_mode(),
        ) == before[3:]
        assert (train.digest, validation.digest, vocabulary.digest) == source_digests
        assert result.history[0].optimizer_steps == 2  # Three records include final partial batch.
        assert all(array.dtype == np.dtype("<f4") for array in result.parameters.arrays().values())
    finally:
        # Preserve the caller's state even when a regression makes assertions fail.
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])
        torch.set_default_dtype(old_dtype)


def test_two_real_updates_match_independent_clipped_adam_bias_correction(monkeypatch):
    train = fitting_data("training")
    validation = fitting_data("validation")
    vocabulary = fit_sequence_vocabulary(train)
    architecture = NeuralArchitecture(vocabulary.size, 3, 2, 4, 2, 2)
    config = training.NeuralTrainingConfig(
        epochs=2, batch_conversations=3, learning_rate=0.02, gradient_clip=0.0001, seed=53
    )
    serial = training.make_torch_hierarchy(torch, architecture, seed=config.seed)
    records = encode_records(train, vocabulary)
    moments = {name: np.zeros(tuple(value.shape)) for name, value in serial.named_parameters()}
    variances = {name: np.zeros(tuple(value.shape)) for name, value in serial.named_parameters()}
    rng = random.Random(config.seed)
    clip_was_active = False
    for step in (1, 2):
        order = list(range(len(records)))
        rng.shuffle(order)
        serial.zero_grad(set_to_none=True)
        # The preceding test independently checks this graph's full gradients.
        # Reuse that graph here to isolate Adam arithmetic from the small
        # float32 reduction differences between packed and unpadded kernels.
        training._batch_loss(torch, serial, tuple(records[index] for index in order)).backward()
        gradients = {
            name: parameter.grad.detach().numpy().astype(np.float64)
            for name, parameter in serial.named_parameters()
        }
        norm = math.sqrt(math.fsum(float(np.sum(value * value)) for value in gradients.values()))
        scale = min(1.0, config.gradient_clip / (norm + 1e-6))
        clip_was_active |= scale < 1
        with torch.no_grad():
            for name, parameter in serial.named_parameters():
                gradient = gradients[name] * scale
                moments[name] = 0.9 * moments[name] + 0.1 * gradient
                variances[name] = 0.999 * variances[name] + 0.001 * gradient * gradient
                update = (moments[name] / (1 - 0.9**step)) / (
                    np.sqrt(variances[name] / (1 - 0.999**step)) + 1e-8
                )
                value = (
                    parameter.detach().numpy().astype(np.float64) - config.learning_rate * update
                )
                parameter.copy_(torch.from_numpy(value.astype(np.float32)))
    assert clip_was_active
    # Only epoch selection is scripted here; both actual optimization steps run.
    losses = iter((2.0, 1.0))
    monkeypatch.setattr(training, "_validation_loss", lambda *_args: next(losses))
    result = training.train_sequence_model(
        train, validation, vocabulary, architecture, config=config
    )
    assert result.selected_epoch == 2
    for name, parameter in serial.named_parameters():
        np.testing.assert_allclose(
            result.parameters.arrays()[name],
            parameter.detach().numpy(),
            rtol=0,
            atol=8e-7,
            err_msg=name,
        )
