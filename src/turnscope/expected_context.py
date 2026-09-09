"""Frozen SVD-plus-ridge prediction of explicitly defined utterance contexts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .graph import reply_forest
from .models import Conversation, Utterance
from .sparse import normalize_sparse
from .transformers import TfidfState, TfidfVectorizer, _tokens

_RELATIONS = ("reply", "predecessor", "sequence-successor")


def _integer(value: object, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class ContextPrediction:
    """Latent context coordinates with explicit source-token coverage."""

    vector: tuple[float, ...]
    tokens: int
    known_tokens: int

    @property
    def coverage(self) -> float:
        return self.known_tokens / self.tokens if self.tokens else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "vector": list(self.vector),
            "tokens": self.tokens,
            "known_tokens": self.known_tokens,
            "coverage": self.coverage,
        }


@dataclass(frozen=True, slots=True)
class ExpectedContextState:
    """Immutable learned basis/map and pair-weighted training summaries."""

    source_tfidf: TfidfState
    context_tfidf: TfidfState
    basis: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[float, ...], ...]
    source_mean: tuple[float, ...]
    context_mean: tuple[float, ...]
    singular_values: tuple[float, ...]
    training_pairs: int
    estimated_dense_cells: int
    training_mse: float
    mean_baseline_mse: float

    @property
    def dimensions(self) -> int:
        return len(self.singular_values)


@dataclass(frozen=True, slots=True)
class ContextPair:
    """One explicitly defined context sample; source/context IDs are conversation-scoped."""

    conversation_id: str
    source: Utterance
    context: Utterance


def iter_context_pairs(
    conversations: Iterable[Conversation], *, relation: str = "reply"
) -> Iterable[ContextPair]:
    """Yield actual replies/predecessors or explicitly declared sequence successors.

    Sequence successors mean adjacent input positions, not inferred reply-tree
    edges. Every mode rejects duplicate IDs. Reply modes also validate the
    entire reply forest, including dangling references and cycles.
    """
    if relation not in _RELATIONS:
        raise ValueError(f"relation must be one of {_RELATIONS}")
    seen: set[str] = set()
    for conversation in conversations:
        if not isinstance(conversation, Conversation):
            raise TypeError("conversations must contain Conversation values")
        if conversation.id in seen:
            raise ValueError("conversation IDs must be unique")
        seen.add(conversation.id)
        by_id = conversation.by_id()
        if len(by_id) != len(conversation.utterances):
            raise ValueError("utterance IDs must be unique within each conversation")
        if relation == "sequence-successor":
            for source, context in zip(
                conversation.utterances, conversation.utterances[1:], strict=False
            ):
                yield ContextPair(conversation.id, source, context)
        else:
            forest = reply_forest(conversation)
            for identifier, parent in forest.parents.items():
                if parent is not None:
                    source, context = by_id[parent], by_id[identifier]
                    if relation == "predecessor":
                        source, context = context, source
                    yield ContextPair(conversation.id, source, context)


def _text_vector(text: str, state: TfidfState) -> tuple[Mapping[str, float], int, int]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens = _tokens(text)
    counts = Counter(token for token in tokens if token in state.idf)
    known = sum(counts.values())
    weighted = {token: count / known * state.idf[token] for token, count in counts.items()}
    return normalize_sparse(weighted), len(tokens), known


class ExpectedContextModel:
    """Train a low-rank context space and a centered ridge predictor into it.

    NumPy is needed only by ``fit``. Artifact loading and prediction use the
    standard library. Fitting is bounded and all-or-nothing; successful heldout
    prediction cannot change vocabulary, IDF, context basis, or regression map.
    """

    def __init__(
        self,
        *,
        n_components: int = 16,
        regularization: float = 1.0,
        relation: str = "reply",
        min_document_frequency: int = 1,
        max_features: int = 256,
        max_training_pairs: int = 5000,
        max_dense_cells: int = 16_000_000,
    ) -> None:
        self._components = _integer(n_components, "n_components", 256)
        self._max_features = _integer(max_features, "max_features", 1024)
        self._min_df = _integer(min_document_frequency, "min_document_frequency", 20_000)
        self._max_pairs = _integer(max_training_pairs, "max_training_pairs", 20_000)
        self._max_cells = _integer(max_dense_cells, "max_dense_cells", 64_000_000)
        if isinstance(regularization, bool) or not isinstance(regularization, (int, float)):
            raise ValueError("regularization must be a finite positive number")
        try:
            penalty = float(regularization)
        except OverflowError as error:
            raise ValueError("regularization must be a finite positive number") from error
        if not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("regularization must be a finite positive number")
        if relation not in _RELATIONS:
            raise ValueError(f"relation must be one of {_RELATIONS}")
        self._regularization = penalty
        self._relation = relation
        self._state: ExpectedContextState | None = None
        self._source_model: TfidfVectorizer | None = None
        self._context_model: TfidfVectorizer | None = None

    @property
    def state(self) -> ExpectedContextState:
        if self._state is None:
            raise ValueError("expected-context model has not been fitted")
        return self._state

    @property
    def relation(self) -> str:
        return self._relation

    def fit(self, conversations: Iterable[Conversation]) -> ExpectedContextModel:
        """Fit using one sample per defined context pair; endpoints fit TF-IDF once each."""
        pairs: list[ContextPair] = []
        for pair in iter_context_pairs(conversations, relation=self._relation):
            if len(pairs) == self._max_pairs:
                raise ValueError("training pairs exceed max_training_pairs")
            pairs.append(pair)
        if not pairs:
            raise ValueError("at least one context pair is required for fitting")
        pairs.sort(key=lambda pair: (pair.conversation_id, pair.source.id, pair.context.id))
        sources = {(pair.conversation_id, pair.source.id): pair.source for pair in pairs}
        contexts = {(pair.conversation_id, pair.context.id): pair.context for pair in pairs}

        def tfidf(utterances: Mapping[tuple[str, str], Utterance]) -> TfidfVectorizer:
            return TfidfVectorizer(
                min_document_frequency=self._min_df, max_features=self._max_features
            ).fit(
                Conversation(json.dumps(key, ensure_ascii=True), [item])
                for key, item in sorted(utterances.items())
            )

        source_model, context_model = tfidf(sources), tfidf(contexts)
        p, q = len(source_model.state.vocabulary), len(context_model.state.vocabulary)
        if not p or not q:
            raise ValueError("source and context training vocabularies must both be non-empty")
        k = min(self._components, len(contexts), q)
        cells = (
            4 * len(pairs) * p + 4 * len(contexts) * q + 3 * p * p + 3 * len(pairs) * k + 2 * q * k
        )
        if cells > self._max_cells:
            raise ValueError(f"estimated dense fitting workspace {cells} exceeds max_dense_cells")

        def row(item: Utterance, state: TfidfState) -> list[float]:
            vector, _, _ = _text_vector(item.text, state)
            return [vector.get(token, 0.0) for token in state.vocabulary]

        source_rows = [row(pair.source, source_model.state) for pair in pairs]
        context_keys = sorted(contexts)
        context_indices = {key: index for index, key in enumerate(context_keys)}
        context_rows = [row(contexts[key], context_model.state) for key in context_keys]
        pair_indices = [context_indices[pair.conversation_id, pair.context.id] for pair in pairs]
        from ._context_numeric import fit_numeric_context

        numeric = fit_numeric_context(
            source_rows, context_rows, pair_indices, self._components, self._regularization
        )
        new_state = ExpectedContextState(
            source_model.state,
            context_model.state,
            numeric.basis,
            numeric.coefficients,
            numeric.source_mean,
            numeric.context_mean,
            numeric.singular_values,
            len(pairs),
            cells,
            numeric.training_mse,
            numeric.mean_baseline_mse,
        )
        self._source_model, self._context_model, self._state = (
            source_model,
            context_model,
            new_state,
        )
        return self

    def predict(self, text: str) -> ContextPrediction:
        """Predict latent context coordinates; zero/OOV text maps to the fitted intercept."""
        state = self.state
        vector, tokens, known = _text_vector(text, state.source_tfidf)
        try:
            result = tuple(
                state.context_mean[dimension]
                + math.fsum(
                    (vector.get(term, 0.0) - state.source_mean[index])
                    * state.coefficients[index][dimension]
                    for index, term in enumerate(state.source_tfidf.vocabulary)
                )
                for dimension in range(state.dimensions)
            )
        except OverflowError as error:
            raise ValueError("expected-context prediction exceeded finite numeric range") from error
        if not all(math.isfinite(value) for value in result):
            raise ValueError("expected-context prediction exceeded finite numeric range")
        return ContextPrediction(result, tokens, known)

    def project_context(self, text: str) -> ContextPrediction:
        """Project observed context text into the same frozen SVD space, without prediction."""
        state = self.state
        vector, tokens, known = _text_vector(text, state.context_tfidf)
        return ContextPrediction(
            tuple(
                math.fsum(
                    vector.get(term, 0.0) * state.basis[index][dimension]
                    for index, term in enumerate(state.context_tfidf.vocabulary)
                )
                for dimension in range(state.dimensions)
            ),
            tokens,
            known,
        )

    def transform(self, conversation: Conversation) -> Mapping[str, ContextPrediction]:
        """Return predictions for every utterance, without inspecting heldout reply text."""
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation value")
        if len(conversation.by_id()) != len(conversation.utterances):
            raise ValueError("utterance IDs must be unique within each conversation")
        _ = self.state
        return MappingProxyType(
            {item.id: self.predict(item.text) for item in conversation.utterances}
        )

    def evaluate(self, conversations: Iterable[Conversation]) -> dict[str, object]:
        """Compare heldout contexts to predictions and the pair-weighted training mean."""
        state = self.state
        count = uncovered_sources = uncovered_contexts = 0
        error = baseline_error = 0.0
        for pair in iter_context_pairs(conversations, relation=self._relation):
            predicted = self.predict(pair.source.text)
            actual = self.project_context(pair.context.text)
            try:
                error += math.fsum(
                    (left - right) ** 2
                    for left, right in zip(predicted.vector, actual.vector, strict=True)
                )
            except OverflowError as problem:
                raise ValueError("context evaluation exceeded finite numeric range") from problem
            if not math.isfinite(error):
                raise ValueError("context evaluation exceeded finite numeric range")
            baseline_error += math.fsum(
                (left - right) ** 2
                for left, right in zip(state.context_mean, actual.vector, strict=True)
            )
            count += 1
            uncovered_sources += predicted.known_tokens == 0
            uncovered_contexts += actual.known_tokens == 0
        if not count:
            raise ValueError("at least one context pair is required for evaluation")
        return {
            "relation": self._relation,
            "pairs": count,
            "dimensions": state.dimensions,
            "mse": error / (count * state.dimensions),
            "training_mean_baseline_mse": baseline_error / (count * state.dimensions),
            "zero_source_vectors": uncovered_sources,
            "zero_context_vectors": uncovered_contexts,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize frozen configuration and learned parameters with a checksum."""
        from .context_artifact import encode_context_model

        return encode_context_model(self)

    @classmethod
    def from_dict(cls, value: object) -> ExpectedContextModel:
        from .context_artifact import decode_context_model

        return decode_context_model(value)

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def save(self, path: str | Path) -> None:
        from .context_artifact import save_context_model

        save_context_model(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> ExpectedContextModel:
        from .context_artifact import load_context_model

        return cls.from_dict(load_context_model(Path(path)))
