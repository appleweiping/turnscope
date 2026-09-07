from datetime import datetime, timezone

import pytest

from turnscope import (
    CallableTransformer,
    Conversation,
    FeaturePipeline,
    TfidfVectorizer,
    Utterance,
    conversation_features,
)


def conversations() -> tuple[Conversation, ...]:
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return (
        Conversation("a", (Utterance("a1", "user", "alpha beta", timestamp),)),
        Conversation("b", (Utterance("b1", "assistant", "beta", timestamp),)),
    )


def test_pipeline_fits_stateful_and_pure_steps_in_order() -> None:
    vectorizer = TfidfVectorizer()
    pipeline = FeaturePipeline(
        (
            ("tfidf", vectorizer),
            ("counts", CallableTransformer(conversation_features)),
        )
    )
    with pytest.raises(ValueError, match="fitted"):
        pipeline.transform(conversations()[0])
    records = pipeline.fit_transform(conversations())
    assert pipeline.fitted and pipeline.step_names == ("tfidf", "counts")
    assert set(records[0].blocks) == {"tfidf", "counts"}
    assert records[0].blocks["counts"].tokens == 2
    assert len(records[0].digest()) == 64


def test_pipeline_validates_duplicate_steps_and_inputs() -> None:
    transformer = CallableTransformer(conversation_features)
    with pytest.raises(ValueError, match="unique"):
        FeaturePipeline((("x", transformer), ("x", transformer)))
    with pytest.raises(ValueError, match="at least one"):
        FeaturePipeline(())
    with pytest.raises(TypeError, match="Conversation"):
        FeaturePipeline((("x", transformer),)).fit(("bad",))  # type: ignore[arg-type]
