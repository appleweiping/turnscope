"""Deterministic context windows and conversation reliability audits."""

from .adapters import (
    ChatFormat,
    adapt_anthropic,
    adapt_conversation,
    adapt_openai,
    adapt_sharegpt,
    iter_adapted_conversations,
    iter_adapted_jsonl,
    iter_adapted_path,
)
from .audit import Auditor, default_auditor
from .builder import ContextBuilder
from .corpus import CorpusStore
from .graph import InteractionEdge, ReplyForest, interaction_edges, reply_forest
from .io import iter_conversations, iter_path
from .models import AuditReport, ContextWindow, Conversation, Issue, Severity, Utterance
from .policies import (
    ReplyChainPolicy,
    TimeWindowPolicy,
    TokenBudgetPolicy,
    TokenCounter,
    TurnWindowPolicy,
    Utf8ByteTokenCounter,
    WhitespaceTokenCounter,
)
from .transformers import (
    ConversationFeatures,
    SpeakerProfile,
    TfidfState,
    TfidfVectorizer,
    conversation_features,
    speaker_profiles,
)

__all__ = [
    "AuditReport",
    "Auditor",
    "ChatFormat",
    "ContextBuilder",
    "ContextWindow",
    "Conversation",
    "ConversationFeatures",
    "CorpusStore",
    "InteractionEdge",
    "Issue",
    "ReplyChainPolicy",
    "ReplyForest",
    "Severity",
    "SpeakerProfile",
    "TfidfState",
    "TfidfVectorizer",
    "TimeWindowPolicy",
    "TokenBudgetPolicy",
    "TokenCounter",
    "TurnWindowPolicy",
    "Utf8ByteTokenCounter",
    "Utterance",
    "WhitespaceTokenCounter",
    "adapt_anthropic",
    "adapt_conversation",
    "adapt_openai",
    "adapt_sharegpt",
    "conversation_features",
    "default_auditor",
    "interaction_edges",
    "iter_adapted_conversations",
    "iter_adapted_jsonl",
    "iter_adapted_path",
    "iter_conversations",
    "iter_path",
    "reply_forest",
    "speaker_profiles",
]

__version__ = "0.2.0"
