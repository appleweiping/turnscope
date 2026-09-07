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
from .classifier import ClassifierState, ConversationClassifier
from .corpus import CorpusStore
from .graph import (
    InteractionEdge,
    InteractionNetwork,
    InteractionNetworkAccumulator,
    NetworkEdge,
    ReplyForest,
    SpeakerSummary,
    interaction_edges,
    interaction_network,
    reply_forest,
)
from .io import iter_conversations, iter_path
from .models import AuditReport, ContextWindow, Conversation, Issue, Severity, Utterance
from .pipeline import CallableTransformer, ConversationTransformer, FeaturePipeline, FeatureRecord
from .policies import (
    ReplyChainPolicy,
    TimeWindowPolicy,
    TokenBudgetPolicy,
    TokenCounter,
    TurnWindowPolicy,
    Utf8ByteTokenCounter,
    WhitespaceTokenCounter,
)
from .search import ConversationSearchIndex, SearchHit
from .transformers import (
    ConversationFeatures,
    CoordinationScore,
    CorpusSpeakerProfile,
    DiversityProfile,
    SpeakerProfile,
    TfidfState,
    TfidfVectorizer,
    conversation_features,
    corpus_speaker_profiles,
    linguistic_coordination,
    linguistic_diversity,
    speaker_profiles,
)

__all__ = [
    "AuditReport",
    "Auditor",
    "CallableTransformer",
    "ChatFormat",
    "ClassifierState",
    "ContextBuilder",
    "ContextWindow",
    "Conversation",
    "ConversationClassifier",
    "ConversationFeatures",
    "ConversationSearchIndex",
    "ConversationTransformer",
    "CoordinationScore",
    "CorpusSpeakerProfile",
    "CorpusStore",
    "DiversityProfile",
    "FeaturePipeline",
    "FeatureRecord",
    "InteractionEdge",
    "InteractionNetwork",
    "InteractionNetworkAccumulator",
    "Issue",
    "NetworkEdge",
    "ReplyChainPolicy",
    "ReplyForest",
    "SearchHit",
    "Severity",
    "SpeakerProfile",
    "SpeakerSummary",
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
    "corpus_speaker_profiles",
    "default_auditor",
    "interaction_edges",
    "interaction_network",
    "iter_adapted_conversations",
    "iter_adapted_jsonl",
    "iter_adapted_path",
    "iter_conversations",
    "iter_path",
    "linguistic_coordination",
    "linguistic_diversity",
    "reply_forest",
    "speaker_profiles",
]

__version__ = "0.2.0"
