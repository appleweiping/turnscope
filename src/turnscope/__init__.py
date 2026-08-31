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

__all__ = [
    "AuditReport",
    "Auditor",
    "ChatFormat",
    "ContextBuilder",
    "ContextWindow",
    "Conversation",
    "Issue",
    "ReplyChainPolicy",
    "Severity",
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
    "default_auditor",
    "iter_adapted_conversations",
    "iter_adapted_jsonl",
    "iter_adapted_path",
    "iter_conversations",
    "iter_path",
]

__version__ = "0.2.0"
