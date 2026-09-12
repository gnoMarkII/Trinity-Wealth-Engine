"""Stable application-level errors for multi-app write clients."""
from __future__ import annotations


class KnowledgeWriteError(RuntimeError):
    """Base error for command submission and receipt handling."""


class WriteCommandValidationError(KnowledgeWriteError):
    """The command is invalid and must not be retried automatically."""


class WriteConflictError(KnowledgeWriteError):
    """The command conflicts with current identity, revision, or ownership."""


class WriteUnavailableError(KnowledgeWriteError):
    """The broker is unavailable; retry may be safe using the same command."""


class WriteDeadLetterError(KnowledgeWriteError):
    """The command exhausted retries and needs explicit operator action."""
