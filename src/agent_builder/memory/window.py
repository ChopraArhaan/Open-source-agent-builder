"""Pure history windowing utilities for conversation sessions."""

from __future__ import annotations

from collections.abc import Sequence

from src.agent_builder.memory.models import StoredMessage
from src.agent_builder.models import LLMMessage


def select_history(
    messages: Sequence[StoredMessage],
    max_messages: int | None = None,
    max_chars: int | None = None,
) -> list[StoredMessage]:
    """Select the most recent turns that fit within message and character budgets.

    Characters are used as a lightweight proxy for tokens.

    Invariants:
    - Never starts with an ``assistant`` message.
    - Never splits a ``(user, assistant)`` turn pair; drops whole oldest turns instead.
    - If the newest turn alone exceeds either budget, returns ``[]`` rather than
      exceeding the budget or returning stale older turns without the latest turn.
    - Idempotent: applying ``select_history`` twice with the same budgets yields the
      exact same list.
    """
    if max_messages is not None and max_messages < 0:
        raise ValueError("max_messages must be >= 0")
    if max_chars is not None and max_chars < 0:
        raise ValueError("max_chars must be >= 0")

    if not messages or max_messages == 0 or max_chars == 0:
        return []

    # Group messages into whole turns, each starting with a 'user' message.
    turns: list[list[StoredMessage]] = []
    for msg in messages:
        if msg.role == "user":
            turns.append([msg])
        elif msg.role == "assistant" and turns:
            turns[-1].append(msg)
            # Leading orphan assistant messages are skipped so a window never
            # starts with an assistant message.

    if not turns:
        return []

    selected_reversed: list[list[StoredMessage]] = []
    used_messages = 0
    used_chars = 0

    for turn in reversed(turns):
        turn_msg_count = len(turn)
        turn_char_count = sum(len(m.content) for m in turn)

        if max_messages is not None and used_messages + turn_msg_count > max_messages:
            break
        if max_chars is not None and used_chars + turn_char_count > max_chars:
            break

        selected_reversed.append(turn)
        used_messages += turn_msg_count
        used_chars += turn_char_count

    result: list[StoredMessage] = []
    for turn in reversed(selected_reversed):
        result.extend(turn)
    return result


def to_llm_messages(messages: Sequence[StoredMessage]) -> list[LLMMessage]:
    """Convert StoredMessage instances to provider-independent LLMMessage models."""
    return [LLMMessage(role=m.role, content=m.content) for m in messages]
