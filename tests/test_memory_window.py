"""Unit and property-style tests for history windowing."""

from __future__ import annotations

import random

import pytest

from src.agent_builder.memory import StoredMessage, select_history, to_llm_messages


def _msg(role: str, content: str, seq: int = 1) -> StoredMessage:
    return StoredMessage(role=role, content=content, seq=seq)  # type: ignore[arg-type]


def test_window_empty_and_zero_budgets() -> None:
    """Empty input or zero budgets return empty list."""
    assert select_history([]) == []
    msgs = [_msg("user", "hi", 1), _msg("assistant", "hello", 2)]
    assert select_history(msgs, max_messages=0) == []
    assert select_history(msgs, max_chars=0) == []


def test_window_negative_budgets_raise() -> None:
    """Negative budgets raise ValueError."""
    with pytest.raises(ValueError, match="max_messages"):
        select_history([], max_messages=-1)
    with pytest.raises(ValueError, match="max_chars"):
        select_history([], max_chars=-5)


def test_window_never_splits_pairs_or_starts_with_assistant() -> None:
    """Dropping old history removes whole (user, assistant) pairs and skips leading assistant."""
    msgs = [
        _msg("assistant", "orphan", 1),
        _msg("user", "u1", 2),
        _msg("assistant", "a1", 3),
        _msg("user", "u2", 4),
        _msg("assistant", "a2", 5),
    ]

    # Budget of 3 messages can only fit the newest 2-message turn (u2, a2)
    res = select_history(msgs, max_messages=3)
    assert [m.content for m in res] == ["u2", "a2"]
    assert res[0].role == "user"

    # Budget of 4 messages fits both complete turns, still skipping leading orphan assistant
    res_all = select_history(msgs, max_messages=4)
    assert [m.content for m in res_all] == ["u1", "a1", "u2", "a2"]

    # Only orphan assistant messages returns []
    assert select_history([_msg("assistant", "only", 1)]) == []


def test_window_newest_turn_exceeds_budget_returns_empty() -> None:
    """If the newest turn exceeds max_chars or max_messages=1, return []."""
    msgs = [
        _msg("user", "u1", 1),
        _msg("assistant", "a1", 2),
        _msg("user", "u2" * 20, 3),
        _msg("assistant", "a2" * 20, 4),
    ]
    assert select_history(msgs, max_chars=10) == []
    assert select_history(msgs, max_messages=1) == []


def test_window_char_budget_and_to_llm_messages() -> None:
    """Character budget selects most recent turns fitting within max_chars and converts cleanly."""
    msgs = [
        _msg("user", "12345", 1),
        _msg("assistant", "12345", 2),  # turn 1: 10 chars
        _msg("user", "1234", 3),
        _msg("assistant", "1234", 4),  # turn 2: 8 chars
        _msg("user", "123", 5),
        _msg("assistant", "123", 6),  # turn 3: 6 chars
    ]
    # Budget 15 fits turn 3 (6) + turn 2 (8) = 14 <= 15
    selected = select_history(msgs, max_chars=15)
    assert [m.seq for m in selected] == [3, 4, 5, 6]

    llm_msgs = to_llm_messages(selected)
    assert [(m.role, m.content) for m in llm_msgs] == [
        ("user", "1234"),
        ("assistant", "1234"),
        ("user", "123"),
        ("assistant", "123"),
    ]


def test_window_property_style_random_inputs() -> None:
    """Property-style test over 200 random message sequences and budgets."""
    rng = random.Random(42)

    for _ in range(200):
        length = rng.randint(0, 30)
        msgs: list[StoredMessage] = []
        for idx in range(length):
            role = rng.choice(["user", "assistant"])
            content = "x" * rng.randint(0, 50)
            msgs.append(_msg(role, content, seq=idx + 1))

        max_msgs = rng.choice([None, 0, 1, 2, 4, 7, 15])
        max_chars = rng.choice([None, 0, 10, 50, 150, 500])

        window = select_history(msgs, max_messages=max_msgs, max_chars=max_chars)

        # Invariant 1: Never starts with assistant
        if window:
            assert window[0].role == "user"

        # Invariant 2: Budgets respected
        if max_msgs is not None:
            assert len(window) <= max_msgs
        if max_chars is not None:
            assert sum(len(m.content) for m in window) <= max_chars

        # Invariant 3: Idempotent
        assert select_history(window, max_messages=max_msgs, max_chars=max_chars) == window

        # Invariant 4: Suffix property (window is a suffix of messages after any trailing/orphan cleanup)
        if window:
            assert window == msgs[-len(window):]
