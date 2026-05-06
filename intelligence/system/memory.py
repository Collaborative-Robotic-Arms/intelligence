"""
memory.py
=========
Conversation memory for the assembly AI interface.

Provides two things:

1. ConversationMemory — wraps LangChain ConversationBufferWindowMemory
   to give the interpreter and agent context about what was discussed
   in previous turns. Supports:
     - "rotate it 90 degrees"   → knows what "it" refers to
     - "same layout but with L bricks"  → knows the previous plan
     - "what did I build last time?"  → can answer from history

2. build_langchain_history() — converts state.ChatMessage list into
   LangChain HumanMessage/AIMessage objects the agent and interpreter
   can use directly as chat_history.

Usage:
    from memory import ConversationMemory
    mem = ConversationMemory()

    mem.add_user("Build a T shape")
    mem.add_assistant("Plan ready: T-shape, 3 I-bricks")

    lc_history = mem.as_langchain_messages()
    # → [HumanMessage("Build a T shape"), AIMessage("Plan ready...")]

    summary = mem.get_context_summary()
    # → "User built T-shape (3 I-bricks) on AR4 side."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Max turns to keep in active memory (each turn = 1 human + 1 AI message)
MAX_WINDOW_TURNS = 8


@dataclass
class MemoryTurn:
    """A single conversation turn (human + assistant pair)."""
    human:     str
    assistant: str
    metadata:  dict = field(default_factory=dict)
    # e.g. {"plan_structure": "T-shape", "bricks_placed": [1,2,3]}


class ConversationMemory:
    """
    Sliding window conversation memory.

    Keeps the last MAX_WINDOW_TURNS turns (human + assistant pairs)
    and converts them to LangChain message objects for injection
    into interpreter and agent prompts.

    Also maintains a structured context dict that tracks:
      - last_structure  : most recent assembly structure name
      - last_plan       : most recent AssemblyPlan arrangement
      - last_inventory  : inventory at last planning time
      - bricks_placed   : IDs of bricks placed in last execution
    """

    def __init__(self):
        self._turns: list[MemoryTurn] = []
        self.context: dict = {
            "last_structure":  "",
            "last_plan":       None,
            "last_inventory":  {},
            "bricks_placed":   [],
            "executions":      0,
        }

    # ── Add turns ─────────────────────────────────────────────────────────────

    def add_user(self, text: str):
        """Start a new turn with a user message."""
        self._turns.append(MemoryTurn(human=text, assistant=""))

    def add_assistant(self, text: str, metadata: dict = None):
        """Complete the current turn with an assistant response."""
        if self._turns and not self._turns[-1].assistant:
            self._turns[-1].assistant = text
            if metadata:
                self._turns[-1].metadata = metadata
        else:
            # Unpaired assistant message — create a turn anyway
            self._turns.append(MemoryTurn(
                human="[context]", assistant=text,
                metadata=metadata or {}
            ))

        # Trim to window size
        if len(self._turns) > MAX_WINDOW_TURNS:
            self._turns = self._turns[-MAX_WINDOW_TURNS:]

    def update_context(self, key: str, value):
        """Update a context field (e.g. after plan generation or execution)."""
        self.context[key] = value

    # ── Query ─────────────────────────────────────────────────────────────────

    def as_langchain_messages(self) -> list:
        """
        Convert memory turns to LangChain HumanMessage / AIMessage objects.
        Only includes complete turns (both human and assistant filled).
        """
        from langchain_core.messages import HumanMessage, AIMessage
        msgs = []
        for turn in self._turns:
            if turn.human and turn.assistant:
                msgs.append(HumanMessage(content=turn.human))
                msgs.append(AIMessage(content=turn.assistant))
        return msgs

    def get_context_summary(self) -> str:
        """
        One-paragraph summary of conversation context.
        Injected into interpreter prompt when is_refinement=True.
        """
        parts = []
        if self.context["last_structure"]:
            parts.append(f"Last structure: {self.context['last_structure']}")
        if self.context["last_inventory"]:
            inv = self.context["last_inventory"]
            parts.append(
                f"Inventory at last plan: "
                f"I={inv.get('I',0)} L={inv.get('L',0)} "
                f"T={inv.get('T',0)} Z={inv.get('Z',0)}"
            )
        if self.context["bricks_placed"]:
            parts.append(
                f"Last execution placed bricks: {self.context['bricks_placed']}"
            )
        if self.context["executions"] > 0:
            parts.append(
                f"Total executions this session: {self.context['executions']}"
            )
        return " | ".join(parts) if parts else "No prior context."

    def get_recent_turns_text(self, n: int = 3) -> str:
        """Last n turns as plain text (for debugging / display)."""
        recent = [t for t in self._turns if t.assistant][-n:]
        lines  = []
        for i, turn in enumerate(recent, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  User: {turn.human[:120]}")
            lines.append(f"  AI:   {turn.assistant[:120]}")
        return "\n".join(lines) or "No history yet."

    def clear(self):
        """Wipe all memory — call when user explicitly starts a new session."""
        self._turns = []
        self.context = {
            "last_structure":  "",
            "last_plan":       None,
            "last_inventory":  {},
            "bricks_placed":   [],
            "executions":      0,
        }

    @property
    def turn_count(self) -> int:
        return len([t for t in self._turns if t.assistant])

    @property
    def is_empty(self) -> bool:
        return self.turn_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: convert state.ChatMessage list → LangChain messages
# ─────────────────────────────────────────────────────────────────────────────

def build_langchain_history(
    chat_history: list,
    max_turns:    int = MAX_WINDOW_TURNS,
) -> list:
    """
    Convert a list of state.ChatMessage objects into LangChain message objects.

    Filters:
      - Only "user" and "assistant" roles (skips "system" messages)
      - Keeps last max_turns pairs
      - Skips messages longer than 800 chars (validation reports etc.)
        to avoid bloating context

    Args:
        chat_history : list of state.ChatMessage
        max_turns    : maximum number of user+assistant pairs to include

    Returns:
        list of HumanMessage / AIMessage
    """
    from langchain_core.messages import HumanMessage, AIMessage

    # Filter to user/assistant only, skip very long messages
    filtered = [
        m for m in chat_history
        if m.role in ("user", "assistant")
        and len(m.content) <= 800
    ]

    # Keep last max_turns * 2 messages (pairs)
    filtered = filtered[-(max_turns * 2):]

    msgs = []
    for m in filtered:
        if m.role == "user":
            msgs.append(HumanMessage(content=m.content))
        else:
            msgs.append(AIMessage(content=m.content))

    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# Singleton accessor (one per Streamlit session via session_state)
# ─────────────────────────────────────────────────────────────────────────────

def get_memory() -> ConversationMemory:
    """
    Returns the ConversationMemory from st.session_state.
    Creates one if it doesn't exist.
    Call this from app.py instead of using a module-level singleton
    so each browser session gets its own memory.
    """
    try:
        import streamlit as st
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = ConversationMemory()
        return st.session_state.conversation_memory
    except Exception:
        # Outside Streamlit (tests) — return a fresh instance
        return ConversationMemory()