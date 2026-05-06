"""
session_store.py
================
Persistent session storage — saves and restores key session data
to/from a JSON file so work survives a Streamlit restart.

What gets saved:
  - Chat history (last 50 messages)
  - Current assembly plan (structure + arrangement)
  - Brick inventory (manual overrides)
  - Conversation memory context
  - Session metadata (timestamp, plan count)

What does NOT get saved:
  - ROS connection state (always re-established on startup)
  - LLM chain instances (always re-created)
  - Agent executor (always re-created)
  - Mock bridge state (always re-initialised)

Storage: .session_data/ directory next to app.py (gitignored)

Usage:
    from session_store import SessionStore
    store = SessionStore()

    store.save(st.session_state)         # call on every plan/execution
    store.load_into(st.session_state)    # call once at startup
    store.clear()                        # wipe saved data
    store.list_sessions()                # show available saves
"""

from __future__ import annotations

import json
import os
import time
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Optional

logger = logging.getLogger(__name__)

# Storage directory — next to app.py
STORE_DIR  = Path(__file__).parent / ".session_data"
INDEX_FILE = STORE_DIR / "index.json"
MAX_SESSIONS = 10   # keep last 10 sessions


class SessionStore:
    """
    Simple JSON-based session persistence.
    Each session is stored as a separate JSON file.
    An index file tracks all sessions with metadata.
    """

    def __init__(self, session_id: Optional[str] = None):
        STORE_DIR.mkdir(exist_ok=True)
        self.session_id   = session_id or _make_session_id()
        self.session_file = STORE_DIR / f"{self.session_id}.json"

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, ss) -> bool:
        """
        Serialise key session_state fields to JSON.

        Args:
            ss : st.session_state (or any dict-like object)

        Returns:
            True on success, False on error.
        """
        try:
            data = _serialise(ss)
            with open(self.session_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            _update_index(self.session_id, data)
            logger.info(f"[SessionStore] saved to {self.session_file.name}")
            return True
        except Exception as exc:
            logger.error(f"[SessionStore] save failed: {exc}")
            return False

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_into(self, ss) -> bool:
        """
        Restore saved session data into session_state.

        Args:
            ss : st.session_state

        Returns:
            True if data was found and loaded, False otherwise.
        """
        if not self.session_file.exists():
            return False

        try:
            with open(self.session_file) as f:
                data = json.load(f)
            _deserialise_into(data, ss)
            logger.info(f"[SessionStore] loaded from {self.session_file.name}")
            return True
        except Exception as exc:
            logger.error(f"[SessionStore] load failed: {exc}")
            return False

    # ── Management ────────────────────────────────────────────────────────────

    def clear(self):
        """Delete this session's saved data."""
        if self.session_file.exists():
            self.session_file.unlink()
            logger.info(f"[SessionStore] deleted {self.session_file.name}")

    @staticmethod
    def clear_all():
        """Delete all saved sessions."""
        if STORE_DIR.exists():
            for f in STORE_DIR.glob("*.json"):
                f.unlink()
        logger.info("[SessionStore] all sessions cleared")

    @staticmethod
    def list_sessions() -> list[dict]:
        """
        Return metadata for all saved sessions, newest first.
        Each entry: {id, timestamp, plan_count, last_structure, message_count}
        """
        if not INDEX_FILE.exists():
            return []
        try:
            with open(INDEX_FILE) as f:
                index = json.load(f)
            sessions = list(index.values())
            sessions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            return sessions[:MAX_SESSIONS]
        except Exception:
            return []

    @staticmethod
    def load_session(session_id: str, ss) -> bool:
        """Load a specific session by ID."""
        store = SessionStore(session_id=session_id)
        return store.load_into(ss)


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _serialise(ss) -> dict:
    """Extract serialisable data from session_state."""

    # Chat history — last 50 messages, user/assistant/system only
    chat = []
    for msg in (ss.get("chat_history") or [])[-50:]:
        try:
            chat.append({
                "role":      msg.role,
                "content":   msg.content,
                "timestamp": msg.timestamp,
                "metadata":  msg.metadata,
            })
        except AttributeError:
            if isinstance(msg, dict):
                chat.append(msg)

    # Current plan
    plan_data = None
    plan = ss.get("current_plan")
    if plan is not None:
        try:
            plan_data = {
                "structure":          plan.structure,
                "required_bricks":    plan.required_bricks,
                "arrangement":        plan.arrangement,
                "validated":          plan.validated,
                "validation_message": plan.validation_message,
            }
        except AttributeError:
            pass

    # Inventory
    inv = ss.get("inventory")
    inv_data = {"I": 0, "L": 0, "T": 0, "Z": 0}
    if inv is not None:
        try:
            inv_data = inv.as_dict()
        except Exception:
            pass

    # Memory context
    mem = ss.get("conversation_memory")
    mem_context = {}
    if mem is not None:
        try:
            mem_context = {
                k: v for k, v in mem.context.items()
                if k != "last_plan"   # don't double-store plan
            }
        except Exception:
            pass

    return {
        "session_id":      ss.get("_session_id", "unknown"),
        "timestamp":       time.time(),
        "chat_history":    chat,
        "plan":            plan_data,
        "inventory":       inv_data,
        "manual_inventory": ss.get("manual_inventory", False),
        "ros_mock_mode":   ss.get("ros_mock_mode", True),
        "memory_context":  mem_context,
        "metadata": {
            "message_count":  len(chat),
            "plan_count":     mem_context.get("executions", 0),
            "last_structure": plan_data["structure"] if plan_data else "",
        },
    }


def _deserialise_into(data: dict, ss):
    """Restore serialised data into session_state."""
    from state import ChatMessage, AssemblyPlan, BrickInventory
    import time as time_mod

    # Chat history
    history = []
    for m in data.get("chat_history", []):
        try:
            history.append(ChatMessage(
                role      = m["role"],
                content   = m["content"],
                timestamp = m.get("timestamp", time_mod.time()),
                metadata  = m.get("metadata", {}),
            ))
        except Exception:
            pass
    if history:
        ss["chat_history"] = history

    # Plan
    plan_data = data.get("plan")
    if plan_data:
        plan = AssemblyPlan(
            structure          = plan_data.get("structure", ""),
            required_bricks    = plan_data.get("required_bricks", []),
            arrangement        = plan_data.get("arrangement", []),
            validated          = plan_data.get("validated", False),
            validation_message = plan_data.get("validation_message", ""),
        )
        ss["current_plan"] = plan
        ss["show_layout"]  = True

    # Inventory
    inv_data = data.get("inventory", {})
    if inv_data:
        ss["inventory"] = BrickInventory(
            I = inv_data.get("I", 0),
            L = inv_data.get("L", 0),
            T = inv_data.get("T", 0),
            Z = inv_data.get("Z", 0),
        )

    # Flags
    if "manual_inventory" in data:
        ss["manual_inventory"] = data["manual_inventory"]
    if "ros_mock_mode" in data:
        ss["ros_mock_mode"] = data["ros_mock_mode"]

    # Memory context
    mem_ctx = data.get("memory_context", {})
    if mem_ctx and "conversation_memory" in ss:
        try:
            ss["conversation_memory"].context.update(mem_ctx)
        except Exception:
            pass


def _make_session_id() -> str:
    """Generate a human-readable session ID."""
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _update_index(session_id: str, data: dict):
    """Update the session index with latest metadata."""
    try:
        index = {}
        if INDEX_FILE.exists():
            with open(INDEX_FILE) as f:
                index = json.load(f)

        index[session_id] = {
            "id":             session_id,
            "timestamp":      data.get("timestamp", 0),
            "message_count":  data.get("metadata", {}).get("message_count", 0),
            "plan_count":     data.get("metadata", {}).get("plan_count", 0),
            "last_structure": data.get("metadata", {}).get("last_structure", ""),
        }

        # Prune old sessions
        if len(index) > MAX_SESSIONS:
            oldest = sorted(index, key=lambda k: index[k].get("timestamp", 0))
            for k in oldest[:len(index) - MAX_SESSIONS]:
                del index[k]
                old_file = STORE_DIR / f"{k}.json"
                if old_file.exists():
                    old_file.unlink()

        with open(INDEX_FILE, "w") as f:
            json.dump(index, f, indent=2)
    except Exception as exc:
        logger.warning(f"[SessionStore] index update failed: {exc}")