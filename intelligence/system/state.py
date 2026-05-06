"""
state.py
Central session_state schema for the robotic assembly interface.
All keys are initialised here so every module can import & use them safely.
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import Optional
import time


# ─────────────────────────────────────────────
# Data classes — typed containers used throughout
# ─────────────────────────────────────────────

@dataclass
class BrickInventory:
    """Counts of each brick type currently on the table."""
    I: int = 0
    L: int = 0
    T: int = 0
    Z: int = 0

    def total(self) -> int:
        return self.I + self.L + self.T + self.Z

    def as_dict(self) -> dict:
        return {"I": self.I, "L": self.L, "T": self.T, "Z": self.Z}

    def has_enough(self, required: dict) -> tuple[bool, list[str]]:
        """
        Check if current inventory satisfies required counts.
        Returns (ok, list_of_shortage_messages).
        """
        shortages = []
        for brick_type, needed in required.items():
            available = getattr(self, brick_type, 0)
            if available < needed:
                shortages.append(
                    f"Need {needed}×{brick_type}, only {available} available"
                )
        return len(shortages) == 0, shortages


@dataclass
class ChatMessage:
    """A single message in the conversation history."""
    role: str          # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)  # e.g. {"type": "validation_result"}


@dataclass
class AssemblyPlan:
    """The structured output from the Interpreter LLM."""
    structure: str = ""                     # e.g. "T", "bridge", "tower"
    required_bricks: list = field(default_factory=list)   # ["I", "I", "L"]
    arrangement: list = field(default_factory=list)       # [{"brick": "I", "x": 0, "y": 0, "rotation": 0}]
    validated: bool = False
    validation_message: str = ""
    # ── Phase A: declared intent ────────────────────────────────
    # The agent commits to these BEFORE we trust the design.
    # The validator checks declared vs actual.
    expected_components: int = 1       # 1 for connected shapes; 2+ for "two L's"
    expected_brick_count: int = 0      # placement count agent expected
    shape_description: str = ""        # one-sentence commitment from agent


@dataclass
class RosStatus:
    """Live connection + system status from the ROS2 bridge."""
    connected: bool = False
    mode: str = "sim"          # "sim" | "real"
    ar4_stage: str = "IDLE"    # matches supervisor_node ar4_stage
    abb_stage: str = "IDLE"
    supervisor_state: str = "INIT"
    zone_status: str = "CLEAR"
    last_update: float = 0.0

    def status_color(self) -> str:
        if not self.connected:
            return "🔴"
        if self.zone_status == "COLLISION_WARNING":
            return "🟡"
        return "🟢"


# ─────────────────────────────────────────────
# Initialiser — call once at app startup
# ─────────────────────────────────────────────

def init_state():
    """
    Initialise all session_state keys if they don't exist yet.
    Safe to call on every rerun — only sets keys that are missing.
    """
    defaults = {
        # Core data objects
        "inventory":         BrickInventory(),
        "chat_history":      [],           # list[ChatMessage]
        "current_plan":      None,         # AssemblyPlan | None
        "ros_status":        RosStatus(),

        # UI control flags
        "ros_mock_mode":     True,         # True = no real ROS needed
        "show_layout":       False,        # show 2D preview panel
        "executing":         False,        # a plan is being sent to robot
        "waiting_for_confirmation": False, # validator asked user a question

        # Agent / LLM state (used from Task 3 onward)
        "agent_chain":       None,
        "pending_suggestion": None,        # refinement suggestion awaiting user reply

        # Sidebar manual overrides
        "manual_inventory":  False,        # user typed counts manually
        "pending_execution": False,   # Phase D: explicit confirm gate before push
        
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


# ─────────────────────────────────────────────
# Convenience accessors
# ─────────────────────────────────────────────

def get_inventory() -> BrickInventory:
    return st.session_state.inventory

def get_ros_status() -> RosStatus:
    return st.session_state.ros_status

def get_chat_history() -> list:
    return st.session_state.chat_history

def add_message(role: str, content: str, metadata: dict = None):
    msg = ChatMessage(role=role, content=content, metadata=metadata or {})
    st.session_state.chat_history.append(msg)

def clear_chat():
    st.session_state.chat_history = []

def set_plan(plan: AssemblyPlan):
    st.session_state.current_plan = plan
    st.session_state.show_layout = True
    st.session_state.pending_execution = False  # Phase D

def get_plan() -> Optional[AssemblyPlan]:
    return st.session_state.current_plan
