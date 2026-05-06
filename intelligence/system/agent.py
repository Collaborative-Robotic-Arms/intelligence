"""
agent.py
========
LangChain ReAct agent — the orchestration brain of the system.

Responsibilities:
  1. Receive a validated AssemblyPlan
  2. Reason step-by-step (Thought → Action → Observation loop)
  3. Call skill tools in the correct sequence:
       status → detect → grasp → pick → place       (normal bricks)
       status → detect → grasp → pick → handover → place  (Z-bricks)
  4. Stream each reasoning step into Streamlit chat as it happens
  5. Handle failures gracefully — stop and report, never guess

Model: llama-3.3-70b-versatile via Groq (same as interpreter/validator)
Pattern: LangChain create_react_agent + AgentExecutor with streaming
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    success:       bool
    summary:       str
    steps:         list[dict] = field(default_factory=list)
    bricks_placed: list[int]  = field(default_factory=list)
    errors:        list[str]  = field(default_factory=list)
    tokens_used:   int        = 0


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class AssemblyAgent:
    """
    ReAct agent that executes a validated AssemblyPlan using
    the 7 skill tools (pick, place, handover, detect, grasp, home, status).

    Streaming: each step (thought + action + observation) is pushed
    to the UI via a callback so the operator sees reasoning in real time.
    """

    def __init__(self):
        self._agent_executor = None
        self._initialised    = False

    # ── lazy init ─────────────────────────────────────────────────────────────

    def _ensure_init(self):
        if self._initialised:
            return

        if cfg.llm_provider not in ("groq", "ollama"):
            raise ValueError(
                "No LLM configured.\n"
                "Set GROQ_API_KEY in .env file.\n"
                "Get a free key at: https://console.groq.com"
            )

        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.agents import create_react_agent, AgentExecutor
        from prompts.agent import SYSTEM_PROMPT
        from skills import ALL_TOOLS

        llm    = self._make_llm()
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human",  "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_react_agent(llm, ALL_TOOLS, prompt)
        self._agent_executor = AgentExecutor(
            agent                    = agent,
            tools                    = ALL_TOOLS,
            verbose                  = False,   # we do our own streaming
            max_iterations           = 30,
            handle_parsing_errors    = True,
            return_intermediate_steps = True,
        )

        self._initialised = True
        logger.info(f"[Agent] ready — provider={cfg.llm_provider} "
                    f"model={cfg.llm_model}")

    def _make_llm(self):
        if cfg.llm_provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                api_key     = cfg.groq_api_key,
                model_name  = cfg.llm_model,
                temperature = 0.0,
                max_tokens  = 2048,
            )
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url    = cfg.ollama_base_url,
            model       = cfg.ollama_model,
            temperature = 0.0,
        )

    # ── public API ─────────────────────────────────────────────────────────────

    def execute_plan(
        self,
        plan,
        inventory:       dict,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        chat_history:    list = None,
    ) -> AgentResult:
        """
        Execute a validated AssemblyPlan step-by-step using skill tools.

        Args:
            plan            : state.AssemblyPlan (validated=True required)
            inventory       : {"I": int, "L": int, "T": int, "Z": int}
            stream_callback : fn(role, message) called after each step
            chat_history    : optional prior messages for context

        Returns:
            AgentResult
        """
        self._ensure_init()
        _inject_bridge()

        task_prompt = _build_task_prompt(plan, inventory)

        if stream_callback:
            stream_callback(
                "system",
                f"🤖 Agent starting **{plan.structure}** "
                f"({len(plan.arrangement)} bricks)…"
            )

        try:
            response = self._agent_executor.invoke({
                "input":        task_prompt,
                "chat_history": chat_history or [],
            })

            steps   = []
            placed  = []
            errors  = []

            for action, observation in response.get("intermediate_steps", []):
                step = {
                    "thought": getattr(action, "log", ""),
                    "tool":    getattr(action, "tool", ""),
                    "input":   getattr(action, "tool_input", {}),
                    "output":  str(observation),
                }
                steps.append(step)

                if stream_callback:
                    _stream_step(step, stream_callback)

                # Track placed bricks
                if step["tool"] == "place_brick" and "[OK]" in step["output"]:
                    inp = step["input"]
                    bid = inp.get("brick_id") if isinstance(inp, dict) else None
                    if bid:
                        placed.append(int(bid))

                # Track errors
                if "[FAIL]" in step["output"]:
                    errors.append(f"{step['tool']}: {step['output']}")

            final_output = response.get("output", "")
            if stream_callback and final_output:
                stream_callback("assistant", final_output)

            return AgentResult(
                success       = len(errors) == 0,
                summary       = final_output,
                steps         = steps,
                bricks_placed = placed,
                errors        = errors,
            )

        except Exception as exc:
            logger.error(f"[Agent] execution error: {exc}")
            msg = f"❌ Agent failed: {exc}"
            if stream_callback:
                stream_callback("system", msg)
            return AgentResult(success=False, summary=msg, errors=[str(exc)])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_task_prompt(plan, inventory: dict) -> str:
    """Convert AssemblyPlan → agent input string."""
    lines = [
        f"Execute the validated assembly plan below.",
        f"",
        f"Structure : {plan.structure}",
        f"Inventory : I={inventory.get('I',0)}  L={inventory.get('L',0)}"
        f"  T={inventory.get('T',0)}  Z={inventory.get('Z',0)}",
        f"",
        f"Bricks ({len(plan.arrangement)} total, in placement order):",
    ]
    for b in sorted(plan.arrangement,
                    key=lambda x: (x.get("layer", 0), x.get("id", 0))):
        z_note = " ← Z-BRICK: needs handover" \
                 if b.get("brick") == "Z" else ""
        lines.append(
            f"  id={b['id']}  {b['brick']}-brick  "
            f"x={b['x']:.3f} y={b['y']:.3f}  "
            f"rot={b.get('rotation',0)}°  "
            f"arm={b.get('start_side','AR4')}→{b.get('target_side','AR4')}  "
            f"layer={b.get('layer',0)}{z_note}"
        )
    lines += [
        "",
        "Steps: check status → detect → (grasp → pick → place) × N → home.",
        "Use parallel execution when AR4 and ABB bricks don't overlap in Y.",
    ]
    return "\n".join(lines)


def _inject_bridge():
    """Sync the Streamlit session bridge into skills module."""
    try:
        import streamlit as st
        if hasattr(st, "session_state") and "ros_bridge" in st.session_state:
            from skills._bridge import set_test_bridge
            set_test_bridge(st.session_state.ros_bridge)
    except Exception:
        pass


def _stream_step(step: dict, callback: Callable[[str, str], None]):
    """Format one ReAct step and push it to the chat UI."""
    tool   = step.get("tool", "unknown")
    output = step.get("output", "")
    log    = step.get("thought", "")

    # Extract thought text (strip "Thought:" prefix)
    thought = ""
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("Thought:"):
            thought = line[len("Thought:"):].strip()
            break
        elif line and not line.lower().startswith("action"):
            thought = line
            break

    icons = {
        "detect_bricks":     "🔍",
        "get_grasp_point":   "✋",
        "pick_brick":        "⬆️",
        "place_brick":       "⬇️",
        "handover_brick":    "🤝",
        "home_arm":          "🏠",
        "get_system_status": "📊",
    }
    icon   = icons.get(tool, "⚙️")
    status = "✅" if "[OK]" in output else "❌"

    # First line of output only (keep it concise in chat)
    first_line = output.split("\n")[0].strip()

    msg = f"{icon} **{tool}**"
    if thought:
        msg += f"\n> _{thought}_"
    msg += f"\n{status} {first_line}"

    callback("assistant", msg)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_AGENT: Optional[AssemblyAgent] = None


def get_agent() -> AssemblyAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = AssemblyAgent()
    return _AGENT