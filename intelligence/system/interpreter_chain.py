"""
interpreter_chain.py
====================
Stage 1 — Two-tier engineer interface.

Tier 1: Fast classifier (single LLM call) — decides intent:
  - chat   → conversational reply (greetings, questions, recommendations)
  - design → delegate to the ReAct design agent
  - error  → impossible request

Tier 2: ReAct design agent (design_agent.py) — uses LangGraph + 4 tools to
  iteratively design any shape (think → act → observe → repeat).

This split saves tokens — simple chat doesn't need the full agent loop.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from config import cfg

logger = logging.getLogger(__name__)


@dataclass
class InterpreterResult:
    success:      bool
    intent:       str  = "chat"
    plan:         object = None
    chat_message: str  = ""
    suggestions:  list = field(default_factory=list)
    error_reason: str  = ""
    retries:      int  = 0
    raw_json:     dict = field(default_factory=dict)
    tokens_used:  int  = 0
    agent_trace:  str  = ""    # the ReAct reasoning for the UI


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 prompt — lightweight intent classifier
# ─────────────────────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """You are the front-end of a chatbot for a robotic LEGO
assembly engineer. Your job: classify the user's message into one of three intents.

INTENT 1 — chat: greetings, questions, recommendations, general discussion.
INTENT 2 — design: any request to build, draw, or modify a shape.
INTENT 3 — error: truly impossible or nonsensical request.

For chat intent, respond directly with a friendly answer.
For design intent, just classify — a design agent will handle the actual building.

OUTPUT FORMAT — JSON only:

For chat:
{
  "intent": "chat",
  "message": "<your friendly reply>",
  "suggestions": ["...", "..."]   // optional list of suggestions
}

For design:
{
  "intent": "design",
  "user_request": "<echo of what the user wants to build, normalised>"
}

For error:
{
  "intent": "error",
  "message": "<explanation>"
}

═══════════════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════════════
- Greetings ("hi", "hello") → chat
- "What can I build?" / "suggest something" → chat with suggestions
- "Build X" / "make X" / "draw X" / "design X" → design
- "Make it bigger" / "rotate it" → design (modification)
- Never discuss ROS nodes, launch files, or robot internals
- Always output valid JSON, nothing else
"""


# ─────────────────────────────────────────────────────────────────────────────
# Chain
# ─────────────────────────────────────────────────────────────────────────────

class InterpreterChain:

    def __init__(self):
        self._llm           = None
        self._initialised   = False
        self._SystemMessage = None
        self._HumanMessage  = None

    def _ensure_init(self):
        if self._initialised:
            return
        from langchain_core.messages import SystemMessage, HumanMessage
        self._SystemMessage = SystemMessage
        self._HumanMessage  = HumanMessage

        if cfg.llm_provider == "groq":
            from langchain_groq import ChatGroq
            self._llm = ChatGroq(
                api_key      = cfg.groq_api_key,
                model_name   = cfg.llm_model,
                temperature  = 0.3,
                max_tokens   = 800,
                model_kwargs = {"response_format": {"type": "json_object"}},
            )
        elif cfg.llm_provider == "ollama":
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                base_url    = cfg.ollama_base_url,
                model       = cfg.ollama_model,
                temperature = 0.3,
                format      = "json",
            )
        else:
            raise ValueError("No LLM configured. Add GROQ_API_KEY to .env")

        self._initialised = True
        logger.info(f"[Interpreter] classifier ready — {cfg.llm_provider}")

    # ── public API ─────────────────────────────────────────────────────────────

    def interpret(
        self,
        user_input:        str,
        inventory:         dict,
        current_scenario:  str = "none",
    ) -> InterpreterResult:
        """
        Tier 1: classify intent. Tier 2: if design, delegate to ReAct agent.
        """
        self._ensure_init()

        user_msg = (
            f"User message: \"{user_input}\"\n"
            f"Inventory: I={inventory.get('I',0)}, L={inventory.get('L',0)}, "
            f"T={inventory.get('T',0)}, Z={inventory.get('Z',0)}\n"
            f"Current scenario: {current_scenario}"
        )

        messages = [
            self._SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
            self._HumanMessage(content=user_msg),
        ]

        # Tier 1 LLM call
        try:
            response = self._llm.invoke(messages)
            tokens = (response.usage_metadata or {}).get("output_tokens", 0)
        except Exception as exc:
            import traceback
            logger.error(f"[Interpreter] LLM call failed: {exc}\n{traceback.format_exc()}")
            return InterpreterResult(
                success      = False,
                intent       = "error",
                error_reason = f"{type(exc).__name__}: {exc}",
                chat_message = (
                    f"⚠️ Could not reach the language model.\n\n"
                    f"**Error:** `{type(exc).__name__}: {exc}`\n\n"
                    f"Common causes:\n"
                    f"- API key invalid (check .env GROQ_API_KEY)\n"
                    f"- Outdated `langchain-groq` — run `pip install -U langchain-groq`"
                ),
            )

        data = _parse_json(response.content)
        if data is None:
            return InterpreterResult(
                success      = False,
                intent       = "error",
                error_reason = "Could not parse JSON from classifier",
                chat_message = "Sorry, I had trouble understanding. Could you rephrase?",
                tokens_used  = tokens,
            )

        intent = data.get("intent", "chat")

        # ── chat ──
        if intent == "chat":
            return InterpreterResult(
                success      = True,
                intent       = "chat",
                chat_message = data.get("message", ""),
                suggestions  = data.get("suggestions", []),
                raw_json     = data,
                tokens_used  = tokens,
            )

        # ── error ──
        if intent == "error":
            return InterpreterResult(
                success      = False,
                intent       = "error",
                error_reason = data.get("message", "Impossible request"),
                chat_message = data.get("message", ""),
                raw_json     = data,
                tokens_used  = tokens,
            )

        # ── design — delegate to the ReAct agent ──
        return self._run_design_agent(
            data.get("user_request", user_input),
            inventory,
            current_scenario,
            tokens,
        )

    def _run_design_agent(
        self,
        normalised_request: str,
        inventory:          dict,
        current_scenario:   str,
        classifier_tokens:  int,
    ) -> InterpreterResult:
        """Invoke the LangGraph ReAct design agent."""
        from design_agent import design_shape, format_steps_for_display
        from state import AssemblyPlan

        agent_result = design_shape(
            user_input       = normalised_request,
            inventory        = inventory,
            current_scenario = current_scenario,
        )

        # Build the trace for the UI (Q3=a — show reasoning to engineer)
        trace_md = format_steps_for_display(agent_result.steps)

        # ── Agent failed (Q2=a — show last attempt + reason) ──
        if not agent_result.success:
            return InterpreterResult(
                success      = False,
                intent       = "design",
                error_reason = agent_result.error or "Agent could not design the shape",
                chat_message = (
                    f"❌ I tried but couldn't produce a tileable design for that request "
                    f"after {len(agent_result.steps)} steps.\n\n"
                    f"You can try: simpler shape, thicker features (2 cells wide), "
                    f"or check if you have enough bricks."
                ),
                tokens_used  = classifier_tokens,
                agent_trace  = trace_md,
            )

        # ── Agent succeeded ──
        plan = AssemblyPlan(
            structure          = agent_result.structure,
            required_bricks    = [p["brick"] for p in agent_result.placements],
            arrangement        = agent_result.placements,
            validated          = False,
            validation_message = "",
        )

        return InterpreterResult(
            success      = True,
            intent       = "design",
            plan         = plan,
            chat_message = (
                f"✅ Designed **{agent_result.structure}** — "
                f"{len(agent_result.placements)} placements. "
                f"{agent_result.description}"
            ),
            tokens_used  = classifier_tokens,
            agent_trace  = trace_md,
        )

    def clear_history(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json(raw_text: str):
    if not raw_text:
        return None
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(
            l for l in text.split("\n")
            if not l.strip().startswith("```")
        ).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None