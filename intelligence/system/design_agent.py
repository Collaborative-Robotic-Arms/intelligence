"""
design_agent.py
===============
ReAct agent for shape design using LangGraph.

The agent has 4 tools and reasons iteratively:
  draw_shape       — ASCII art → cells
  transform_shape  — rotate/scale/thicken/mirror/translate cells
  repair_shape     — auto-fix tileability issues
  try_solve        — test if solver can tile cells with current inventory

The agent loops: think → call tool → observe → think again, until it
finds a working design or hits the max iteration limit.

This module exposes only the design portion. Chat/error intents are
handled by the lighter classifier in interpreter_chain.py to save tokens.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TypedDict, Annotated

from config import cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    success:    bool
    cells:      list = field(default_factory=list)   # final tileable cells
    structure:  str  = "design"
    description:str  = ""
    placements: list = field(default_factory=list)   # solver output if try_solve found one
    steps:      list = field(default_factory=list)   # the trace of think→act→observe
    error:      str  = ""


# ─────────────────────────────────────────────────────────────────────────────
# Agent state (LangGraph)
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State that flows through the graph."""
    user_input:       str
    inventory:        dict
    current_scenario: str

    # Conversation messages (system + human + ai + tool messages)
    messages:         list

    # Cumulative tool trace (what to show in the UI)
    steps:            list

    # Final result fields
    cells:            list
    structure:        str
    description:      str
    placements:       list
    iteration:        int


MAX_ITERATIONS = 6


# ─────────────────────────────────────────────────────────────────────────────
# System prompt for the agent
# ─────────────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are a shape-design agent for a robotic LEGO assembly system.
Your job: take a natural-language shape request and produce a buildable design.

You have 4 tools and reason iteratively: think → call tool → observe → think again.

═══════════════════════════════════════════════════════════════
TOOLS
═══════════════════════════════════════════════════════════════

1) draw_shape — convert ASCII art to grid cells
   Args: {"ascii_grid": "XXXX\\n.X..\\n.X.."}
   X = filled cell, . = empty. Top row = top of shape.
   Returns: {"cells": [[col,row], ...]}

2) transform_shape — modify existing cells
   Args: {"cells": [[0,0],...], "operation": "<op>", "params": {...}}
   Operations:
     "thicken"   params {"axis": "x"|"y"|"both", "amount": 1}
     "rotate"    params {"degrees": 90|180|270}
     "translate" params {"dx": int, "dy": int}
     "scale"     params {"factor": 2}
     "mirror"    params {"axis": "x"|"y"}

3) repair_shape — verify and auto-fix tileability
   Args: {"cells": [[0,0],...]}
   Returns: {"cells": [...], "valid": bool, "issues": [...]}

4) try_solve — test if solver can tile cells with available bricks
   Args: {"cells": [[0,0],...]}
   Returns: {"success": bool, "placements": [...], "reason": "..."}

═══════════════════════════════════════════════════════════════
STRATEGY
═══════════════════════════════════════════════════════════════

For ANY shape request:
  Step 1: draw_shape with ASCII art of the shape
  Step 2: repair_shape on the result (auto-fixes isolated cells)
  Step 3: try_solve on the repaired cells
  Step 4a: if try_solve succeeds → finish with action="finish"
  Step 4b: if try_solve fails → adjust (thicken, redraw, scale up) and retry

If try_solve fails repeatedly, the shape may need:
  - Thicker features (1-cell-wide → 2-cell-wide)
  - Even cell counts in each region
  - More inventory (but you must work with what's available)

═══════════════════════════════════════════════════════════════
RESPONSE FORMAT — EVERY TURN OUTPUTS JSON
═══════════════════════════════════════════════════════════════

To call a tool:
{
  "thought": "<your reasoning>",
  "action": "draw_shape" | "transform_shape" | "repair_shape" | "try_solve",
  "args": { ... }
}

To finish (only after try_solve succeeded):
{
  "thought": "<your reasoning>",
  "action": "finish",
  "structure": "<short name>",
  "description": "<one sentence>",
  "cells": [[col,row], ...]
}

═══════════════════════════════════════════════════════════════
ASCII DRAWING TIPS
═══════════════════════════════════════════════════════════════

Make features at least 2 cells thick where possible. Examples:

T-shape:        L-shape:        Square frame:
XXXX            XX              XXXX
.X..            X.              X..X
.X..            XX              XXXX
                X.

If the inventory is small (e.g. only 4 I-bricks + 2 L-bricks), keep the
shape under ~10 cells total. The smaller bricks tile most things.

═══════════════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════════════
- ALWAYS output valid JSON, nothing else
- ALWAYS verify with try_solve before finishing
- Never guess cell coordinates manually — always use draw_shape
"""


# ─────────────────────────────────────────────────────────────────────────────
# Build the LLM
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm():
    if cfg.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key      = cfg.groq_api_key,
            model_name   = cfg.llm_model,
            temperature  = 0.2,
            max_tokens   = 2000,
            model_kwargs = {"response_format": {"type": "json_object"}},
        )
    if cfg.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url    = cfg.ollama_base_url,
            model       = cfg.ollama_model,
            temperature = 0.2,
            format      = "json",
        )
    raise ValueError("No LLM configured")


# ─────────────────────────────────────────────────────────────────────────────
# Tool execution
# ─────────────────────────────────────────────────────────────────────────────

def _run_tool(tool_name: str, args: dict, inventory: dict) -> dict:
    """Execute a tool and return its result as a dict."""
    from shape_tools import draw_shape_on_grid, modify_cells, verify_and_repair
    from brick_solver import solve_shape, explain_failure

    try:
        if tool_name == "draw_shape":
            ascii_grid = args.get("ascii_grid", "")
            cells = draw_shape_on_grid(ascii_grid)
            return {"cells": [list(c) for c in cells], "count": len(cells)}

        if tool_name == "transform_shape":
            cells = [tuple(c) for c in args.get("cells", [])
                     if isinstance(c, (list, tuple)) and len(c) == 2]
            op = args.get("operation", "thicken")
            params = args.get("params", {})
            result = modify_cells(cells, op, params)
            return {"cells": [list(c) for c in result], "count": len(result)}

        if tool_name == "repair_shape":
            cells = [tuple(c) for c in args.get("cells", [])
                     if isinstance(c, (list, tuple)) and len(c) == 2]
            report = verify_and_repair(cells, auto_repair=True)
            return {
                "cells":           [list(c) for c in report["cells"]],
                "valid":           report["valid"],
                "issues":          report["issues"],
                "repairs_applied": report["repairs_applied"],
            }

        if tool_name == "try_solve":
            cells = [tuple(c) for c in args.get("cells", [])
                     if isinstance(c, (list, tuple)) and len(c) == 2]
            result = solve_shape(cells, inventory)
            if result.success:
                return {
                    "success":    True,
                    "placements": result.placements,
                    "brick_count": len(result.placements),
                    "bricks_used": [p["brick"] for p in result.placements],
                }
            return {
                "success":      False,
                "reason":       result.reason,
                "explanation":  explain_failure(result, len(cells)),
                "untiled_cells": [list(c) for c in result.untiled_cells[:8]],
            }

        return {"error": f"Unknown tool: {tool_name}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph nodes
# ─────────────────────────────────────────────────────────────────────────────

def _think_node(state: AgentState) -> dict:
    """The LLM reasons and outputs a JSON action."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    llm = _make_llm()

    # Build messages for the LLM
    if not state["messages"]:
        # First turn — set up the conversation
        msgs = [
            SystemMessage(content=AGENT_SYSTEM_PROMPT),
            HumanMessage(content=
                f"User request: \"{state['user_input']}\"\n"
                f"Available inventory: I={state['inventory'].get('I', 0)}, "
                f"L={state['inventory'].get('L', 0)}, "
                f"T={state['inventory'].get('T', 0)}\n"
                f"Current scenario: {state['current_scenario']}\n\n"
                f"Begin designing. Output JSON for your first tool call.")
        ]
    else:
        msgs = state["messages"]

    response = llm.invoke(msgs)

    # Append response to message history
    new_messages = list(msgs) + [response]

    return {
        "messages":  new_messages,
        "iteration": state["iteration"] + 1,
    }


def _act_node(state: AgentState) -> dict:
    """Parse the LLM's last message, execute the tool, append result."""
    from langchain_core.messages import HumanMessage

    last = state["messages"][-1]
    raw  = last.content if hasattr(last, "content") else str(last)

    # Parse JSON
    data = _parse_json(raw)
    if data is None:
        # Force the agent to stop with an error
        return {
            "messages": state["messages"] + [
                HumanMessage(content="ERROR: Could not parse your JSON. "
                                     "Please output valid JSON only.")
            ],
            "steps": state["steps"] + [{
                "type": "parse_error",
                "raw": raw[:200],
            }],
        }

    action  = data.get("action", "")
    thought = data.get("thought", "")

    # ── Finish action ──
    if action == "finish":
        return {
            "structure":   data.get("structure", "design"),
            "description": data.get("description", ""),
            "cells":       data.get("cells", state.get("cells", [])),
            "steps": state["steps"] + [{
                "type": "finish",
                "thought": thought,
                "data": data,
            }],
        }

    # ── Tool call ──
    args = data.get("args", {})
    tool_result = _run_tool(action, args, state["inventory"])
    logger.info(f"[Agent] iteration {state['iteration']}: {action} → "
                f"{json.dumps(tool_result)[:120]}")

    # Track step for UI display
    step = {
        "type":     "tool_call",
        "action":   action,
        "thought":  thought,
        "args":     args,
        "result":   tool_result,
    }

    # Update cells if the tool produced them
    new_cells = state.get("cells", [])
    if isinstance(tool_result, dict) and "cells" in tool_result:
        new_cells = tool_result["cells"]

    new_placements = state.get("placements", [])
    if action == "try_solve" and tool_result.get("success"):
        new_placements = tool_result.get("placements", [])

    # Add tool result to message history (as Human turn — Groq doesn't always like ToolMessage in JSON mode)
    feedback = HumanMessage(content=
        f"Tool '{action}' result:\n{json.dumps(tool_result, indent=2)[:1500]}\n\n"
        f"Continue. Output JSON for the next action, "
        f"or use 'finish' if try_solve succeeded.")

    return {
        "messages":   state["messages"] + [feedback],
        "steps":      state["steps"] + [step],
        "cells":      new_cells,
        "placements": new_placements,
    }


def _should_continue(state: AgentState) -> str:
    """Decide if we keep looping or finish."""
    if state["iteration"] >= MAX_ITERATIONS:
        return "end"

    # Check the last step — if it's a finish, end
    if state["steps"]:
        last_step = state["steps"][-1]
        if last_step.get("type") == "finish":
            return "end"

    return "think"


def _build_graph():
    """Create the LangGraph state machine."""
    from langgraph.graph import StateGraph, END

    graph = StateGraph(AgentState)
    graph.add_node("think", _think_node)
    graph.add_node("act",   _act_node)

    graph.set_entry_point("think")
    graph.add_edge("think", "act")
    graph.add_conditional_edges("act", _should_continue, {
        "think": "think",
        "end":   END,
    })

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

_graph = None

def design_shape(
    user_input:       str,
    inventory:        dict,
    current_scenario: str = "none",
) -> AgentResult:
    """
    Run the ReAct design agent. Returns final result with full trace.
    """
    global _graph
    if _graph is None:
        _graph = _build_graph()

    initial: AgentState = {
        "user_input":       user_input,
        "inventory":        inventory,
        "current_scenario": current_scenario,
        "messages":         [],
        "steps":            [],
        "cells":            [],
        "structure":        "",
        "description":      "",
        "placements":       [],
        "iteration":        0,
    }

    try:
        final_state = _graph.invoke(initial, config={"recursion_limit": 50})
    except Exception as exc:
        import traceback
        logger.error(f"[Agent] graph invoke failed: {exc}\n{traceback.format_exc()}")
        return AgentResult(
            success = False,
            error   = f"Agent error: {type(exc).__name__}: {exc}",
        )

    cells      = final_state.get("cells", [])
    placements = final_state.get("placements", [])
    steps      = final_state.get("steps", [])

    success = bool(placements) and bool(cells)

    return AgentResult(
        success     = success,
        cells       = cells,
        structure   = final_state.get("structure", "design"),
        description = final_state.get("description", ""),
        placements  = placements,
        steps       = steps,
        error       = "" if success else "Agent did not produce a valid design",
    )


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


def format_steps_for_display(steps: list) -> str:
    """Render the agent's reasoning trace as markdown for the chat UI."""
    if not steps:
        return ""

    lines = ["**🤖 Agent reasoning trace:**\n"]

    for i, step in enumerate(steps, 1):
        if step["type"] == "tool_call":
            thought = step.get("thought", "")
            action  = step.get("action", "")
            result  = step.get("result", {})

            lines.append(f"\n**Step {i}: 🤔 {thought}**")
            lines.append(f"\n→ Called `{action}` ")

            # Concise result
            if "error" in result:
                lines.append(f"❌ _{result['error']}_")
            elif action == "try_solve":
                if result.get("success"):
                    bricks = result.get("bricks_used", [])
                    lines.append(f"✅ _solver succeeded — "
                                 f"{result.get('brick_count', 0)} bricks: "
                                 f"{', '.join(bricks)}_")
                else:
                    lines.append(f"❌ _solver failed: "
                                 f"{result.get('explanation', result.get('reason', ''))[:100]}_")
            elif "cells" in result:
                lines.append(f"_→ {result.get('count', len(result['cells']))} cells_")
                if result.get("issues"):
                    lines.append(f"\n  ⚠️ {', '.join(result['issues'])}")
                if result.get("repairs_applied"):
                    lines.append(f"\n  🔧 {', '.join(result['repairs_applied'])}")

        elif step["type"] == "finish":
            lines.append(f"\n**Step {i}: ✅ {step.get('thought', 'Done.')}**")

        elif step["type"] == "parse_error":
            lines.append(f"\n**Step {i}: ⚠️ JSON parse error**")

    return "\n".join(lines)