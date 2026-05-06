"""
validator_chain.py
==================
LLM 2 — Scenario Validator

Checks performed (reachability excluded — handled by robot nodes):
  1. Inventory   — do we have enough bricks?
  2. Bounds      — do all brick cells stay within table boundary?
  3. Collision   — do any two bricks overlap on the same layer?
  4. Height      — does the stack stay under TABLE_MAX_HEIGHT?

Fast path: all 4 pass → accept immediately, no LLM.
Slow path: any fail → LLM decides reject OR suggest a fix.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from collections import Counter

from config import cfg
from workspace_constraints import (
    CheckResult,
    check_inventory,
    check_workspace_bounds,
    check_collision,
    check_height,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidatorResult:
    status:        str
    safe_to_run:   bool  = False
    summary:       str   = ""
    reason:        str   = ""
    suggestion:    str   = ""
    risk_level:    str   = "low"
    failed_checks: list  = field(default_factory=list)
    check_details: list  = field(default_factory=list)
    tokens_used:   int   = 0
    llm_used:      bool  = False


class ValidatorChain:

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
                temperature  = 0.1,
                max_tokens   = 512,
                model_kwargs = {"response_format": {"type": "json_object"}},
            )
        elif cfg.llm_provider == "ollama":
            from langchain_ollama import ChatOllama
            self._llm = ChatOllama(
                base_url    = cfg.ollama_base_url,
                model       = cfg.ollama_model,
                temperature = 0.1,
                format      = "json",
            )
        else:
            raise ValueError("No LLM configured")

        self._initialised = True

    def validate(self, plan, inventory: dict) -> ValidatorResult:
        """Run 4 geometry+inventory checks. Reachability left to robot nodes."""
        arrangement     = getattr(plan, "arrangement",     [])
        required_bricks = getattr(plan, "required_bricks", [])
        structure       = getattr(plan, "structure",       "unknown")

        checks = [
            check_inventory(required_bricks, inventory),
            check_workspace_bounds(arrangement),
            check_collision(arrangement),
            check_height(arrangement),
        ]
        failed = [c for c in checks if not c.passed]

        if not failed:
            needed  = Counter(required_bricks)
            summary = (
                f"{structure} ready — "
                + ", ".join(f"{v}×{k}" for k, v in sorted(needed.items()))
            )
            return ValidatorResult(
                status="accept", safe_to_run=True,
                summary=summary,
                reason="All checks passed: inventory, bounds, collision, height.",
                risk_level="low",
                check_details=checks, llm_used=False,
            )

        return self._llm_decide(structure, arrangement, required_bricks,
                                inventory, checks, failed)

    def _llm_decide(self, structure, arrangement, required_bricks,
                    inventory, checks, failed):
        self._ensure_init()
        from prompts.validator import SYSTEM_PROMPT, VALIDATION_TEMPLATE, format_checks
        user_msg = VALIDATION_TEMPLATE.format(
            structure          = structure,
            required_bricks    = required_bricks,
            num_placements     = len(arrangement),
            inv_I=inventory.get("I",0), inv_L=inventory.get("L",0),
            inv_T=inventory.get("T",0), inv_Z=inventory.get("Z",0),
            check_results_text = format_checks(checks),
        )
        messages = [
            self._SystemMessage(content=SYSTEM_PROMPT),
            self._HumanMessage(content=user_msg),
        ]
        try:
            response = self._llm.invoke(messages)
            raw      = response.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(l for l in raw.split("\n")
                                 if not l.strip().startswith("```")).strip()
            data   = json.loads(raw)
            status = data.get("status", "reject")
            return ValidatorResult(
                status        = status,
                safe_to_run   = (status == "accept"),
                summary       = data.get("summary", ""),
                reason        = data.get("reason",  ""),
                suggestion    = data.get("suggestion", ""),
                risk_level    = data.get("risk_level", "high"),
                failed_checks = [c.code for c in failed],
                check_details = checks,
                tokens_used   = (response.usage_metadata or {}).get("output_tokens", 0),
                llm_used      = True,
            )
        except Exception as exc:
            logger.error(f"[Validator] LLM error: {exc}")
            return self._fallback(failed, checks)

    def _fallback(self, failed, checks, tokens=0):
        hard = {"INVENTORY_INSUFFICIENT","COLLISION_DETECTED","HEIGHT_EXCEEDED"}
        is_hard = any(c.code in hard for c in failed)
        return ValidatorResult(
            status        = "reject" if is_hard else "suggest",
            safe_to_run   = False,
            summary       = f"{len(failed)} check(s) failed.",
            reason        = "; ".join(f"{c.message}: {c.detail}" for c in failed),
            suggestion    = "" if is_hard else "Adjust positions and try again.",
            risk_level    = "high" if is_hard else "medium",
            failed_checks = [c.code for c in failed],
            check_details = checks,
            tokens_used   = tokens, llm_used=False,
        )


def apply_validation_to_plan(plan, result: ValidatorResult):
    plan.validated          = result.safe_to_run
    plan.validation_message = result.summary or result.reason


def format_validation_chat_message(result: ValidatorResult) -> str:
    icons  = {"accept":"✅","suggest":"💡","reject":"❌"}
    icon   = icons.get(result.status,"⚠️")
    lines  = [f"{icon} **Validation: {result.status.upper()}**"]
    if result.summary:
        lines.append(f"\n{result.summary}")
    lines.append("\n**Checks:**")
    for c in result.check_details:
        tick = "✓" if c.passed else "✗"
        lines.append(f"  {tick} {c.message}"
                     + (f"\n    ↳ _{c.detail}_" if not c.passed and c.detail else ""))
    if result.reason and result.status != "accept":
        lines.append(f"\n**Reason:** {result.reason}")
    if result.suggestion:
        lines.append(f"\n**Suggestion:** {result.suggestion}")
        lines.append("\nWould you like me to apply this fix and regenerate the plan?")
    if result.safe_to_run:
        lines.append("\n\n✅ Plan is safe — press **▶ Execute plan** when ready.")
    source = "Groq LLM" if result.llm_used else "geometry checks"
    lines.append(f"\n\n_Validated by {source}_")
    return "\n".join(lines)