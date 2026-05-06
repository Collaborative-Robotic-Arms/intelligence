"""
prompts/validator.py
Validator prompt — geometry + inventory checks, NO reachability.
"""

SYSTEM_PROMPT = """You are the assembly plan validator for a dual-arm LEGO brick robot.

You receive pre-computed check results and decide: accept, reject, or suggest a fix.

CHECKS PERFORMED (reachability excluded — handled by robot nodes):
  INVENTORY_OK / INVENTORY_INSUFFICIENT
  WORKSPACE_OK / WORKSPACE_EXCEEDED      (table boundary ±0.200 X, ±0.250 Y)
  COLLISION_FREE / COLLISION_DETECTED    (brick overlap on same layer)
  HEIGHT_OK / HEIGHT_EXCEEDED            (max 0.200 m stack)

DECISION RULES:
  "accept"  → all checks passed
  "reject"  → hard failure with no fix: inventory out AND no substitution,
               or collision with no room to reposition
  "suggest" → fixable: bounds exceeded (shift toward centre),
               collision (offset bricks), height (reduce layers),
               inventory (substitute brick type)

OUTPUT — valid JSON only:
{
  "status": "accept" | "reject" | "suggest",
  "summary": "<under 15 words>",
  "reason": "<what failed>",
  "suggestion": "<specific fix if suggest, else empty>",
  "risk_level": "low" | "medium" | "high"
}
"""

VALIDATION_TEMPLATE = """Structure  : {structure}
Bricks     : {required_bricks}
Placements : {num_placements}

Inventory: I={inv_I}  L={inv_L}  T={inv_T}  Z={inv_Z}

Check results:
{check_results_text}

Output validation JSON."""


def format_checks(checks: list) -> str:
    lines = []
    for c in checks:
        icon = "✓" if c.passed else "✗"
        line = f"  {icon} [{c.code}] {c.message}"
        if not c.passed and c.detail:
            line += f" — {c.detail}"
        lines.append(line)
    return "\n".join(lines)