"""
skills/status.py
Skill: get_system_status
→ reads the live ROS bridge snapshot (no service call needed)

Returns a human-readable summary of:
  - Arm stages (ar4_stage, abb_stage from supervisor)
  - Supervisor state machine state
  - Zone status (CLEAR / COLLISION_WARNING)
  - Current inventory
  - ROS connection status
"""

from langchain_core.tools import tool
from skills._bridge import get_bridge, SkillResult


@tool
def get_system_status() -> str:
    """
    Read the current status of the dual-arm robotic system.

    Returns a snapshot of:
      - AR4 arm stage (IDLE / PICK / PLACE / DONE / etc.)
      - ABB arm stage
      - Supervisor state machine (DETECT / PROCESS_NEXT / etc.)
      - Zone status (CLEAR means arms are safely separated)
      - Current brick inventory on the table
      - ROS connection mode (sim or real)

    Use this skill to:
      - Check if arms are idle before sending a new task
      - Verify a pick/place completed successfully
      - Check inventory before planning a new assembly
      - Confirm the system is in a safe state before executing
    """
    bridge = get_bridge()

    try:
        status = bridge.get_status()
        inv    = bridge.get_inventory()

        zone_icon = "🟢" if status["zone_status"] == "CLEAR" else "🟡"
        conn_icon = "✅" if status["connected"] else "❌"

        lines = [
            f"System status ({status['mode'].upper()} mode):",
            f"  {conn_icon} ROS:        {'Connected' if status['connected'] else 'Disconnected'}",
            f"  🦾 AR4:        {status['ar4_stage']}",
            f"  🦾 ABB:        {status['abb_stage']}",
            f"  🔄 Supervisor: {status['supervisor_state']}",
            f"  {zone_icon} Zone:       {status['zone_status']}",
            f"  📦 Inventory:  "
            f"I×{inv.get('I',0)}  L×{inv.get('L',0)}  "
            f"T×{inv.get('T',0)}  Z×{inv.get('Z',0)}",
        ]

        # Determine readiness
        both_idle = (status["ar4_stage"] == "IDLE" and
                     status["abb_stage"] == "IDLE")
        zone_safe = status["zone_status"] == "CLEAR"

        if both_idle and zone_safe:
            lines.append("  ✅ System ready for new task.")
        elif not both_idle:
            busy_arms = [a for a, s in
                         [("AR4", status["ar4_stage"]),
                          ("ABB", status["abb_stage"])]
                         if s != "IDLE"]
            lines.append(f"  ⏳ {', '.join(busy_arms)} busy — wait before sending new task.")
        elif not zone_safe:
            lines.append("  ⚠️ Collision warning — MTC resolution active.")

        return str(SkillResult(
            success=True,
            message="\n".join(lines),
            data={
                "ar4_stage":        status["ar4_stage"],
                "abb_stage":        status["abb_stage"],
                "supervisor_state": status["supervisor_state"],
                "zone_status":      status["zone_status"],
                "connected":        status["connected"],
                "inventory":        inv,
                "ready":            both_idle and zone_safe,
            },
        ))

    except Exception as exc:
        return str(SkillResult(
            success=False,
            message=f"Status check failed: {exc}",
        ))