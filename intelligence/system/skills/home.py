"""
skills/home.py
Skill: home_arm
→ sends ExecuteTask(task_type="HOME")
  to ar4_control or abb_control action server
  (supervisor_node.py lines 895, 957)

Used to safely return an arm to its home position
after a pick/place sequence or after an emergency stop.
"""

from langchain_core.tools import tool
from skills._bridge import get_bridge, SkillResult


@tool
def home_arm(arm: str = "both") -> str:
    """
    Move one or both arms to their home (safe rest) position.

    The home position is defined in the SRDF/MoveIt configuration
    for each arm. Calling this after a task ensures the arms don't
    block each other for the next operation.

    Args:
        arm: Which arm to home — "AR4", "ABB", or "both" (default).
             "both" sends HOME to AR4 first, then ABB.

    Returns:
        Success/failure message for each arm.

    Call this:
      - After completing an assembly sequence
      - After an emergency stop
      - Before starting a new task if arms are in unknown positions
    """
    bridge = get_bridge()

    arm = arm.upper()
    if arm not in ("AR4", "ABB", "BOTH"):
        return str(SkillResult(
            success=False,
            message=f"Invalid arm '{arm}'. Use 'AR4', 'ABB', or 'both'.",
        ))

    arms_to_home = ["AR4", "ABB"] if arm == "BOTH" else [arm]

    try:
        if bridge.is_mock:
            return str(SkillResult(
                success=True,
                message=f"{', '.join(arms_to_home)} moved to home position.",
                data={"homed": arms_to_home, "action": "HOME"},
            ))

        # Live mode
        results = []
        all_ok  = True
        for a in arms_to_home:
            ok, msg = bridge.push_plan([{
                "task_type": "HOME",
                "arm":       a,
            }])
            results.append(f"{a}: {msg}")
            if not ok:
                all_ok = False

        return str(SkillResult(
            success=all_ok,
            message="\n".join(results),
            data={"homed": arms_to_home},
        ))

    except Exception as exc:
        return str(SkillResult(
            success=False,
            message=f"Home failed: {exc}",
        ))