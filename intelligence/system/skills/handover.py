"""
skills/handover.py
Skill: handover_brick
→ AR4 moves to handover_pose then sends:
    ExecuteTask(task_type="INTERMEDIATE_GIVE",  target_pose=handover_pose)
  ABB receives and sends:
    ExecuteTask(task_type="INTERMEDIATE_TAKE",  target_pose=abb_grasp_pose)
  then AR4 releases:
    ExecuteTask(task_type="RELEASE")
  (supervisor_node.py lines 1242, 1322, 1328)

Handover pose (from supervisor_node.py lines 1098-1099):
  x=0.56, y=0.10, z=handover_meeting_height + ar4_z_offset
  handover_meeting_height = 0.127m  (line 117)
  ar4_z_offset = 0.043m             (line 118)
  Total Z = 0.127 + 0.043 = 0.170m

Only valid for Z-bricks (supervisor enforces this).
"""

from langchain_core.tools import tool
from skills._bridge import get_bridge, SkillResult
from workspace_constraints import (
    HANDOVER_POSE_X, HANDOVER_POSE_Y,
    HANDOVER_MEET_Z, AR4_Z_OFFSET,
)


@tool
def handover_brick(brick_id: int) -> str:
    """
    Execute a Z-brick handover from AR4 to ABB.

    This is a coordinated dual-arm operation:
      1. AR4 moves to handover pose (x=0.56, y=0.10, z=0.170m)
      2. AR4 sends INTERMEDIATE_GIVE — holds brick at meeting point
      3. ABB approaches and sends INTERMEDIATE_TAKE — grabs the brick
      4. AR4 sends RELEASE — opens gripper and retreats

    The handover pose is fixed in the supervisor at:
      x=0.56m, y=0.10m  (supervisor_node.py lines 1098-1099)
      z=0.170m (HANDOVER_MEET_Z + AR4_Z_OFFSET = 0.127 + 0.043)

    Args:
        brick_id: ID of the Z-brick to transfer (must be in AR4's gripper).

    Returns:
        Success/failure with handover sequence details.

    IMPORTANT: Only use this skill for Z-bricks.
    The supervisor automatically triggers MTC collision avoidance
    when arms are within 0.30m of each other during handover.
    """
    bridge = get_bridge()

    handover_z = HANDOVER_MEET_Z + AR4_Z_OFFSET   # 0.127 + 0.043 = 0.170m

    try:
        # Verify brick exists
        bricks = bridge.get_detected_bricks()
        target = next((b for b in bricks if b.get("id") == brick_id), None)

        # In mock mode we don't enforce Z-brick type (for testing flexibility)
        # In live mode supervisor enforces it
        if target and target.get("type") not in ("Z", None) and not bridge.is_mock:
            return str(SkillResult(
                success=False,
                message=f"Handover is only for Z-bricks. "
                        f"Brick id={brick_id} is a {target['type']}-brick.",
            ))

        if bridge.is_mock:
            return str(SkillResult(
                success=True,
                message=(
                    f"Handover complete: AR4 → ABB for brick id={brick_id}\n"
                    f"  GIVE pose: ({HANDOVER_POSE_X}, {HANDOVER_POSE_Y}, "
                    f"{handover_z:.3f})\n"
                    f"  Sequence: PICK → INTERMEDIATE_GIVE → "
                    f"INTERMEDIATE_TAKE → RELEASE"
                ),
                data={
                    "brick_id":   brick_id,
                    "from_arm":   "AR4",
                    "to_arm":     "ABB",
                    "handover_pose": {
                        "x": HANDOVER_POSE_X,
                        "y": HANDOVER_POSE_Y,
                        "z": handover_z,
                    },
                    "sequence": [
                        "AR4: INTERMEDIATE_GIVE",
                        "ABB: INTERMEDIATE_TAKE",
                        "AR4: RELEASE",
                    ],
                },
            ))

        # Live mode
        ok, msg = bridge.push_plan([{
            "id":        brick_id,
            "task_type": "HANDOVER",
            "from_arm":  "AR4",
            "to_arm":    "ABB",
        }])
        return str(SkillResult(success=ok, message=msg,
                               data={"brick_id": brick_id}))

    except Exception as exc:
        return str(SkillResult(
            success=False,
            message=f"Handover failed: {exc}",
        ))