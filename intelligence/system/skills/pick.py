"""
skills/pick.py
Skill: pick_brick
→ sends ExecuteTask(task_type="PICK", target_pose=grasp_pose)
  to ar4_control or abb_control action server
  (supervisor_node.py lines 690, 787)

Also opens the gripper first via:
  ar4_controller/set_gripper  (SetBool, data=True → open)
  abb_controller/set_gripper  (SetBool, data=True → open)
"""

from langchain_core.tools import tool
from skills._bridge import get_bridge, SkillResult


@tool
def pick_brick(
    brick_id:  int,
    arm:       str,
    x:         float,
    y:         float,
    z:         float,
    yaw_deg:   float = 0.0,
) -> str:
    """
    Command an arm to pick up a specific brick from the table.

    This skill:
      1. Opens the gripper (ar4_controller/set_gripper or abb_controller/set_gripper)
      2. Moves the arm to the grasp pose
      3. Closes the gripper to grasp the brick
      (Internally sends ExecuteTask(task_type="PICK") to the action server)

    Args:
        brick_id: ID of the brick to pick (from detect_bricks).
        arm:      "AR4" or "ABB" — which arm performs the pick.
        x:        Target X position in robot frame (metres).
        y:        Target Y position in robot frame (metres).
        z:        Target Z position in robot frame (metres).
                  Use 0.21 for AR4, 0.22 for ABB (from supervisor_node.py).
        yaw_deg:  Gripper yaw angle in degrees (from get_grasp_point).

    Returns:
        Success/failure message with the achieved pose.

    Always call get_grasp_point before pick_brick to get the correct pose.
    """
    bridge = get_bridge()

    # Validate arm
    arm = arm.upper()
    if arm not in ("AR4", "ABB"):
        return str(SkillResult(
            success=False,
            message=f"Invalid arm '{arm}'. Must be 'AR4' or 'ABB'.",
        ))

    try:
        if bridge.is_mock:
            # Simulate pick — update mock state
            status = bridge.get_status()
            stage_key = "ar4_stage" if arm == "AR4" else "abb_stage"
            return str(SkillResult(
                success=True,
                message=(
                    f"{arm} picked {_brick_type_from_id(bridge, brick_id)}"
                    f"-brick id={brick_id} at "
                    f"({x:.3f}, {y:.3f}, {z:.3f}) yaw={yaw_deg:.1f}°"
                ),
                data={
                    "arm":      arm,
                    "brick_id": brick_id,
                    "pose":     {"x": x, "y": y, "z": z, "yaw_deg": yaw_deg},
                    "action":   "PICK",
                },
            ))

        # Live mode — push pick command via bridge
        # The supervisor handles the full PICK state machine;
        # we push the goal as part of the assembly plan
        ok, msg = bridge.push_plan([{
            "id":         brick_id,
            "task_type":  "PICK",
            "arm":        arm,
            "x": x, "y": y, "z": z,
            "yaw_deg":    yaw_deg,
        }])
        return str(SkillResult(success=ok, message=msg,
                               data={"arm": arm, "brick_id": brick_id}))

    except Exception as exc:
        return str(SkillResult(
            success=False,
            message=f"Pick failed: {exc}",
        ))


def _brick_type_from_id(bridge, brick_id: int) -> str:
    """Helper: look up brick type from the detected bricks list."""
    bricks = bridge.get_detected_bricks()
    b = next((b for b in bricks if b.get("id") == brick_id), None)
    return b["type"] if b else "?"