"""
skills/place.py
Skill: place_brick
→ sends ExecuteTask(task_type="PLACE", target_pose=place_pose)
  to ar4_control or abb_control action server
  (supervisor_node.py lines 706, 803)

Place heights from supervisor_node.py (real measured values):
  AR4 sim place Z = 0.21 m
  ABB sim place Z = 0.22 m
"""

from langchain_core.tools import tool
from skills._bridge import get_bridge, SkillResult
from workspace_constraints import TABLE_Z_PLACE_AR4, TABLE_Z_PLACE_ABB


@tool
def place_brick(
    brick_id:   int,
    arm:        str,
    x:          float,
    y:          float,
    rotation:   int   = 0,
    layer:      int   = 0,
) -> str:
    """
    Command an arm to place the currently held brick at a target position.

    The arm must already be holding a brick (call pick_brick first).
    Z height is automatically set based on arm and layer:
      base layer (0) : AR4=0.21m, ABB=0.22m  (from supervisor_node.py)
      each layer adds: 0.040m  (from brick SDF wall height)

    Args:
        brick_id:  ID of the brick being placed.
        arm:       "AR4" or "ABB" — which arm places the brick.
        x:         Target X in robot frame (metres).
        y:         Target Y in robot frame (metres).
        rotation:  Placement rotation in degrees (0, 90, 180, 270).
        layer:     Assembly layer (0=base, 1=on top of layer 0, etc.).

    Returns:
        Success/failure with the place pose and final position.

    The supervisor's PLACE state machine handles approach, release,
    and retreat automatically after receiving the goal.
    """
    bridge = get_bridge()

    arm = arm.upper()
    if arm not in ("AR4", "ABB"):
        return str(SkillResult(
            success=False,
            message=f"Invalid arm '{arm}'. Must be 'AR4' or 'ABB'.",
        ))

    # Compute Z from layer + arm
    from workspace_constraints import BRICK_LAYER_HEIGHT
    base_z = TABLE_Z_PLACE_AR4 if arm == "AR4" else TABLE_Z_PLACE_ABB
    z = base_z + layer * BRICK_LAYER_HEIGHT

    try:
        if bridge.is_mock:
            return str(SkillResult(
                success=True,
                message=(
                    f"{arm} placed brick id={brick_id} at "
                    f"({x:.3f}, {y:.3f}, {z:.3f})  "
                    f"rotation={rotation}°  layer={layer}"
                ),
                data={
                    "arm":      arm,
                    "brick_id": brick_id,
                    "pose":     {"x": x, "y": y, "z": z, "rotation": rotation},
                    "layer":    layer,
                    "action":   "PLACE",
                },
            ))

        # Live mode
        ok, msg = bridge.push_plan([{
            "id":        brick_id,
            "task_type": "PLACE",
            "arm":       arm,
            "x": x, "y": y, "z": z,
            "rotation":  rotation,
            "layer":     layer,
        }])
        return str(SkillResult(success=ok, message=msg,
                               data={"arm": arm, "brick_id": brick_id, "z": z}))

    except Exception as exc:
        return str(SkillResult(
            success=False,
            message=f"Place failed: {exc}",
        ))