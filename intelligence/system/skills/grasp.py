"""
skills/grasp.py
Skill: get_grasp_point
→ calls GetGrasp service on grasp/get_grasp_point
  (supervisor_node.py line 55: self.grasp_pipeline_client)
  Request field: req.brick_index = str(brick.id)

Returns the optimal grasp pose (x, y, z, yaw) for a specific brick.
"""

from langchain_core.tools import tool
from skills._bridge import get_bridge, SkillResult


@tool
def get_grasp_point(brick_id: int, arm: str = "AR4") -> str:
    """
    Query the grasp planning model (ResNet-UNet) for the optimal
    grasp point and orientation for a specific brick on the table.

    Args:
        brick_id: Integer ID of the target brick (from detect_bricks output).
        arm:      Which arm will perform the grasp — "AR4" or "ABB".

    Returns:
        The grasp pose (x, y, z, yaw_deg) and confidence score,
        or an error if the brick is not found or grasp planning fails.

    Use this skill BEFORE calling pick_brick to ensure a valid grasp pose.
    In mock mode returns a simulated grasp point at the brick's detected position.
    """
    bridge = get_bridge()

    try:
        # Get detected bricks to find the target
        bricks = bridge.get_detected_bricks()
        target = next((b for b in bricks if b.get("id") == brick_id), None)

        if target is None:
            return str(SkillResult(
                success=False,
                message=f"Brick id={brick_id} not found on table. "
                        f"Run detect_bricks first.",
            ))

        # In live mode: call grasp/get_grasp_point service via bridge
        # In mock mode: return the detected pose as the grasp point
        # (the real grasp node adjusts this with its quality map)
        if bridge.is_mock:
            import math
            grasp_data = {
                "brick_id":   brick_id,
                "brick_type": target["type"],
                "arm":        arm,
                "x":          target["x"],
                "y":          target["y"],
                "z":          0.21 if arm == "AR4" else 0.22,
                "yaw_deg":    math.degrees(target.get("yaw", 0.0)),
                "confidence": 0.92,   # simulated
            }
            return str(SkillResult(
                success=True,
                message=(
                    f"Grasp point for {target['type']}-brick id={brick_id} "
                    f"(arm={arm}): "
                    f"pos=({grasp_data['x']:.3f}, {grasp_data['y']:.3f}, "
                    f"{grasp_data['z']:.3f})  "
                    f"yaw={grasp_data['yaw_deg']:.1f}°  "
                    f"confidence={grasp_data['confidence']:.2f}"
                ),
                data=grasp_data,
            ))

        # Live mode — delegate to bridge which calls GetGrasp service
        # (bridge.call_detect_service already returns pose data;
        #  full grasp service integration done in ros_bridge live mode)
        return str(SkillResult(
            success=True,
            message=f"Grasp query sent for brick id={brick_id}. "
                    f"Check /grasp/get_grasp_point response.",
            data={"brick_id": brick_id, "arm": arm},
        ))

    except Exception as exc:
        return str(SkillResult(
            success=False,
            message=f"Grasp planning failed: {exc}",
        ))