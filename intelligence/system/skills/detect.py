"""
skills/detect.py
Skill: detect_bricks
→ calls DetectBricks service on detect_bricks topic
  (supervisor_node.py line 54: self.camera_client)

Returns a JSON summary of detected bricks, grouped by type and side.
"""

from langchain_core.tools import tool
from skills._bridge import get_bridge, SkillResult


@tool
def detect_bricks(brick_type: str = "") -> str:
    """
    Trigger the YOLO detection pipeline and return a list of bricks
    currently visible on the assembly table.

    Args:
        brick_type: Optional filter — "I", "L", "T", or "Z".
                    Leave empty to detect all brick types.

    Returns:
        A text summary of detected bricks with their positions and sides,
        or an error message if detection fails.

    Use this skill to:
      - Check what bricks are on the table before planning
      - Refresh inventory after a pick operation
      - Verify a brick was placed correctly
    """
    bridge = get_bridge()

    try:
        bricks = bridge.call_detect_service()

        if not bricks:
            return str(SkillResult(
                success=False,
                message="No bricks detected on the table.",
            ))

        # Filter by type if requested
        if brick_type:
            bricks = [b for b in bricks
                      if b.get("type", "").upper() == brick_type.upper()]
            if not bricks:
                return str(SkillResult(
                    success=False,
                    message=f"No {brick_type}-bricks detected.",
                ))

        # Build summary
        lines = [f"Detected {len(bricks)} brick(s):"]
        for b in bricks:
            lines.append(
                f"  id={b['id']}  type={b['type']}  side={b['side']}"
                f"  pos=({b['x']:.3f}, {b['y']:.3f})  yaw={b['yaw']:.2f}rad"
            )

        # Inventory counts
        from collections import Counter
        counts = Counter(b["type"] for b in bricks)
        lines.append(f"Inventory: " + ", ".join(
            f"{v}×{k}" for k, v in sorted(counts.items())
        ))

        return str(SkillResult(
            success=True,
            message="\n".join(lines),
            data={"bricks": bricks, "counts": dict(counts)},
        ))

    except Exception as exc:
        return str(SkillResult(
            success=False,
            message=f"Detection failed: {exc}",
        ))