"""
workspace_constraints.py
========================
Physical workspace rules for the dual-arm assembly system.
All numbers are derived from the actual codebase:
  - supervisor_node.py  : coordinate transforms, handover distance (0.3 m)
  - advanced_detector_node.py : zone split at 42% of frame height
  - master_launch.py    : URDF/SRDF, arm model specs

These constants are used by:
  - validator_chain.py  : LLM-grounded validation checks
  - layout_renderer.py  : 2D preview (Task 5)
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math


# ─────────────────────────────────────────────────────────────────────────────
# Table coordinate system
# Origin = table centre. Units = metres.
#
# Derived from live tf2_echo measurements (April 2026):
#   world → abb_table  : [0.333, 0.000, 1.100]
#   world → ar4_base   : [1.443, 0.000, 1.107]  (rotated -90° around Z)
#   world → camera     : [1.002, 0.010, 1.904]
#   ar4_base → ar4_ee  : [-0.007, -0.328, 0.474]  (home pose)
#   base_link → tool0  : [0.374, -0.000, 0.629]   (home pose)
#
# Plan frame convention (matches supervisor_node.py):
#   World X (arm-to-arm depth, 1.110 m separation) → Plan Y axis
#   World Y (lateral width)                         → Plan X axis
#   AR4 side → positive plan Y  (AR4 world-X = 1.443, larger)
#   ABB side → negative plan Y  (ABB world-X = 0.333, smaller)
# ─────────────────────────────────────────────────────────────────────────────

TABLE_X_MIN = -0.20    # lateral left edge  (metres)
TABLE_X_MAX =  0.20    # lateral right edge
TABLE_Y_MIN = -0.25    # ABB side boundary  (derived: arm reach 0.580 - base offset 0.555 + margin)
TABLE_Y_MAX =  0.25    # AR4 side boundary
TABLE_Z_PLACE = 0.21   # brick place height in robot frame (from supervisor sim transforms)
TABLE_MAX_HEIGHT = 0.20   # 5 layers × 0.040m/layer (from SDF wall height)

# Place heights from supervisor_node.py (exact values, not estimated)
TABLE_Z_PLACE_AR4 = 0.21   # metres — supervisor_node.py line 357
TABLE_Z_PLACE_ABB = 0.22   # metres — supervisor_node.py line 364
STATIC_Z_HEIGHT   = 0.712  # metres — grasping_node.py camera back-projection Z

# Zone split: camera splits table at its centre in world X → plan Y = 0
# Bricks with plan Y > 0 → AR4 side
# Bricks with plan Y < 0 → ABB side
ZONE_SPLIT_Y = 0.00

# Grid spacing = stud-to-stud pitch measured from SDF collision geometry
# L-brick stud spacing: (0.015,0.045)→(0.045,0.015) = 0.030 m confirmed
# Z-brick stud spacing: all consecutive pairs = 0.030 m confirmed
GRID_STEP = 0.030   # metres  (was assumed 0.040 — CORRECTED from SDF)

# Camera height above table surface (1.904 - 1.100 = 0.804 m)
CAMERA_HEIGHT = 0.804   # metres

# Arm separation in world X (1.443 - 0.333 = 1.110 m)
ARM_SEPARATION = 1.110   # metres

# ─────────────────────────────────────────────────────────────────────────────
# Arm reach envelopes — from TF data + kinematic specs
# ─────────────────────────────────────────────────────────────────────────────

# AR4 Mk3: 580mm kinematic reach. Home EE 3D distance = 0.576m ✓
AR4_REACH_RADIUS = 0.580   # metres

# ABB IRB 120: 580mm kinematic reach. Home EE 3D = 0.732m (arm elevated at home)
ABB_REACH_RADIUS = 0.580   # metres

# Arm base positions in plan frame (each is ARM_SEPARATION/2 = 0.555 m from centre)
# AR4 is at world X=1.443, which maps to plan Y = +(1.443-0.888) = +0.555
# ABB is at world X=0.333, which maps to plan Y = -(0.888-0.333) = -0.555
AR4_BASE_X =  0.00
AR4_BASE_Y = +0.555   # AR4 base behind AR4 side of table (from tf2_echo)

ABB_BASE_X =  0.00
ABB_BASE_Y = -0.555   # ABB base behind ABB side of table (from tf2_echo)

# ─────────────────────────────────────────────────────────────────────────────
# Collision / proximity
# ─────────────────────────────────────────────────────────────────────────────

# From supervisor_node.py line 47: handover_trigger_distance = 0.3 m
HANDOVER_TRIGGER_DISTANCE = 0.30   # metres  ← confirmed from supervisor_node.py

# From supervisor_node.py lines 1098-1099: handover_pose.position
HANDOVER_POSE_X = 0.56   # metres — actual handover X coordinate
HANDOVER_POSE_Y = 0.10   # metres — actual handover Y coordinate
HANDOVER_MEET_Z = 0.127  # metres — supervisor_node.py line 117

# Z offsets per arm (supervisor_node.py lines 118-119)
AR4_Z_OFFSET = 0.043   # metres
ABB_Z_OFFSET = 0.136   # metres

MIN_EE_SEPARATION = 0.15   # assumed safety margin (half of handover trigger)

# From SDF wall_left/wall_right thickness = 0.004 m
# Two adjacent bricks have 0.004m walls touching — minimum physical separation
MIN_BRICK_SEPARATION = 0.004   # metres  (was assumed 0.035 — CORRECTED from SDF wall thickness)

# ─────────────────────────────────────────────────────────────────────────────
# Brick physical dimensions — derived from SDF collision geometry
#
# Source: i_brick_model.sdf, l_brick_model.sdf, t_brick_model.sdf, z_brick_model.sdf
# Grid unit (GRID_STEP) = 0.030 m confirmed from stud-to-stud spacing
#
# Brick heights (from SDF Z spans):
#   BRICK_LAYER_HEIGHT = 0.040 m  (wall height — effective stacking pitch)
#   BRICK_TOTAL_HEIGHT = 0.050 m  (wall 0.040 + stud 0.010)
#   WALL_THICKNESS     = 0.004 m
#
# Bounding boxes (exact from SDF):
#   I-brick: 0.030 × 0.058 m  (1×2 grid cells)
#   L-brick: 0.060 × 0.060 m  (2×2 bounding box, 3 cells occupied)
#   T-brick: 0.060 × 0.090 m  (2×3 bounding box, 4 cells occupied)
#   Z-brick: 0.120 × 0.090 m  (4×3 bounding box, 6 cells occupied)
# ─────────────────────────────────────────────────────────────────────────────

BRICK_LAYER_HEIGHT = 0.040   # metres — from SDF wall collision Z span
BRICK_TOTAL_HEIGHT = 0.050   # metres — wall + stud
WALL_THICKNESS     = 0.004   # metres — from SDF wall box sizes

# Each brick type → list of (dx, dy) cell offsets from anchor (bottom-left).
# Rotation 90/180/270 computed by rotate_cells().
# All offsets in multiples of GRID_STEP (0.030 m).
BRICK_CELLS: dict[str, list[tuple[float, float]]] = {
    # I-brick: 1 col × 2 rows (straight bar)
    # SDF: X span 0.030m, Y span 0.058m ≈ 2 × 0.030m
    "I": [(0.000, 0.000),
          (0.000, 0.030)],

    # L-brick: 3 cells in L pattern (2×2 bounding box, top-right cell missing)
    # SDF roofs at: (col=0,row=0), (col=0,row=1), (col=1,row=0)
    # Studs at: (0.015, 0.045) and (0.045, 0.015) confirmed
    "L": [(0.000, 0.000),
          (0.000, 0.030),
          (0.030, 0.000)],

    # T-brick: 4 cells — bar (col=0, rows 0-2) + leg (col=1, row=1)
    # SDF studs at: (0.015,0.015), (0.015,0.075), (0.045,0.045)
    # Bar Y span = 0.090m = 3 × 0.030m, Leg X span = 0.030m
    "T": [(0.000, 0.000),
          (0.000, 0.030),
          (0.000, 0.060),
          (0.030, 0.030)],

    # Z-brick: 6 cells in Z/S pattern (4×3 bounding box, 0.120 × 0.090m)
    # SDF studs (normalised to origin):
    #   col=0: rows 1,2  |  col=1,2,3: row 1  |  col=3: rows 0,1
    "Z": [(0.000, 0.060),
          (0.000, 0.030),
          (0.030, 0.030),
          (0.060, 0.030),
          (0.090, 0.030),
          (0.090, 0.000)],
}


def rotate_cells(
    cells: list[tuple[float, float]], degrees: int
) -> list[tuple[float, float]]:
    """Rotate cell offsets by 0/90/180/270 degrees around origin."""
    rad = math.radians(degrees)
    cos_a, sin_a = round(math.cos(rad)), round(math.sin(rad))
    return [(cos_a * x - sin_a * y, sin_a * x + cos_a * y) for x, y in cells]


def get_occupied_cells(
    brick_type: str, x: float, y: float, rotation: int
) -> list[tuple[float, float]]:
    """
    Return the list of (x, y) world positions occupied by a brick.
    rotation: 0, 90, 180, or 270 degrees.
    """
    base_cells = BRICK_CELLS.get(brick_type, [(0.0, 0.0)])
    rotated    = rotate_cells(base_cells, rotation)
    return [(x + dx, y + dy) for dx, dy in rotated]


# ─────────────────────────────────────────────────────────────────────────────
# Validation result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    passed:  bool
    code:    str    # e.g. "INVENTORY_OK", "WORKSPACE_EXCEEDED"
    message: str
    detail:  str = ""


@dataclass
class ValidationReport:
    status:       str               # "accept" | "reject" | "suggest"
    checks:       list[CheckResult] = field(default_factory=list)
    reason:       str = ""          # human-readable summary if rejected
    suggestion:   str = ""          # what the LLM should propose instead
    safe_to_run:  bool = False

    def add(self, check: CheckResult):
        self.checks.append(check)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def passed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.passed]


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python pre-checks  (fast, no LLM needed)
# The LLM validator calls these first and uses the results as context.
# ─────────────────────────────────────────────────────────────────────────────

def check_inventory(
    required_bricks: list[str],
    inventory: dict[str, int],
) -> CheckResult:
    """Do we have enough of each brick type?"""
    from collections import Counter
    needed   = Counter(required_bricks)
    shortages = []
    for btype, count in needed.items():
        have = inventory.get(btype, 0)
        if have < count:
            shortages.append(f"{count}×{btype} needed, {have} available")

    if shortages:
        return CheckResult(
            passed  = False,
            code    = "INVENTORY_INSUFFICIENT",
            message = "Not enough bricks in inventory",
            detail  = "; ".join(shortages),
        )
    return CheckResult(passed=True, code="INVENTORY_OK",
                       message="Inventory sufficient")


def check_workspace_bounds(arrangement: list[dict]) -> CheckResult:
    """Are all brick placements within the table boundary?"""
    violations = []
    for brick in arrangement:
        cells = get_occupied_cells(
            brick.get("brick", "I"),
            brick.get("x", 0.0),
            brick.get("y", 0.0),
            brick.get("rotation", 0),
        )
        for cx, cy in cells:
            if not (TABLE_X_MIN <= cx <= TABLE_X_MAX and
                    TABLE_Y_MIN <= cy <= TABLE_Y_MAX):
                violations.append(
                    f"brick id={brick.get('id')} cell ({cx:.3f},{cy:.3f}) "
                    f"out of bounds"
                )

    if violations:
        return CheckResult(
            passed  = False,
            code    = "WORKSPACE_EXCEEDED",
            message = "One or more bricks fall outside the table boundary",
            detail  = "; ".join(violations[:3]),  # cap at 3 for readability
        )
    return CheckResult(passed=True, code="WORKSPACE_OK",
                       message="All bricks within table boundary")



def check_collision(arrangement: list[dict]) -> CheckResult:
    """Do any two bricks overlap on the same layer?"""
    from collections import defaultdict

    # Group occupied cells by layer
    layer_cells: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
    for brick in arrangement:
        layer = brick.get("layer", 0)
        cells = get_occupied_cells(
            brick.get("brick", "I"),
            brick.get("x", 0.0),
            brick.get("y", 0.0),
            brick.get("rotation", 0),
        )
        for cx, cy in cells:
            # Round to 3dp to avoid float comparison issues
            layer_cells[layer].append((round(cx, 3), round(cy, 3),
                                       brick.get("id", -1)))

    conflicts = []
    for layer, cell_list in layer_cells.items():
        seen: dict[tuple, int] = {}
        for cx, cy, bid in cell_list:
            key = (cx, cy)
            if key in seen:
                conflicts.append(
                    f"layer {layer}: brick {bid} overlaps brick "
                    f"{seen[key]} at ({cx},{cy})"
                )
            else:
                seen[key] = bid

    if conflicts:
        return CheckResult(
            passed  = False,
            code    = "COLLISION_DETECTED",
            message = "Brick positions overlap — collision would occur",
            detail  = "; ".join(conflicts[:3]),
        )
    return CheckResult(passed=True, code="COLLISION_FREE",
                       message="No brick overlaps detected")


def check_height(arrangement: list[dict]) -> CheckResult:
    """Does the stack stay within maximum height?"""
    max_layer = max((b.get("layer", 0) for b in arrangement), default=0)
    # Each layer adds ~0.038 m (brick height) + 0.002 m mortar
    estimated_height = (max_layer + 1) * 0.04

    if estimated_height > TABLE_MAX_HEIGHT:
        return CheckResult(
            passed  = False,
            code    = "HEIGHT_EXCEEDED",
            message = f"Estimated height {estimated_height:.2f}m exceeds "
                      f"maximum {TABLE_MAX_HEIGHT}m",
            detail  = f"Max layer={max_layer}, ~{estimated_height:.2f}m tall",
        )
    return CheckResult(passed=True, code="HEIGHT_OK",
                       message=f"Height OK (~{estimated_height:.2f}m)")

def check_components_match(arrangement: list[dict],
                           expected_components: int) -> CheckResult:
    """Phase A — actual component count must equal what the agent declared."""
    if not arrangement:
        return CheckResult(passed=True, code="COMPONENTS_OK",
                           message="Empty plan — trivially OK")

    all_cells = set()
    for brick in arrangement:
        cells = get_occupied_cells(
            brick.get("brick", "I"),
            brick.get("x", 0.0),
            brick.get("y", 0.0),
            brick.get("rotation", 0),
        )
        for cx, cy in cells:
            all_cells.add((round(cx / GRID_STEP), round(cy / GRID_STEP)))

    actual = _count_connected_components(all_cells)

    if actual != expected_components:
        return CheckResult(
            passed=False,
            code="COMPONENTS_MISMATCH",
            message=f"Agent declared {expected_components} component(s), "
                    f"design has {actual}",
            detail=("Either the design has unintended gaps, or the agent "
                    "miscounted. Force a redesign."),
        )
    return CheckResult(
        passed=True, code="COMPONENTS_OK",
        message=f"Component count matches declaration ({actual})",
    )


def check_brick_count_match(arrangement: list[dict],
                            expected_brick_count: int,
                            tolerance: int = 1) -> CheckResult:
    """Phase A — actual brick count must be close to declared (±tolerance)."""
    actual = len(arrangement)

    if expected_brick_count <= 0:
        return CheckResult(
            passed=True, code="BRICK_COUNT_SKIPPED",
            message="No declared brick count — skipping check",
        )

    diff = abs(actual - expected_brick_count)
    if diff > tolerance:
        return CheckResult(
            passed=False,
            code="BRICK_COUNT_MISMATCH",
            message=f"Agent declared {expected_brick_count} bricks, "
                    f"design uses {actual}",
            detail=f"Off by {diff} (tolerance {tolerance}).",
        )
    return CheckResult(
        passed=True, code="BRICK_COUNT_OK",
        message=f"Brick count matches declaration ({actual})",
    )


def _count_connected_components(cells: set) -> int:
    """4-connectivity flood fill — returns number of disjoint regions."""
    if not cells:
        return 0
    remaining = set(cells)
    count = 0
    while remaining:
        seed = next(iter(remaining))
        comp = {seed}
        stack = [seed]
        while stack:
            c, r = stack.pop()
            for n in [(c+1, r), (c-1, r), (c, r+1), (c, r-1)]:
                if n in remaining and n not in comp:
                    comp.add(n)
                    stack.append(n)
        remaining -= comp
        count += 1
    return count


def run_all_checks(
    arrangement:     list[dict],
    required_bricks: list[str],
    inventory:       dict[str, int],
) -> list[CheckResult]:
    """Run all pure-Python checks and return results."""
    return [
        check_inventory(required_bricks, inventory),
        check_workspace_bounds(arrangement),
        check_arm_reachability(arrangement),
        check_collision(arrangement),
        check_height(arrangement),
    ]


def run_all_checks(
    arrangement:     list,
    required_bricks: list,
    inventory:       dict,
) -> list:
    """
    Run all validation checks EXCEPT arm reachability.
    Reachability is handled by the robot nodes (MoveIt/MTC).
    Checks: inventory, workspace bounds, collision, height.
    """
    return [
        check_inventory(required_bricks, inventory),
        check_workspace_bounds(arrangement),
        check_collision(arrangement),
        check_height(arrangement),
    ]