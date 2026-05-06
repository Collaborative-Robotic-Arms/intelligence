"""
brick_solver.py
===============
Tiles a set of grid cells using the available LEGO bricks.

NO hardcoded shape patterns. The LLM designs the shape (any shape),
this solver attempts to tile it. On failure, the solver reports which
cells could not be tiled so the LLM can redesign.

Pieces:
  I-brick: 2 cells in a line (rotations: vertical, horizontal)
  L-brick: 3 cells in L-shape (4 rotations)
  T-brick: 4 cells in T-shape (4 rotations)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

from workspace_constraints import BRICK_CELLS, GRID_STEP


# ─────────────────────────────────────────────────────────────────────────────
# Solver result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    success:        bool
    placements:     list = field(default_factory=list)   # brick dicts on success
    untiled_cells:  list = field(default_factory=list)   # cells couldn't tile
    reason:         str  = ""                             # human explanation


# ─────────────────────────────────────────────────────────────────────────────
# Brick footprint helpers (grid units, not metres)
# ─────────────────────────────────────────────────────────────────────────────

def _rotate_int(col: int, row: int, deg: int) -> tuple[int, int]:
    """Rotate (col, row) by 0/90/180/270 degrees around origin, exact integer."""
    rad = math.radians(deg)
    c, s = round(math.cos(rad)), round(math.sin(rad))
    return c * col - s * row, s * col + c * row


def _brick_cells_grid(brick_type: str, rotation: int) -> list[tuple[int, int]]:
    """Get brick cell offsets in grid units at given rotation."""
    base = BRICK_CELLS.get(brick_type, [(0.0, 0.0)])
    rotated = []
    for x_m, y_m in base:
        col = round(x_m / GRID_STEP)
        row = round(y_m / GRID_STEP)
        rotated.append(_rotate_int(col, row, rotation))
    return rotated


def _can_place(
    brick_type: str,
    rotation:   int,
    anchor_col: int,
    anchor_row: int,
    target:     set[tuple[int, int]],
) -> Optional[set[tuple[int, int]]]:
    """Check if a brick fits entirely within target cells. Returns occupied set or None."""
    occupied = set()
    for dc, dr in _brick_cells_grid(brick_type, rotation):
        cell = (anchor_col + dc, anchor_row + dr)
        if cell not in target:
            return None
        occupied.add(cell)
    return occupied


# ─────────────────────────────────────────────────────────────────────────────
# Cell input normalisation (LLM might send malformed lists)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_cells(cells_input: list) -> set[tuple[int, int]]:
    """Convert any cell-list-like structure into a clean set of (col, row) tuples."""
    result = set()
    for c in cells_input or []:
        try:
            if isinstance(c, (list, tuple)) and len(c) == 2:
                result.add((int(c[0]), int(c[1])))
        except (ValueError, TypeError):
            continue
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks — quick filters before expensive solving
# ─────────────────────────────────────────────────────────────────────────────

def _find_isolated_cells(target: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """
    A cell is 'isolated' if it has no orthogonal neighbour in the target set.
    Such cells can never be covered (smallest brick = 2 cells).
    """
    isolated = set()
    for col, row in target:
        neighbours = {(col+1, row), (col-1, row), (col, row+1), (col, row-1)}
        if not (neighbours & target):
            isolated.add((col, row))
    return isolated


def _connected_components(target: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Return list of connected cell groups (4-connectivity)."""
    remaining = set(target)
    components = []
    while remaining:
        # BFS from any cell
        seed = next(iter(remaining))
        comp = {seed}
        frontier = [seed]
        while frontier:
            col, row = frontier.pop()
            for n in [(col+1, row), (col-1, row), (col, row+1), (col, row-1)]:
                if n in remaining and n not in comp:
                    comp.add(n)
                    frontier.append(n)
        components.append(comp)
        remaining -= comp
    return components


# ─────────────────────────────────────────────────────────────────────────────
# Solver — backtracking with heuristics
# ─────────────────────────────────────────────────────────────────────────────

# Try larger bricks first → fewer total bricks, fewer recursion levels
BRICK_ORDER = ["T", "L", "I"]
ROTATIONS   = [0, 90, 180, 270]


def _solve_recursive(
    remaining: set[tuple[int, int]],
    available: dict[str, int],
    placements: list[dict],
    max_depth: int = 50,
) -> bool:
    """Recursive backtracking. Mutates placements; returns True on success."""
    if not remaining:
        return True
    if len(placements) >= max_depth:
        return False  # safety cap

    # Most-constrained-first: pick the cell with fewest target neighbours
    def constraint_key(c):
        col, row = c
        n_count = sum(1 for n in [(col+1, row), (col-1, row), (col, row+1), (col, row-1)]
                      if n in remaining)
        return (n_count, -row, col)   # fewest neighbours, then top-left

    anchor = min(remaining, key=constraint_key)

    for btype in BRICK_ORDER:
        if available.get(btype, 0) <= 0:
            continue

        for rot in ROTATIONS:
            offsets = _brick_cells_grid(btype, rot)
            tried = set()

            # The anchor must be one of the brick's cells
            for dc, dr in offsets:
                anchor_col = anchor[0] - dc
                anchor_row = anchor[1] - dr
                if (anchor_col, anchor_row) in tried:
                    continue
                tried.add((anchor_col, anchor_row))

                occupied = _can_place(btype, rot, anchor_col, anchor_row, remaining)
                if occupied is None:
                    continue

                # Recurse
                x_m = anchor_col * GRID_STEP
                y_m = anchor_row * GRID_STEP
                placements.append({
                    "brick":    btype,
                    "x":        round(x_m, 3),
                    "y":        round(y_m, 3),
                    "rotation": rot,
                })
                available[btype] -= 1

                if _solve_recursive(remaining - occupied, available, placements):
                    return True

                # Backtrack
                placements.pop()
                available[btype] += 1

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def solve_shape(
    cells_input: list,
    inventory:   dict[str, int],
) -> SolverResult:
    """
    Attempt to tile a set of grid cells with the available bricks.

    Returns SolverResult:
      - success=True  → placements ready to use
      - success=False → reason explains why, untiled_cells lists problem cells

    Failure reasons:
      "EMPTY_SHAPE"     — no cells provided
      "ISOLATED_CELLS"  — cells with no neighbours (can never be tiled)
      "INVENTORY_EMPTY" — no bricks at all
      "NO_TILING"       — cells form valid regions but no tiling found
                          with these brick counts
    """
    target = _normalise_cells(cells_input)
    if not target:
        return SolverResult(False, reason="EMPTY_SHAPE")

    # Pre-flight 1: isolated cells
    isolated = _find_isolated_cells(target)
    if isolated:
        return SolverResult(
            False,
            untiled_cells = sorted(isolated),
            reason        = "ISOLATED_CELLS",
        )

    # Pre-flight 2: any inventory at all?
    total = sum(inventory.get(b, 0) for b in ["I", "L", "T"])
    if total == 0:
        return SolverResult(
            False,
            reason = "INVENTORY_EMPTY",
        )

    # Solve each connected component independently
    all_placements: list[dict] = []
    available = dict(inventory)

    for comp in _connected_components(target):
        comp_placements: list[dict] = []
        if not _solve_recursive(set(comp), available, comp_placements):
            # Compute which component cells couldn't be covered
            covered = set()
            for p in comp_placements:
                col = round(p["x"] / GRID_STEP)
                row = round(p["y"] / GRID_STEP)
                for dc, dr in _brick_cells_grid(p["brick"], p["rotation"]):
                    covered.add((col + dc, row + dr))
            untiled = comp - covered
            return SolverResult(
                False,
                untiled_cells = sorted(untiled or comp),
                reason        = "NO_TILING",
            )
        all_placements.extend(comp_placements)

    # Add IDs and arm assignments
    for i, p in enumerate(all_placements, 1):
        p["id"]          = i
        p["start_side"]  = "AR4" if p["y"] >= 0 else "ABB"
        p["target_side"] = p["start_side"]
        p["layer"]       = 0

    return SolverResult(success=True, placements=all_placements)


# ─────────────────────────────────────────────────────────────────────────────
# Build a human-readable failure message for the LLM retry
# ─────────────────────────────────────────────────────────────────────────────

def explain_failure(result: SolverResult, total_cells: int) -> str:
    """Generate a specific, actionable explanation for the LLM to redesign."""
    if result.reason == "EMPTY_SHAPE":
        return "No cells were provided. List at least 2 connected cells."

    if result.reason == "ISOLATED_CELLS":
        cells = ", ".join(f"({c},{r})" for c, r in result.untiled_cells[:6])
        return (
            f"The shape has isolated cells with no neighbours: {cells}. "
            f"Every cell must touch at least one other cell horizontally or "
            f"vertically. The smallest brick (I) covers 2 adjacent cells, so "
            f"thicken the design: replace single cells with 2-cell pairs."
        )

    if result.reason == "INVENTORY_EMPTY":
        return "No bricks are available. The engineer needs at least 1 brick."

    if result.reason == "NO_TILING":
        cells = ", ".join(f"({c},{r})" for c, r in result.untiled_cells[:6])
        return (
            f"Cannot tile {len(result.untiled_cells)} cells: {cells}. "
            f"This usually means a region has an odd cell count that no "
            f"combination of I (2 cells), L (3 cells) or T (4 cells) bricks "
            f"can cover exactly, OR the inventory is too small. "
            f"Try: (a) thicken thin parts to width ≥ 2, (b) avoid odd "
            f"protrusions, (c) keep regions a multiple of 2 cells where possible."
        )

    return "Unknown failure."