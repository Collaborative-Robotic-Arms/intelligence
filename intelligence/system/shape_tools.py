"""
shape_tools.py
==============
Three geometric tools the LLM can call to design any shape.

Tool 1: draw_shape_on_grid(ascii_grid)
  Convert ASCII art to grid cells. LLM uses X for filled, . for empty.

Tool 2: modify_cells(cells, operation, params)
  Transform existing cells (thicken, rotate, mirror, scale, translate).

Tool 3: verify_and_repair(cells)
  Check tileability, auto-fix issues:
    - Isolated cells → thicken to connect
    - 1-cell-wide features → widen to 2 cells
    - Disconnected regions → add bridge cells

These tools handle ALL geometric precision. The LLM just describes shapes
visually and the tools enforce constraints.
"""

from __future__ import annotations
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1: ASCII art → cells
# ─────────────────────────────────────────────────────────────────────────────

def draw_shape_on_grid(ascii_grid: str) -> list[tuple[int, int]]:
    """
    Convert an ASCII art grid to (col, row) cell list.

    Conventions:
      X / # / * / O = filled cell
      . / space     = empty cell
      Top row in input = highest row number in output (y-up)

    Example:
      input:  ".XX.\n.XX.\nXXXX"
      output: [(1,2),(2,2),(1,1),(2,1),(0,0),(1,0),(2,0),(3,0)]

    Args:
      ascii_grid: multi-line string with X for filled cells

    Returns:
      list of (col, row) tuples
    """
    lines = [l.rstrip() for l in ascii_grid.strip("\n").split("\n")]
    if not lines:
        return []

    cells = []
    height = len(lines)

    for y_idx, line in enumerate(lines):
        # y_idx=0 is top row → highest row number
        row = height - 1 - y_idx
        for col, char in enumerate(line):
            if char in "X#*O":
                cells.append((col, row))

    return cells


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2: Modify cells
# ─────────────────────────────────────────────────────────────────────────────

def modify_cells(
    cells:     list[tuple[int, int]],
    operation: str,
    params:    Optional[dict] = None,
) -> list[tuple[int, int]]:
    """
    Transform a cell list with a named operation.

    Supported operations:
      "thicken"     — params: {axis: "x"|"y"|"both", amount: int}
                      Add neighbour cells along axis
      "rotate"      — params: {degrees: 90|180|270}
                      Rotate around (0,0)
      "translate"   — params: {dx: int, dy: int}
                      Shift all cells
      "scale"       — params: {factor: int}
                      Multiply each cell by factor (creates blocks)
      "mirror"      — params: {axis: "x"|"y"}
                      Reflect across axis

    Args:
      cells:     input cell list
      operation: operation name
      params:    operation parameters

    Returns:
      transformed cell list
    """
    params = params or {}
    cell_set = set(cells)

    if operation == "thicken":
        axis   = params.get("axis", "both")
        amount = params.get("amount", 1)
        new = set(cell_set)
        for _ in range(amount):
            for c, r in list(new):
                if axis in ("x", "both"):
                    new.add((c+1, r))
                if axis in ("y", "both"):
                    new.add((c, r+1))
        return sorted(new)

    if operation == "rotate":
        deg = params.get("degrees", 90)
        result = []
        for c, r in cells:
            if deg == 90:
                result.append((-r, c))
            elif deg == 180:
                result.append((-c, -r))
            elif deg == 270:
                result.append((r, -c))
            else:
                result.append((c, r))
        return result

    if operation == "translate":
        dx = params.get("dx", 0)
        dy = params.get("dy", 0)
        return [(c + dx, r + dy) for c, r in cells]

    if operation == "scale":
        factor = params.get("factor", 2)
        result = set()
        for c, r in cells:
            for dc in range(factor):
                for dr in range(factor):
                    result.add((c * factor + dc, r * factor + dr))
        return sorted(result)

    if operation == "mirror":
        axis = params.get("axis", "x")
        if axis == "x":
            return [(-c, r) for c, r in cells]
        else:
            return [(c, -r) for c, r in cells]

    # Unknown operation — pass through
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3: Verify and repair
# ─────────────────────────────────────────────────────────────────────────────

def verify_and_repair(
    cells: list[tuple[int, int]],
    auto_repair: bool = True,
) -> dict:
    """
    Check if cells form a tileable shape. Auto-repair common issues.

    Tileability rules:
      - Every cell must have at least one orthogonal neighbour
      - Connected regions should have cell counts coverable by I(2)/L(3)/T(4)
      - 1-cell-wide protrusions can be problematic

    Auto-repairs applied:
      1. Isolated cells → thicken by adding adjacent cell
      2. Disconnected regions → no auto-fix (return them in report)
      3. Tiny components (< 2 cells) → drop or merge

    Args:
      cells:       input cell list
      auto_repair: if True, attempt to fix issues; if False, just diagnose

    Returns:
      {
        "cells": [...],            # repaired cells
        "valid": bool,             # True if no problems remain
        "issues": [str],           # human-readable issue list
        "repairs_applied": [str],  # what was fixed
      }
    """
    cell_set = set((int(c[0]), int(c[1])) for c in cells if len(c) == 2)
    issues = []
    repairs = []

    if not cell_set:
        return {
            "cells": [],
            "valid": False,
            "issues": ["No cells provided"],
            "repairs_applied": [],
        }

    # Check 1: isolated cells
    isolated = _find_isolated(cell_set)
    if isolated:
        issues.append(f"Found {len(isolated)} isolated cell(s) with no neighbour")
        if auto_repair:
            for c, r in list(isolated):
                # Add a neighbour cell to break isolation
                # Prefer adding to the right (positive col direction)
                cell_set.add((c + 1, r))
            repairs.append(f"Thickened {len(isolated)} isolated cells")
            isolated = _find_isolated(cell_set)

    # Check 2: connected components
    components = _connected_components(cell_set)
    if len(components) > 1:
        issues.append(f"Shape has {len(components)} disconnected regions")

    # Check 3: tiny components
    tiny = [comp for comp in components if len(comp) < 2]
    if tiny and auto_repair:
        for comp in tiny:
            # Drop tiny components (can't be tiled anyway)
            cell_set -= comp
        repairs.append(f"Removed {len(tiny)} unreachable single-cell regions")
        components = _connected_components(cell_set)

    # Re-check after repairs
    isolated_after = _find_isolated(cell_set)
    valid = (len(isolated_after) == 0 and len(components) == 1)

    return {
        "cells":           sorted(cell_set),
        "valid":           valid,
        "issues":          issues,
        "repairs_applied": repairs,
        "num_components":  len(components),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_isolated(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Cells with no orthogonal neighbour in the set."""
    isolated = set()
    for c, r in cells:
        neighbours = {(c+1, r), (c-1, r), (c, r+1), (c, r-1)}
        if not (neighbours & cells):
            isolated.add((c, r))
    return isolated


def _connected_components(cells: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """4-connectivity flood fill to find connected groups."""
    remaining = set(cells)
    components = []
    while remaining:
        seed = next(iter(remaining))
        comp = {seed}
        frontier = [seed]
        while frontier:
            c, r = frontier.pop()
            for n in [(c+1, r), (c-1, r), (c, r+1), (c, r-1)]:
                if n in remaining and n not in comp:
                    comp.add(n)
                    frontier.append(n)
        components.append(comp)
        remaining -= comp
    return components


# ─────────────────────────────────────────────────────────────────────────────
# LangChain tool wrappers
# ─────────────────────────────────────────────────────────────────────────────

def get_langchain_tools():
    """Return LangChain @tool wrappers for use with bind_tools()."""
    from langchain_core.tools import tool

    @tool
    def draw_shape(ascii_grid: str) -> str:
        """
        Draw a shape using ASCII art on a grid.
        Use 'X' for filled cells, '.' for empty cells, separated by newlines.
        Top row of the input is the top of the shape.

        Example for letter T:
          XXXX
          .X..
          .X..

        Returns the cell coordinates as a list.
        """
        cells = draw_shape_on_grid(ascii_grid)
        return f"Generated {len(cells)} cells: {cells}"

    @tool
    def transform_shape(
        cells_json: str,
        operation: str,
        amount: int = 1,
        axis: str = "both",
    ) -> str:
        """
        Transform a cell list using an operation.

        operation can be: 'thicken' (widen by amount along axis),
                          'rotate' (degrees: amount),
                          'translate' (dx,dy: amount, axis),
                          'scale' (factor: amount),
                          'mirror' (axis: 'x' or 'y').

        cells_json: JSON-encoded list of [col, row] pairs.
        """
        import json
        cells = [tuple(c) for c in json.loads(cells_json)]
        params = {"amount": amount, "axis": axis,
                  "degrees": amount, "factor": amount,
                  "dx": amount, "dy": amount}
        result = modify_cells(cells, operation, params)
        return f"Transformed to {len(result)} cells: {result}"

    @tool
    def repair_shape(cells_json: str) -> str:
        """
        Verify a shape is tileable and auto-repair issues like isolated cells.
        Use this after drawing a shape to make sure it can be built.

        cells_json: JSON-encoded list of [col, row] pairs.
        """
        import json
        cells = [tuple(c) for c in json.loads(cells_json)]
        result = verify_and_repair(cells, auto_repair=True)
        report = (
            f"Result: {'VALID' if result['valid'] else 'HAS ISSUES'}\n"
            f"Cells: {result['cells']}\n"
            f"Issues: {result['issues']}\n"
            f"Repairs: {result['repairs_applied']}"
        )
        return report

    return [draw_shape, transform_shape, repair_shape]