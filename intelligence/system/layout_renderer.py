"""
layout_renderer.py
==================
2D top-down layout preview using Matplotlib.

Renders:
  - Table boundary (grey border)
  - AR4 / ABB zone split line
  - AR4 + ABB reach arcs (dashed circles from arm bases)
  - Grid dots at every GRID_STEP
  - Each brick as a filled polygon with:
      • Colour by type  (I=blue, L=teal, T=amber, Z=coral)
      • Opacity by layer (base=1.0, higher=dimmer)
      • Brick ID label
      • Arm assignment badge (AR4 / ABB)
  - Optional: highlight one brick (for animation)

Public API:
    fig = render_layout(plan, highlight_id=None, show_grid=True)
    # fig is a Matplotlib Figure — pass to st.pyplot(fig)
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Optional

from workspace_constraints import (
    TABLE_X_MIN, TABLE_X_MAX, TABLE_Y_MIN, TABLE_Y_MAX,
    GRID_STEP, BRICK_CELLS, ZONE_SPLIT_Y,
    AR4_BASE_X, AR4_BASE_Y, ABB_BASE_X, ABB_BASE_Y,
    AR4_REACH_RADIUS, ABB_REACH_RADIUS,
    BRICK_LAYER_HEIGHT,
)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — matches inventory.py BRICK_COLORS
# ─────────────────────────────────────────────────────────────────────────────
BRICK_FACE = {
    "I": "#378ADD",   # blue
    "L": "#1D9E75",   # teal
    "T": "#EF9F27",   # amber
    "Z": "#D85A30",   # coral
}
BRICK_EDGE = {
    "I": "#1A5FA8",
    "L": "#0D6E4F",
    "T": "#B86E00",
    "Z": "#9E3010",
}
ARM_COLORS = {
    "AR4": "#7B2D8B",   # purple
    "ABB": "#1E6B3C",   # dark green
}


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rotate_point(x: float, y: float, deg: int) -> tuple[float, float]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return c * x - s * y, s * x + c * y


def _brick_polygon(
    brick_type: str, x: float, y: float, rotation: int
) -> list[tuple[float, float]]:
    """
    Return the outline polygon vertices of a brick in world coords.
    Uses the actual SDF-derived cell footprints.
    """
    cells = BRICK_CELLS.get(brick_type, [(0.0, 0.0)])
    G = GRID_STEP
    margin = 0.0003  # tiny inset — adjacent bricks visually connect

    # Build union of cell rectangles → list of corner points
    # For rendering we just draw each cell as a separate rectangle
    rects = []
    for (dx, dy) in cells:
        # Rotate offset
        rdx, rdy = _rotate_point(dx, dy, rotation)
        cx, cy = x + rdx, y + rdy

        # Cell corners before rotation
        corners_local = [
            (margin,     margin),
            (G - margin, margin),
            (G - margin, G - margin),
            (margin,     G - margin),
        ]
        # Rotate each corner and translate
        rotated = []
        for lx, ly in corners_local:
            rx, ry = _rotate_point(lx, ly, rotation)
            rotated.append((cx + rx, cy + ry))
        rects.append(rotated)

    return rects   # list of polygons (one per cell)


def _draw_brick(
    ax,
    brick: dict,
    highlight: bool = False,
    placed: bool = True,
    alpha_override: Optional[float] = None,
):
    """Draw a single brick on the axes."""
    btype    = brick.get("brick", "I")
    x        = brick.get("x", 0.0)
    y        = brick.get("y", 0.0)
    rotation = brick.get("rotation", 0)
    layer    = brick.get("layer", 0)
    bid      = brick.get("id", "?")
    side     = brick.get("start_side", "AR4")

    face  = BRICK_FACE.get(btype, "#888888")
    edge  = BRICK_EDGE.get(btype, "#333333")

    # Dim higher layers slightly
    base_alpha = max(0.4, 1.0 - layer * 0.15)
    alpha = alpha_override if alpha_override is not None else base_alpha

    if not placed:
        alpha = 0.15   # ghost for unplaced bricks

    if highlight:
        edge = "#FF0000"
        lw   = 2.5
    else:
        lw   = 1.2

    rects = _brick_polygon(btype, x, y, rotation)

    for poly_verts in rects:
        xs = [v[0] for v in poly_verts] + [poly_verts[0][0]]
        ys = [v[1] for v in poly_verts] + [poly_verts[0][1]]
        ax.fill(xs, ys, color=face, alpha=alpha, zorder=3)
        ax.plot(xs, ys, color=edge, linewidth=lw, alpha=min(1.0, alpha + 0.2), zorder=4)

    # Compute centroid of all cells for label placement
    all_verts = [v for rect in rects for v in rect]
    cx = sum(v[0] for v in all_verts) / len(all_verts)
    cy = sum(v[1] for v in all_verts) / len(all_verts)

    # Brick type label
    ax.text(cx, cy + 0.004, btype,
            ha="center", va="center", fontsize=6, fontweight="bold",
            color="white", zorder=5)

    # ID badge
    ax.text(cx, cy - 0.006, f"#{bid}",
            ha="center", va="center", fontsize=4.5,
            color="white", alpha=0.85, zorder=5)

    # Arm badge (small coloured dot)
    dot_color = ARM_COLORS.get(side, "#555")
    ax.plot(cx + 0.012, cy + 0.008, "o",
            color=dot_color, markersize=3, zorder=6, alpha=alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_layout(
    plan,
    highlight_id:   Optional[int] = None,
    placed_ids:     Optional[set] = None,
    show_grid:      bool = True,
    show_reach:     bool = True,
    figsize:        tuple = (6, 6),
    title:          str = "",
) -> plt.Figure:
    """
    Render the assembly plan as a 2D top-down view.

    Args:
        plan          : state.AssemblyPlan
        highlight_id  : brick id to highlight in red (for animation)
        placed_ids    : set of brick ids already placed (others are ghosts)
        show_grid     : draw dot grid
        show_reach    : draw arm reach arcs
        figsize       : Matplotlib figure size
        title         : figure title

    Returns:
        matplotlib.figure.Figure  — pass directly to st.pyplot()
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1A1A2E")
    ax.set_facecolor("#1A1A2E")

    # ── Table boundary ────────────────────────────────────────────
    table_rect = mpatches.FancyBboxPatch(
        (TABLE_X_MIN, TABLE_Y_MIN),
        TABLE_X_MAX - TABLE_X_MIN,
        TABLE_Y_MAX - TABLE_Y_MIN,
        boxstyle="round,pad=0.002",
        linewidth=2, edgecolor="#4A4A6A", facecolor="#252540",
        zorder=0,
    )
    ax.add_patch(table_rect)

    # ── Zone split line ───────────────────────────────────────────
    ax.axhline(y=ZONE_SPLIT_Y, color="#3A3A5A", linewidth=1.5,
               linestyle="--", alpha=0.7, zorder=1)
    ax.text(TABLE_X_MAX - 0.01, ZONE_SPLIT_Y + 0.01,
            "AR4 side", ha="right", va="bottom",
            fontsize=6, color=ARM_COLORS["AR4"], alpha=0.8)
    ax.text(TABLE_X_MAX - 0.01, ZONE_SPLIT_Y - 0.01,
            "ABB side", ha="right", va="top",
            fontsize=6, color=ARM_COLORS["ABB"], alpha=0.8)

    # ── Grid dots ─────────────────────────────────────────────────
    if show_grid:
        gx = np.arange(TABLE_X_MIN, TABLE_X_MAX + GRID_STEP/2, GRID_STEP)
        gy = np.arange(TABLE_Y_MIN, TABLE_Y_MAX + GRID_STEP/2, GRID_STEP)
        for gxi in gx:
            for gyi in gy:
                ax.plot(gxi, gyi, ".", color="#3A3A5A",
                        markersize=1.5, zorder=1, alpha=0.6)

    # ── Arm reach arcs ────────────────────────────────────────────
    if show_reach:
        for (bx, by, radius, arm) in [
            (AR4_BASE_X, AR4_BASE_Y, AR4_REACH_RADIUS, "AR4"),
            (ABB_BASE_X, ABB_BASE_Y, ABB_REACH_RADIUS, "ABB"),
        ]:
            color = ARM_COLORS[arm]
            circle = plt.Circle(
                (bx, by), radius,
                color=color, fill=False,
                linewidth=1, linestyle=":", alpha=0.25, zorder=1,
            )
            ax.add_patch(circle)
            # Arm base marker
            ax.plot(bx, by, marker="^" if arm == "AR4" else "v",
                    color=color, markersize=8, zorder=2, alpha=0.5)
            ax.text(bx, by + (0.02 if arm == "AR4" else -0.03),
                    arm, ha="center", va="center",
                    fontsize=6, color=color, alpha=0.7, fontweight="bold")

    # ── Bricks ────────────────────────────────────────────────────
    arrangement = plan.arrangement if hasattr(plan, "arrangement") else []

    for brick in arrangement:
        bid = brick.get("id", -1)
        is_highlight = (bid == highlight_id)
        is_placed    = (placed_ids is None) or (bid in placed_ids)

        _draw_brick(ax, brick,
                    highlight=is_highlight,
                    placed=is_placed)

    # ── Legend ────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=BRICK_FACE[t], label=f"{t}-brick")
        for t in ["I", "L", "T", "Z"]
        if any(b.get("brick") == t for b in arrangement)
    ]
    if legend_patches:
        ax.legend(
            handles=legend_patches,
            loc="upper left", fontsize=5.5,
            facecolor="#252540", edgecolor="#4A4A6A",
            labelcolor="white", framealpha=0.8,
        )

    # ── Axes formatting ───────────────────────────────────────────
    pad = 0.04
    ax.set_xlim(TABLE_X_MIN - pad, TABLE_X_MAX + pad)
    ax.set_ylim(TABLE_Y_MIN - pad, TABLE_Y_MAX + pad)
    ax.set_aspect("equal")
    ax.tick_params(colors="#6A6A8A", labelsize=6)
    ax.xaxis.label.set_color("#6A6A8A")
    ax.yaxis.label.set_color("#6A6A8A")
    for spine in ax.spines.values():
        spine.set_edgecolor("#4A4A6A")

    # Axis labels in cm for readability
    ax.set_xlabel("X (m)", fontsize=7, color="#6A6A8A")
    ax.set_ylabel("Y (m) — AR4↑  ABB↓", fontsize=7, color="#6A6A8A")

    title_str = title or (
        f"{plan.structure}  —  {len(arrangement)} bricks"
        if hasattr(plan, "structure") else "Layout preview"
    )
    ax.set_title(title_str, fontsize=9, color="white",
                 pad=8, fontweight="bold")

    plt.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Plan summary helper for the sidebar card
# ─────────────────────────────────────────────────────────────────────────────

def plan_summary(plan) -> dict:
    """
    Return a summary dict for the UI card:
      layers, brick_counts, bounding_box
    """
    from collections import Counter
    arrangement = plan.arrangement if hasattr(plan, "arrangement") else []
    counts = Counter(b.get("brick", "?") for b in arrangement)
    layers = max((b.get("layer", 0) for b in arrangement), default=0)

    xs = [b.get("x", 0) for b in arrangement]
    ys = [b.get("y", 0) for b in arrangement]

    return {
        "total_bricks": len(arrangement),
        "brick_counts": dict(counts),
        "layers":       layers,
        "height_m":     (layers + 1) * BRICK_LAYER_HEIGHT,
        "x_span":       (min(xs, default=0), max(xs, default=0)),
        "y_span":       (min(ys, default=0), max(ys, default=0)),
        "ar4_bricks":   sum(1 for b in arrangement
                            if b.get("start_side") == "AR4"),
        "abb_bricks":   sum(1 for b in arrangement
                            if b.get("start_side") == "ABB"),
    }