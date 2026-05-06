"""
demo_animator.py
================
Pre-execution demo animation — shows bricks appearing one by one
in their final positions before any ROS call is made.

Two modes:
  step_through()  — user clicks Next/Prev to step manually
  auto_play()     — renders frames at a set speed using st.empty()

Usage in app.py:
    from demo_animator import DemoAnimator
    animator = DemoAnimator(plan)
    animator.render_controls()    # draws Play/Pause/Step buttons
    animator.render_current_frame()  # draws current frame
"""

from __future__ import annotations

import time
import streamlit as st

from layout_renderer import render_layout, plan_summary, BRICK_FACE, ARM_COLORS


class DemoAnimator:
    """
    Manages animation state for a single AssemblyPlan.
    State is stored in st.session_state so it survives Streamlit reruns.
    """

    # session_state key prefix
    _KEY = "demo_anim"

    def __init__(self, plan):
        self._plan = plan
        self._arrangement = sorted(
            plan.arrangement if hasattr(plan, "arrangement") else [],
            key=lambda b: (b.get("layer", 0), b.get("id", 0)),
        )
        self._total = len(self._arrangement)
        self._init_state()

    # ── State management ─────────────────────────────────────────────────────

    def _init_state(self):
        if f"{self._KEY}_frame" not in st.session_state:
            st.session_state[f"{self._KEY}_frame"]   = 0
            st.session_state[f"{self._KEY}_playing"] = False
            st.session_state[f"{self._KEY}_speed"]   = 1.0   # seconds per frame

    @property
    def _frame(self) -> int:
        return st.session_state.get(f"{self._KEY}_frame", 0)

    @_frame.setter
    def _frame(self, v: int):
        st.session_state[f"{self._KEY}_frame"] = max(0, min(v, self._total))

    @property
    def _playing(self) -> bool:
        return st.session_state.get(f"{self._KEY}_playing", False)

    @_playing.setter
    def _playing(self, v: bool):
        st.session_state[f"{self._KEY}_playing"] = v

    @property
    def _speed(self) -> float:
        return st.session_state.get(f"{self._KEY}_speed", 1.0)

    # ── Public API ───────────────────────────────────────────────────────────

    def render_controls(self):
        """
        Draw the animation control bar:
          [◀ Reset]  [◀ Prev]  [▶ Play / ⏸ Pause]  [Next ▶]  [⏭ End]
        and a speed slider + progress bar.
        """
        if self._total == 0:
            st.info("No bricks in plan.")
            return

        # Progress bar
        progress = self._frame / self._total if self._total > 0 else 0
        st.progress(progress,
                    text=f"Step {self._frame} / {self._total}")

        # Control buttons
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            if st.button("⏮ Reset", use_container_width=True):
                self._frame   = 0
                self._playing = False
                st.rerun()

        with c2:
            if st.button("◀ Prev", use_container_width=True,
                         disabled=(self._frame == 0)):
                self._frame  -= 1
                self._playing = False
                st.rerun()

        with c3:
            if self._playing:
                if st.button("⏸ Pause", use_container_width=True,
                             type="primary"):
                    self._playing = False
                    st.rerun()
            else:
                if st.button("▶ Play", use_container_width=True,
                             type="primary",
                             disabled=(self._frame >= self._total)):
                    self._playing = True
                    st.rerun()

        with c4:
            if st.button("Next ▶", use_container_width=True,
                         disabled=(self._frame >= self._total)):
                self._frame  += 1
                self._playing = False
                st.rerun()

        with c5:
            if st.button("⏭ End", use_container_width=True):
                self._frame   = self._total
                self._playing = False
                st.rerun()

        # Speed slider
        speed = st.slider(
            "Animation speed (sec/brick)",
            min_value=0.3, max_value=3.0,
            value=self._speed, step=0.1,
            key=f"{self._KEY}_speed_slider",
        )
        st.session_state[f"{self._KEY}_speed"] = speed

    def render_current_frame(self, container=None):
        """
        Render the current animation frame into `container`
        (defaults to st directly).

        If playing, advance frame and rerun after a delay.
        """
        target = container or st

        placed_ids = {
            self._arrangement[i]["id"]
            for i in range(self._frame)
        }

        # Highlight the brick ABOUT to be placed
        highlight_id = None
        if 0 < self._frame <= self._total:
            highlight_id = self._arrangement[self._frame - 1]["id"]

        # Build title
        if self._frame == 0:
            title = f"{self._plan.structure}  —  press ▶ Play to start"
        elif self._frame >= self._total:
            title = f"{self._plan.structure}  —  ✅ Assembly complete!"
        else:
            brick = self._arrangement[self._frame - 1]
            title = (f"Placing brick #{brick['id']} "
                     f"({brick.get('brick','?')}-brick "
                     f"→ {brick.get('target_side','?')})")

        fig = render_layout(
            self._plan,
            highlight_id = highlight_id,
            placed_ids   = placed_ids,
            show_grid    = True,
            show_reach   = (self._frame == 0),   # show reach only on overview
            title        = title,
        )

        target.pyplot(fig, use_container_width=True)
        import matplotlib.pyplot as plt
        plt.close(fig)   # free memory

        # Step info card
        if 0 < self._frame <= self._total:
            brick = self._arrangement[self._frame - 1]
            self._render_step_card(brick)

        # Auto-advance if playing
        if self._playing and self._frame < self._total:
            time.sleep(self._speed)
            self._frame += 1
            if self._frame >= self._total:
                self._playing = False
            st.rerun()

    def _render_step_card(self, brick: dict):
        """Small info card below the frame showing current brick details."""
        btype  = brick.get("brick", "?")
        side   = brick.get("start_side", "?")
        target = brick.get("target_side", "?")
        x      = brick.get("x", 0)
        y      = brick.get("y", 0)
        rot    = brick.get("rotation", 0)
        layer  = brick.get("layer", 0)
        color  = BRICK_FACE.get(btype, "#888")
        arm_c  = ARM_COLORS.get(side, "#555")

        st.markdown(
            f"""
            <div style="
                display:flex; gap:12px; padding:10px 14px;
                background:#252540; border-radius:8px;
                border:1px solid #4A4A6A; margin-top:6px;
            ">
                <div style="
                    width:40px;height:40px;border-radius:6px;
                    background:{color};display:flex;align-items:center;
                    justify-content:center;font-size:16px;font-weight:700;
                    color:white;flex-shrink:0
                ">{btype}</div>
                <div style="flex:1">
                    <div style="color:white;font-size:13px;font-weight:600">
                        {btype}-brick  &nbsp;·&nbsp;  ID #{brick.get('id','?')}
                        &nbsp;·&nbsp; Layer {layer}
                    </div>
                    <div style="color:#9A9ABB;font-size:11px;margin-top:3px">
                        Position ({x:.3f}, {y:.3f}) m &nbsp;·&nbsp;
                        Rotation {rot}° &nbsp;·&nbsp;
                        <span style="color:{arm_c};font-weight:600">{side}</span>
                        &nbsp;→&nbsp;
                        <span style="color:{arm_c};font-weight:600">{target}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def reset(self):
        self._frame   = 0
        self._playing = False


# ─────────────────────────────────────────────────────────────────────────────
# Standalone full preview (no animation) — used in the main layout panel
# ─────────────────────────────────────────────────────────────────────────────

def render_static_preview(plan, container=None):
    """
    Render a static (non-animated) layout preview of the full plan.
    All bricks shown at full opacity.
    Used in the right panel of app.py before the user starts the demo.
    """
    target = container or st

    if not plan or not getattr(plan, "arrangement", None):
        target.markdown(
            """
            <div style="height:360px;border:2px dashed #4A4A6A;border-radius:12px;
                 display:flex;flex-direction:column;align-items:center;
                 justify-content:center;color:#6A6A8A;font-size:14px;text-align:center">
                <div style="font-size:36px;margin-bottom:10px">📐</div>
                <div>Layout preview will appear here</div>
                <div style="font-size:12px;margin-top:4px">once a plan is generated</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    fig = render_layout(plan, show_grid=True, show_reach=True)
    target.pyplot(fig, use_container_width=True)
    import matplotlib.pyplot as plt
    plt.close(fig)

    # Summary stats below the chart
    summary = plan_summary(plan)
    col1, col2, col3 = target.columns(3)
    with col1:
        st.metric("Total bricks", summary["total_bricks"])
    with col2:
        st.metric("AR4", f"{summary['ar4_bricks']} bricks")
    with col3:
        st.metric("ABB", f"{summary['abb_bricks']} bricks")