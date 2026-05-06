"""
inventory.py
Sidebar brick inventory panel.
Shows live counts (from ROS or manual entry) with visual brick tiles.
"""

import streamlit as st
from state import get_inventory, get_ros_status, BrickInventory
from config import cfg


# Brick colours used throughout the UI
BRICK_COLORS = {
    "I": "#378ADD",   # blue
    "L": "#1D9E75",   # teal
    "T": "#EF9F27",   # amber
    "Z": "#D85A30",   # coral
}

BRICK_DESCRIPTIONS = {
    "I": "Straight 3-unit bar",
    "L": "Corner piece",
    "T": "T-junction piece",
    "Z": "Z-shape (handover only)",
}


def render_brick_tile(brick_type: str, count: int):
    """Render a single coloured brick tile with count badge."""
    color = BRICK_COLORS[brick_type]
    desc  = BRICK_DESCRIPTIONS[brick_type]
    empty = count == 0

    opacity = "0.35" if empty else "1.0"
    badge_bg = "#e0e0e0" if empty else color

    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 10px;
            margin-bottom: 6px;
            border-radius: 8px;
            background: {'#f5f5f5' if empty else f'{color}18'};
            border: 1px solid {'#ddd' if empty else color};
            opacity: {opacity};
        ">
            <div style="
                width: 36px; height: 36px;
                background: {color};
                border-radius: 6px;
                display: flex; align-items: center; justify-content: center;
                font-size: 16px; font-weight: 700; color: white;
                flex-shrink: 0;
            ">{brick_type}</div>
            <div style="flex: 1; min-width: 0;">
                <div style="font-size: 13px; font-weight: 600; color: #333;">{brick_type}-brick</div>
                <div style="font-size: 11px; color: #888; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{desc}</div>
            </div>
            <div style="
                min-width: 28px; height: 28px;
                background: {badge_bg};
                border-radius: 14px;
                display: flex; align-items: center; justify-content: center;
                font-size: 14px; font-weight: 700;
                color: {'#999' if empty else 'white'};
                flex-shrink: 0;
            ">{count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ros_status_badge():
    """Small connection status badge at the top of the sidebar."""
    ros = get_ros_status()
    icon  = ros.status_color()
    label = "Connected" if ros.connected else "Mock mode"
    mode  = ros.mode.upper()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"<div style='font-size:22px;line-height:1'>{icon}</div>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown(
            f"<div style='font-size:12px;font-weight:600;margin-top:2px'>{label}</div>"
            f"<div style='font-size:11px;color:#888'>Mode: {mode}</div>",
            unsafe_allow_html=True,
        )


def render_arm_status():
    """Show AR4 and ABB stage badges."""
    ros = get_ros_status()

    stage_colors = {
        "IDLE": ("#e8f5e9", "#2e7d32"),
        "PICK": ("#fff3e0", "#e65100"),
        "PLACE": ("#e3f2fd", "#1565c0"),
        "MOVE_TO_HANDOVER": ("#f3e5f5", "#6a1b9a"),
        "HOLDING_AT_HANDOVER": ("#fce4ec", "#880e4f"),
        "DONE": ("#e8f5e9", "#2e7d32"),
    }

    def badge(label, stage):
        bg, fg = stage_colors.get(stage, ("#f5f5f5", "#555"))
        st.markdown(
            f"""<div style="
                display:flex; justify-content:space-between; align-items:center;
                padding:6px 10px; border-radius:6px;
                background:{bg}; margin-bottom:4px;
            ">
                <span style="font-size:12px;font-weight:600;color:{fg}">{label}</span>
                <span style="font-size:11px;color:{fg};background:white;
                      padding:2px 8px;border-radius:4px;">{stage}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    badge("AR4", ros.ar4_stage)
    badge("ABB", ros.abb_stage)

    if ros.supervisor_state not in ("IDLE", "INIT"):
        st.markdown(
            f"<div style='font-size:11px;color:#888;margin-top:2px;text-align:center'>"
            f"Supervisor: <b>{ros.supervisor_state}</b></div>",
            unsafe_allow_html=True,
        )



def render_llm_status():
    """Small LLM provider badge shown at the bottom of the sidebar."""
    provider = cfg.llm_provider
    model    = cfg.ollama_model if cfg.use_ollama else cfg.llm_model

    if provider == "groq":
        icon, color, label = "⚡", "#1D9E75", f"Groq · {model}"
    elif provider == "ollama":
        icon, color, label = "🖥", "#378ADD", f"Ollama · {model}"
    else:
        icon, color, label = "⚠️", "#D85A30", "No LLM key — set GROQ_API_KEY in .env"

    st.markdown(
        f"""<div style="
            padding:8px 12px;border-radius:8px;margin-top:4px;
            background:{'#e6f7ef' if provider=='groq' else '#e8f1fb' if provider=='ollama' else '#fce8e3'};
            border:1px solid {color};
        ">
            <div style="font-size:12px;font-weight:600;color:{color}">{icon} LLM</div>
            <div style="font-size:11px;color:{color};margin-top:2px">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_inventory_sidebar():
    """
    Full sidebar inventory panel.
    Call this inside a `with st.sidebar:` block.
    """
    st.markdown("### Brick inventory")

    # ── ROS connection status ──────────────────
    render_ros_status_badge()
    st.divider()

    # ── Arm status ────────────────────────────
    st.markdown("**Arm status**")
    render_arm_status()
    st.divider()

    # ── Inventory tiles ───────────────────────
    st.markdown("**Available bricks**")

    inv = get_inventory()
    total = inv.total()

    for brick_type in ["I", "L", "T", "Z"]:
        count = getattr(inv, brick_type)
        render_brick_tile(brick_type, count)

    st.markdown(
        f"<div style='text-align:right;font-size:12px;color:#888;margin-top:4px'>"
        f"Total: <b>{total}</b> bricks</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Manual override ────────────────────────
    with st.expander("Manual inventory override", expanded=False):
        st.caption("Use when ROS is not running or for testing.")
        col1, col2 = st.columns(2)
        with col1:
            new_i = st.number_input("I", min_value=0, max_value=20,
                                    value=inv.I, key="manual_I")
            new_t = st.number_input("T", min_value=0, max_value=20,
                                    value=inv.T, key="manual_T")
        with col2:
            new_l = st.number_input("L", min_value=0, max_value=20,
                                    value=inv.L, key="manual_L")
            new_z = st.number_input("Z", min_value=0, max_value=20,
                                    value=inv.Z, key="manual_Z")

        if st.button("Apply", use_container_width=True):
            st.session_state.inventory = BrickInventory(
                I=new_i, L=new_l, T=new_t, Z=new_z
            )
            st.session_state.manual_inventory = True
            st.rerun()

    # ── Mock mode toggle ──────────────────────
    st.divider()
    mock = st.toggle(
        "Mock mode (no ROS)",
        value=st.session_state.ros_mock_mode,
        help="Run without a live ROS2 system. Inventory and arm status are simulated.",
    )
    if mock != st.session_state.ros_mock_mode:
        st.session_state.ros_mock_mode = mock
        st.rerun()

    # ── LLM status ───────────────────────────────
    st.divider()
    render_llm_status()