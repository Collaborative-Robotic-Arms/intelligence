"""
app.py
Main Streamlit application — Task 1 skeleton.

Layout:
  ┌─────────────┬──────────────────────────────────┐
  │  Sidebar    │  Main area                        │
  │  Inventory  │  ┌────────────────────────────┐   │
  │  Arm status │  │  Chat window               │   │
  │  ROS badge  │  │  (messages + input)        │   │
  │             │  └────────────────────────────┘   │
  │             │  ┌────────────────────────────┐   │
  │             │  │  2D layout preview         │   │
  │             │  │  (placeholder for Task 5)  │   │
  │             │  └────────────────────────────┘   │
  └─────────────┴──────────────────────────────────┘
"""

import streamlit as st
import time
import os
from config import cfg

# ── Page config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Robot Assembly AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local modules ─────────────────────────────────────────────────
from state import (
    init_state, add_message, get_chat_history,
    get_inventory, get_ros_status, get_plan,
    clear_chat, RosStatus,
)
from inventory import render_inventory_sidebar

# ── Initialise all session state keys ────────────────────────────
init_state()

# ═════════════════════════════════════════════
# ROS2 BRIDGE — singleton per session
# ═════════════════════════════════════════════

from ros_bridge import RosBridge

# Initialise once per browser session
if "ros_bridge" not in st.session_state:
    st.session_state.ros_bridge = RosBridge(
        mock=st.session_state.ros_mock_mode,
        mode="sim",
    )
    st.session_state.ros_bridge.start()

bridge: RosBridge = st.session_state.ros_bridge

# Restart bridge if user toggled mock mode
if bridge.is_mock != st.session_state.ros_mock_mode:
    bridge.restart(mock=st.session_state.ros_mock_mode, mode="sim")

# Sync latest snapshot into session_state on every Streamlit rerun.
# Skip inventory sync if user has manually overridden it in the sidebar.
if not st.session_state.manual_inventory:
    bridge.sync_to_state(
        st.session_state.inventory,
        st.session_state.ros_status,
    )
else:
    status = bridge.get_status()
    ros = st.session_state.ros_status
    ros.connected        = status["connected"]
    ros.mode             = status["mode"]
    ros.ar4_stage        = status["ar4_stage"]
    ros.abb_stage        = status["abb_stage"]
    ros.supervisor_state = status["supervisor_state"]
    ros.zone_status      = status["zone_status"]
    ros.last_update      = status["last_update"]


# ═════════════════════════════════════════════
# INTERPRETER CHAIN — singleton per session
# ═════════════════════════════════════════════

if "interpreter" not in st.session_state:
    from interpreter_chain import InterpreterChain
    st.session_state.interpreter = InterpreterChain()

if "validator" not in st.session_state:
    from validator_chain import ValidatorChain
    st.session_state.validator = ValidatorChain()

if "agent" not in st.session_state:
    from agent import AssemblyAgent
    st.session_state.agent = AssemblyAgent()

# ── Memory & session store ────────────────────────────────────────
from memory import get_memory
from session_store import SessionStore

if "conversation_memory" not in st.session_state:
    from memory import ConversationMemory
    st.session_state.conversation_memory = ConversationMemory()

if "session_store" not in st.session_state:
    store = SessionStore()
    st.session_state.session_store = store
    store.load_into(st.session_state)   # restore on first load

if "_session_id" not in st.session_state:
    st.session_state._session_id = st.session_state.session_store.session_id

if "agent" not in st.session_state:
    from agent import AssemblyAgent
    st.session_state.agent = AssemblyAgent()

# Execution stop flag (set True by emergency stop button)
if "stop_execution" not in st.session_state:
    st.session_state.stop_execution = False



# ═════════════════════════════════════════════
# PLACEHOLDER RESPONSE LOGIC (Task 1 only)
# Replaced by LangChain agent in Task 3
# ═════════════════════════════════════════════


def _format_interpreter_result(result, inv) -> tuple[str, object]:
    """
    Convert the new 3-intent InterpreterResult into a (chat_message, plan) pair.
      - intent="design" : message about the new scenario, return the plan
      - intent="chat"   : conversational message + optional suggestions
      - intent="error"  : error message, no plan
    Returns: (chat_message_str, AssemblyPlan | None)
    """

    # ── intent: chat ───────────────────────────────────────────────
    if result.intent == "chat":
        msg = result.chat_message or "Let me know what you would like to build."
        if result.suggestions:
            msg += "\n\n**Ideas:**"
            for s in result.suggestions:
                msg += f"\n- {s}"
        return msg, None

    # ── intent: error ──────────────────────────────────────────────
    if result.intent == "error":
        return (result.chat_message or
                f"❌ {result.error_reason or 'I could not process that request.'}",
                None)

    # ── intent: design (success) ───────────────────────────────────
    if result.intent == "design" and result.plan is not None:
        plan = result.plan
        brick_counts = {}
        for b in plan.required_bricks:
            brick_counts[b] = brick_counts.get(b, 0) + 1
        brick_summary = ", ".join(f"{v}×{k}" for k, v in sorted(brick_counts.items()))

        # Inventory pre-check
        shortages = []
        for btype, needed in brick_counts.items():
            have = getattr(inv, btype, 0)
            if have < needed:
                shortages.append(f"{needed}×{btype} (have {have})")

        msg_lines = []
        if result.chat_message:
            msg_lines.append(result.chat_message)
        msg_lines.append(f"\n📐 **{plan.structure}** — {brick_summary} "
                         f"({len(plan.arrangement)} placements)")

        if shortages:
            msg_lines.append(
                f"\n⚠️ Inventory short on: {', '.join(shortages)} — "
                f"the design is shown but you may need more bricks before executing."
            )
        else:
            msg_lines.append(
                "\n✅ Inventory sufficient. Layout updated below — "
                "press **Execute Simulation** when ready to build it in Gazebo."
            )

        return "\n".join(msg_lines), plan

    # Fallback (shouldn't happen)
    return result.chat_message or "Sorry, I had trouble with that.", None


# ═════════════════════════════════════════════
# CUSTOM CSS
# ═════════════════════════════════════════════

st.markdown("""
<style>
/* Hide default Streamlit header/footer */
#MainMenu, footer { visibility: hidden; }

/* Chat bubble styles */
.chat-bubble {
    padding: 10px 14px;
    border-radius: 12px;
    margin-bottom: 8px;
    max-width: 85%;
    line-height: 1.5;
    font-size: 14px;
}
.chat-user {
    background: #E6F1FB;
    border: 1px solid #B5D4F4;
    color: #042C53;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}
.chat-assistant {
    background: #F1EFE8;
    border: 1px solid #D3D1C7;
    color: #2C2C2A;
    margin-right: auto;
    border-bottom-left-radius: 4px;
}
.chat-system {
    background: #FAEEDA;
    border: 1px solid #FAC775;
    color: #412402;
    margin: 4px auto;
    text-align: center;
    font-size: 12px;
    border-radius: 20px;
    padding: 4px 12px;
}
.chat-wrapper {
    display: flex;
    flex-direction: column;
}
/* Compact the sidebar number inputs */
.stNumberInput > div > div > input {
    padding: 4px 8px !important;
}
/* Status bar at the bottom of the main area */
.status-bar {
    padding: 6px 14px;
    background: #F1EFE8;
    border-top: 1px solid #D3D1C7;
    border-radius: 0 0 8px 8px;
    font-size: 12px;
    color: #5F5E5A;
    display: flex;
    gap: 16px;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<h2 style='margin-bottom:0;font-size:20px'>🤖 Assembly AI</h2>"
        "<p style='color:#888;font-size:12px;margin-top:2px'>Dual-arm robotic system</p>",
        unsafe_allow_html=True,
    )
    render_inventory_sidebar()


# ═════════════════════════════════════════════
# MAIN AREA HEADER
# ═════════════════════════════════════════════

header_col, ctrl_col = st.columns([3, 1])

with header_col:
    st.markdown(
        "<h1 style='font-size:26px;margin-bottom:0'>Assembly scenario planner</h1>"
        "<p style='color:#888;font-size:13px;margin-top:2px'>"
        "Describe what you want to build in plain language</p>",
        unsafe_allow_html=True,
    )

with ctrl_col:
    ros = get_ros_status()
    zone_color = {"CLEAR": "🟢", "COLLISION_WARNING": "🟡"}.get(ros.zone_status, "⚪")
    executing  = st.session_state.get("executing", False)

    st.markdown(
        f"<div style='text-align:right;padding-top:4px'>"
        f"<div style='font-size:11px;color:#888'>Zone</div>"
        f"<div style='font-size:15px'>{zone_color} {ros.zone_status}</div>"
        f"<div style='font-size:11px;color:#888;margin-top:2px'>Agent</div>"
        f"<div style='font-size:13px;font-weight:600;"
        f"color:{"#D85A30" if executing else "#1D9E75"}'>"
        f"{'⏳ RUNNING' if executing else '✅ IDLE'}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if executing:
        if st.button("🛑 Stop", type="primary", use_container_width=True,
                     help="Emergency stop — cancel all robot goals"):
            bridge.emergency_stop()
            st.session_state.executing = False
            add_message("system", "⚠️ EMERGENCY STOP — all goals cancelled.")
            st.rerun()

st.divider()


# ═════════════════════════════════════════════
# MAIN LAYOUT: chat + preview
# ═════════════════════════════════════════════

chat_col, preview_col = st.columns([3, 2])


# ─────────────────────────────────────────────
# LEFT: CHAT WINDOW
# ─────────────────────────────────────────────

with chat_col:
    st.markdown("#### Conversation")

    # ── Message display area ──────────────────
    chat_container = st.container(height=420, border=True)

    with chat_container:
        history = get_chat_history()

        if not history:
            st.markdown(
                "<div style='color:#aaa;text-align:center;padding:60px 20px;font-size:14px'>"
                "No messages yet.<br>Try: <i>\"Build a T shape using available bricks\"</i>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            for msg in history:
                role = msg.role
                content = msg.content
                css_class = {
                    "user": "chat-user",
                    "assistant": "chat-assistant",
                    "system": "chat-system",
                }.get(role, "chat-assistant")

                if role == "system":
                    st.markdown(
                        f'<div class="chat-wrapper">'
                        f'<div class="chat-bubble {css_class}">{content}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    label = "You" if role == "user" else "AI"
                    align = "right" if role == "user" else "left"
                    st.markdown(
                        f'<div class="chat-wrapper" style="align-items:flex-{"end" if role=="user" else "start"}">'
                        f'<div style="font-size:11px;color:#aaa;margin-bottom:2px;text-align:{align}">{label}</div>'
                        f'<div class="chat-bubble {css_class}">{content}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Input bar ─────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        input_col, btn_col = st.columns([5, 1])
        with input_col:
            user_input = st.text_input(
                "Your message",
                placeholder='e.g. "Build a T shape using 3 I-bricks"',
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_input.strip():
        user_text = user_input.strip()
        add_message("user", user_text)

        # ── Interpreter chain (Task 3) ──────────────────────────────
        inv = get_inventory()
        if cfg.has_groq_key or cfg.use_ollama:
            interpreter = st.session_state.interpreter
            mem = get_memory()
            # Determine current scenario for context
            current_plan = get_plan()
            current_scenario_name = (current_plan.structure
                                     if current_plan and current_plan.arrangement
                                     else "none")

            with st.spinner("Thinking…"):
                result = interpreter.interpret(
                    user_input       = user_text,
                    inventory        = inv.as_dict(),
                    current_scenario = current_scenario_name,
                )

            # Display the agent reasoning trace if available (Q3=a)
            if hasattr(result, "agent_trace") and result.agent_trace:
                add_message("assistant", result.agent_trace)

            response, plan_update = _format_interpreter_result(result, inv)
            add_message("assistant", response)

            if plan_update is not None:
                from state import set_plan
                set_plan(plan_update)
                st.session_state.waiting_for_confirmation = False

                # ── Update memory context ─────────────────────────
                mem.update_context("last_structure", plan_update.structure)
                mem.update_context("last_plan",      plan_update.arrangement)
                mem.update_context("last_inventory",  inv.as_dict())
                mem.add_user(user_text)

                # ── Auto-validate immediately after interpretation ──
                with st.spinner("Validating plan…"):
                    from validator_chain import (
                        apply_validation_to_plan,
                        format_validation_chat_message,
                    )
                    val_result = st.session_state.validator.validate(
                        plan      = plan_update,
                        inventory = inv.as_dict(),
                    )
                apply_validation_to_plan(plan_update, val_result)
                val_msg = format_validation_chat_message(val_result)
                add_message("assistant", val_msg)

                # If validator suggests a fix, enter refinement mode
                if val_result.status == "suggest":
                    st.session_state.waiting_for_confirmation = True

                # Complete memory turn + autosave
                mem.add_assistant(val_msg, metadata={
                    "plan_structure": plan_update.structure if plan_update else "",
                    "validation_status": val_result.status,
                })
                st.session_state.session_store.save(st.session_state)
        else:
            add_message("assistant",
                "⚠️ No LLM configured. Set GROQ_API_KEY in .env file.")

        st.rerun()

    # ── ROS action buttons ───────────────────
    ros_col1, ros_col2 = st.columns(2)
    with ros_col1:
        if st.button("🔍 Detect bricks", use_container_width=True,
                     help="Trigger detect_bricks service and refresh inventory"):
            with st.spinner("Detecting…"):
                bricks = bridge.call_detect_service()
            inv = get_inventory()
            lines = "\n".join(
                f"  • {b['type']}-brick (id={b['id']}, side={b['side']})"
                for b in bricks
            ) or "  No bricks detected."
            add_message("system",
                f"Detection complete — {len(bricks)} brick(s) found:\n{lines}")
            if not st.session_state.manual_inventory:
                bridge.sync_to_state(st.session_state.inventory,
                                     st.session_state.ros_status)
            st.rerun()

    with ros_col2:
        if st.button("🛑 Emergency stop", use_container_width=True,
                     type="primary" if st.session_state.executing else "secondary",
                     help="Cancel all active robot goals immediately"):
            bridge.emergency_stop()
            add_message("system", "⚠️ EMERGENCY STOP sent — all arm goals cancelled.")
            st.session_state.executing = False
            st.rerun()

    st.divider()

    # ── Quick action buttons ──────────────────
    st.markdown("**Quick actions**")
    qa_cols = st.columns(4)
    quick_actions = [
        ("T-shape", "Build a T shape using available bricks"),
        ("I-line",  "Build a straight line using I bricks"),
        ("L-corner","Build an L shape corner"),
        ("Clear",   None),
    ]
    for col, (label, prompt) in zip(qa_cols, quick_actions):
        with col:
            if label == "Clear":
                if st.button("🗑 Clear", use_container_width=True):
                    clear_chat()
                    st.session_state.current_plan = None
                    st.session_state.show_layout  = False
                    # Also clear memory + interpreter history
                    mem = get_memory()
                    mem.clear()
                    if "interpreter" in st.session_state:
                        st.session_state.interpreter.clear_history()
                    st.rerun()
            else:
                if st.button(label, use_container_width=True):
                    add_message("user", prompt)
                    inv = get_inventory()
                    if cfg.has_groq_key or cfg.use_ollama:
                        interpreter = st.session_state.interpreter
                        current_plan = get_plan()
                        cur_name = (current_plan.structure
                                    if current_plan and current_plan.arrangement
                                    else "none")
                        with st.spinner("Thinking…"):
                            result = interpreter.interpret(
                                user_input       = prompt,
                                inventory        = inv.as_dict(),
                                current_scenario = cur_name,
                            )
                        response, plan_update = _format_interpreter_result(result, inv)
                        if plan_update is not None:
                            from state import set_plan
                            set_plan(plan_update)
                    else:
                        add_message("assistant",
                            "⚠️ No LLM configured. Set GROQ_API_KEY in .env file.")
                    st.rerun()


# ─────────────────────────────────────────────
# RIGHT: 2D LAYOUT PREVIEW
# ─────────────────────────────────────────────

with preview_col:
    from demo_animator import render_static_preview, DemoAnimator

    plan = get_plan()
    show = st.session_state.show_layout

    # ── Tab switcher: Static preview | Animation ──────────────────
    if show and plan and plan.arrangement:
        tab_preview, tab_anim = st.tabs(["📐 Layout", "▶ Demo animation"])
    else:
        tab_preview = st.container()
        tab_anim    = None

    # ── Tab 1: Static layout ─────────────────────────────────────
    with tab_preview:
        if not show or plan is None or not plan.arrangement:
            st.markdown(
                """
                <div style="height:360px;border:2px dashed #4A4A6A;
                     border-radius:12px;display:flex;flex-direction:column;
                     align-items:center;justify-content:center;
                     color:#6A6A8A;font-size:14px;text-align:center;
                     background:#1A1A2E;padding:20px">
                    <div style="font-size:36px;margin-bottom:10px">📐</div>
                    <div>Layout preview will appear here</div>
                    <div style="font-size:12px;margin-top:4px;color:#4A4A6A">
                        once a plan is generated</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Plan status card
            val_icon  = "✅" if plan.validated else "⏳"
            val_color = "#27500A" if plan.validated else "#5F4A00"
            val_bg    = "#EAF3DE" if plan.validated else "#FFF8E0"
            val_border= "#C0DD97" if plan.validated else "#E8D070"
            st.markdown(
                f"""<div style="padding:10px 14px;background:{val_bg};
                    border:1px solid {val_border};border-radius:8px;
                    margin-bottom:8px">
                    <div style="font-size:14px;font-weight:600;color:{val_color}">
                        {val_icon} {plan.structure}
                    </div>
                    <div style="font-size:11px;color:{val_color};margin-top:3px">
                        {plan.validation_message or "Awaiting validation…"}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # Render the Matplotlib chart
            render_static_preview(plan)

            # Execute / Demo buttons
            st.markdown("---")
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if plan.validated:
                    lbl = "⏳ Running…" if st.session_state.executing                           else "▶ Execute plan"
                    if st.button(lbl, type="primary",
                                 use_container_width=True,
                                 disabled=st.session_state.executing):
                        st.session_state.executing = True
                        agent = st.session_state.agent
                        inv   = get_inventory()
                        with st.spinner("Agent executing plan…"):
                            result = agent.execute_plan(
                                plan           = plan,
                                inventory      = inv.as_dict(),
                                stream_callback = add_message,
                            )
                        st.session_state.executing = False
                        if result.success:
                            add_message("system",
                                f"✅ Assembly complete! "
                                f"Placed {len(result.bricks_placed)} brick(s): "
                                f"{result.bricks_placed}")
                            mem = get_memory()
                            mem.update_context("bricks_placed", result.bricks_placed)
                            mem.update_context("executions",
                                mem.context.get("executions", 0) + 1)
                        else:
                            errs = "; ".join(result.errors[:3])
                            add_message("system", f"❌ Execution failed: {errs}")
                        # Autosave after every execution
                        st.session_state.session_store.save(st.session_state)
                        st.rerun()
                else:
                    st.button("▶ Execute plan", disabled=True,
                              use_container_width=True,
                              help="Plan must be validated first")
            with btn_col2:
                if st.button("🎬 Run demo", use_container_width=True,
                             help="Preview assembly animation before executing"):
                    st.session_state["demo_anim_frame"]   = 0
                    st.session_state["demo_anim_playing"] = True
                    st.rerun()

    # ── Tab 2: Animation ─────────────────────────────────────────
    if tab_anim is not None:
        with tab_anim:
            if plan and plan.arrangement:
                animator = DemoAnimator(plan)
                animator.render_controls()
                anim_container = st.empty()
                animator.render_current_frame(anim_container)
            else:
                st.info("Generate a plan first to see the animation.")

    # ── Workspace reference ───────────────────────────────────────
    with st.expander("Workspace limits (real values)", expanded=False):
        from workspace_constraints import (
            TABLE_X_MIN, TABLE_X_MAX, TABLE_Y_MIN, TABLE_Y_MAX,
            AR4_REACH_RADIUS, ABB_REACH_RADIUS,
            TABLE_MAX_HEIGHT, GRID_STEP, BRICK_LAYER_HEIGHT,
        )
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Table X bounds | {TABLE_X_MIN} → {TABLE_X_MAX} m |
        | Table Y bounds | {TABLE_Y_MIN} → {TABLE_Y_MAX} m |
        | AR4 reach | {AR4_REACH_RADIUS} m |
        | ABB reach | {ABB_REACH_RADIUS} m |
        | Max height | {TABLE_MAX_HEIGHT} m |
        | Grid step | {GRID_STEP} m |
        | Layer height | {BRICK_LAYER_HEIGHT} m |
        | Z-brick | Handover only |
        """)


# ═════════════════════════════════════════════
# STATUS BAR
# ═════════════════════════════════════════════

ros = get_ros_status()
inv = get_inventory()
st.markdown(
    f"""
    <div class="status-bar">
        <span>{ros.status_color()} ROS: {'Connected' if ros.connected else 'Mock'}</span>
        <span>📦 Inventory: {inv.total()} bricks</span>
        <span>🦾 AR4: {ros.ar4_stage}</span>
        <span>🦾 ABB: {ros.abb_stage}</span>
        <span>🔄 Supervisor: {ros.supervisor_state}</span>
    </div>
    """,
    unsafe_allow_html=True,
)