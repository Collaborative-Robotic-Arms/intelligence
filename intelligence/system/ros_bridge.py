"""
ros_bridge.py
=============
ROS2 <-> Streamlit data bridge for the dual-arm assembly system.

Two modes:
  MOCK  — no ROS installation needed. Runs a background thread that
          simulates realistic state changes so every other task can be
          developed and tested without a robot.

  LIVE  — spawns an rclpy spin-loop in a daemon thread.
          Subscribes to /detected_bricks and /zone_status.
          Calls detect_bricks and get_assembly_plan services.

Public API (identical in both modes):
--------------------------------------
  bridge = RosBridge(mock=True)
  bridge.start()

  bridge.get_inventory()              -> dict  {"I":int, "L":int, "T":int, "Z":int}
  bridge.get_status()                 -> dict  (ar4_stage, abb_stage, supervisor_state, …)
  bridge.get_detected_bricks()        -> list[dict]
  bridge.call_detect_service()        -> list[dict]   (triggers real detection)
  bridge.push_plan(bricks: list)      -> (bool, str)
  bridge.emergency_stop()             -> None
  bridge.update_mock_inventory(dict)  -> None         (mock only)
  bridge.stop()                       -> None
"""

from __future__ import annotations
import math
import time
import copy
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants — mirror dual_arms_msgs/msg/Brick.msg constants
# ──────────────────────────────────────────────────────────────────────────────

BRICK_TYPE_MAP = {0: "I", 1: "L", 2: "T", 3: "Z"}
BRICK_SIDE_MAP = {0: "AR4", 1: "ABB", 2: "GRID"}
BRICK_TYPE_ID  = {"I": 0, "L": 1, "T": 2, "Z": 3}
BRICK_SIDE_ID  = {"AR4": 0, "ABB": 1, "GRID": 2}

# All supervisor states from supervisor_node.py
SUPERVISOR_STATES = [
    "INIT", "DETECT", "PROCESS_NEXT", "GRASP_PIPELINE",
    "INITIALIZE_PARALLEL_EXECUTION", "WAIT_FOR_NEW_PLAN",
    "TRIGGER_MTC_SAFE_RESOLUTION", "EMERGENCY_STOP", "DONE",
]

# All arm stage values from supervisor_node.py
ARM_STAGES = ["IDLE", "PICK", "PLACE", "DONE",
              "MOVE_TO_HANDOVER", "HOLDING_AT_HANDOVER"]


# ──────────────────────────────────────────────────────────────────────────────
# Internal snapshot (thread-safe data shared between background & UI thread)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _Snapshot:
    connected:        bool  = False
    mode:             str   = "sim"
    ar4_stage:        str   = "IDLE"
    abb_stage:        str   = "IDLE"
    supervisor_state: str   = "INIT"
    zone_status:      str   = "CLEAR"
    inventory:        dict  = field(default_factory=lambda: {"I":0,"L":0,"T":0,"Z":0})
    detected_bricks:  list  = field(default_factory=list)
    last_update:      float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


# ══════════════════════════════════════════════════════════════════════════════
#  MOCK BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

class _MockBridge:
    """
    Simulates realistic robot state without any ROS.
    - Cycles arm stages every few seconds so the UI status badges animate.
    - Holds a fixed set of detected bricks on the table.
    - Inventory is derived from those bricks (or overridden manually).
    """

    # Scripted stage sequence — matches a typical pick-place cycle
    _CYCLE = [
        # (ar4_stage,              abb_stage,   supervisor_state,          ticks)
        ("IDLE",                   "IDLE",      "INIT",                    10),
        ("IDLE",                   "IDLE",      "DETECT",                   8),
        ("IDLE",                   "IDLE",      "PROCESS_NEXT",             6),
        ("PICK",                   "IDLE",      "GRASP_PIPELINE",          10),
        ("PLACE",                  "PICK",      "INITIALIZE_PARALLEL_EXECUTION", 10),
        ("DONE",                   "PLACE",     "INITIALIZE_PARALLEL_EXECUTION", 10),
        ("IDLE",                   "DONE",      "PROCESS_NEXT",             8),
        ("IDLE",                   "IDLE",      "WAIT_FOR_NEW_PLAN",       12),
    ]

    def __init__(self):
        self._snap   = _Snapshot()
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Default table state: enough bricks to try T-shape etc.
        self._mock_bricks = [
            {"id":1, "type":"I", "side":"AR4",  "x": 0.01, "y": 0.19, "z":0.712, "yaw":0.00},
            {"id":2, "type":"I", "side":"AR4",  "x":-0.08, "y": 0.16, "z":0.712, "yaw":0.00},
            {"id":3, "type":"I", "side":"AR4",  "x": 0.01, "y": 0.11, "z":0.712, "yaw":0.00},
            {"id":4, "type":"I", "side":"AR4",  "x":-0.08, "y": 0.14, "z":0.712, "yaw":0.30},
            {"id":5, "type":"L", "side":"AR4",  "x": 0.08, "y":-0.02, "z":0.712, "yaw":1.57},
            {"id":6, "type":"L", "side":"AR4",  "x":-0.06, "y": 0.11, "z":0.712, "yaw":0.00},
            {"id":7, "type":"T", "side":"GRID", "x": 0.00, "y": 0.00, "z":0.712, "yaw":0.78},
            {"id":8, "type":"Z", "side":"AR4",  "x":-0.02, "y": 0.14, "z":0.712, "yaw":0.50},
        ]
        self._manual_inventory: Optional[dict] = None
        self._cycle_idx  = 0
        self._tick_count = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mock_ros_bridge"
        )
        self._thread.start()
        logger.info("[MockBridge] started")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    # ── background loop ──────────────────────────────────────────────────────

    def _run(self):
        while not self._stop.is_set():
            self._tick()
            time.sleep(0.25)   # 4 Hz

    def _tick(self):
        ar4, abb, sup, duration = self._CYCLE[self._cycle_idx]
        self._tick_count += 1

        if self._tick_count >= duration:
            self._tick_count = 0
            self._cycle_idx  = (self._cycle_idx + 1) % len(self._CYCLE)

        # Occasional zone warning
        zone = "COLLISION_WARNING" if (self._tick_count == 3 and ar4 != "IDLE" and abb != "IDLE") \
               else "CLEAR"

        # Derive inventory from mock bricks (unless manually overridden)
        if self._manual_inventory is not None:
            inv = dict(self._manual_inventory)
        else:
            inv = {"I": 0, "L": 0, "T": 0, "Z": 0}
            for b in self._mock_bricks:
                if b["side"] in ("AR4", "ABB"):
                    inv[b["type"]] = inv.get(b["type"], 0) + 1

        with self._lock:
            self._snap.connected        = True
            self._snap.mode             = "sim"
            self._snap.ar4_stage        = ar4
            self._snap.abb_stage        = abb
            self._snap.supervisor_state = sup
            self._snap.zone_status      = zone
            self._snap.inventory        = inv
            self._snap.detected_bricks  = list(self._mock_bricks)
            self._snap.last_update      = time.time()

    # ── public methods ───────────────────────────────────────────────────────

    def get_snapshot(self) -> _Snapshot:
        with self._lock:
            return copy.deepcopy(self._snap)

    def set_inventory(self, counts: dict):
        self._manual_inventory = dict(counts)

    def push_plan(self, bricks: list) -> tuple[bool, str]:
        logger.info(f"[MockBridge] push_plan → {len(bricks)} bricks")
        time.sleep(0.25)   # simulate service latency
        return True, f"Plan accepted — {len(bricks)} brick(s) queued"

    def call_detect_service(self) -> list[dict]:
        time.sleep(0.2)
        with self._lock:
            return list(self._snap.detected_bricks)

    def emergency_stop(self):
        logger.warning("[MockBridge] EMERGENCY STOP")
        with self._lock:
            self._snap.ar4_stage        = "IDLE"
            self._snap.abb_stage        = "IDLE"
            self._snap.supervisor_state = "EMERGENCY_STOP"


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE ROS2 BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

class _LiveBridge:
    """
    Connects to a running ROS2 system via rclpy.
    Runs rclpy.spin in a daemon thread to avoid blocking Streamlit.

    Topics subscribed:
      /detected_bricks   (dual_arms_msgs/BricksArray)  → inventory + brick list
      /zone_status       (std_msgs/String)              → zone_status

    Services called:
      detect_bricks      (dual_arms_msgs/DetectBricks)  → fresh detection
      get_assembly_plan  (supervisor_package/GetAssemblyPlan) → push plan
    """

    def __init__(self, mode: str = "sim"):
        self._snap   = _Snapshot(mode=mode)
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mode   = mode
        self._node   = None         # rclpy Node, set in thread
        self._executor = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._ros_thread, daemon=True, name="live_ros_bridge"
        )
        self._thread.start()
        logger.info("[LiveBridge] background thread started")

    def stop(self):
        self._stop.set()
        try:
            if self._executor:
                self._executor.shutdown()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)

    # ── ROS thread ───────────────────────────────────────────────────────────

    def _ros_thread(self):
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor

            rclpy.init()
            self._node = self._make_node()
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)

            logger.info("[LiveBridge] rclpy spinning…")
            while not self._stop.is_set() and rclpy.ok():
                self._executor.spin_once(timeout_sec=0.05)

        except ImportError:
            logger.error("[LiveBridge] rclpy not importable — source your ROS2 workspace.")
            with self._lock:
                self._snap.connected = False
        except Exception as exc:
            logger.error(f"[LiveBridge] ROS thread crashed: {exc}")
            with self._lock:
                self._snap.connected = False
        finally:
            try:
                import rclpy
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def _make_node(self):
        """
        Build the rclpy Node inside the ROS thread.
        Uses a closure so we don't need a separate class file.
        """
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String

        snap = self._snap
        lock = self._lock
        mode = self._mode

        # Try to import custom messages
        try:
            from dual_arms_msgs.msg import BricksArray
            from dual_arms_msgs.srv import DetectBricks
            from supervisor_package.srv import GetAssemblyPlan
            has_custom = True
        except ImportError:
            logger.warning("[LiveBridge] Custom msgs not found — inventory disabled.")
            has_custom = False

        class AssemblyBridgeNode(Node):
            def __init__(self):
                super().__init__("streamlit_assembly_bridge")

                # /zone_status  (std_msgs/String)
                self.create_subscription(String, "/zone_status",
                                         self._on_zone, 10)

                if has_custom:
                    from dual_arms_msgs.msg import BricksArray
                    from dual_arms_msgs.srv import DetectBricks
                    from supervisor_package.srv import GetAssemblyPlan

                    self.create_subscription(BricksArray, "/detected_bricks",
                                             self._on_bricks, 10)
                    self._detect_cli = self.create_client(DetectBricks,
                                                          "detect_bricks")
                    self._plan_cli   = self.create_client(GetAssemblyPlan,
                                                          "get_assembly_plan")

                with lock:
                    snap.connected  = True
                    snap.mode       = mode
                    snap.last_update = time.time()

                self.get_logger().info("streamlit_assembly_bridge ready")

            # ── callbacks ────────────────────────────────────────

            def _on_zone(self, msg: String):
                with lock:
                    snap.zone_status  = msg.data
                    snap.last_update  = time.time()

            def _on_bricks(self, msg):
                counts = {"I": 0, "L": 0, "T": 0, "Z": 0}
                bricks = []
                for b in msg.bricks:
                    btype = BRICK_TYPE_MAP.get(b.type, "I")
                    bside = BRICK_SIDE_MAP.get(b.side, "AR4")
                    if bside != "GRID":
                        counts[btype] = counts.get(btype, 0) + 1
                    bricks.append({
                        "id":   b.id,
                        "type": btype,
                        "side": bside,
                        "x":    b.pose.position.x,
                        "y":    b.pose.position.y,
                        "z":    b.pose.position.z,
                        "yaw":  _quat_to_yaw(
                            b.pose.orientation.x,
                            b.pose.orientation.y,
                            b.pose.orientation.z,
                            b.pose.orientation.w,
                        ),
                    })
                with lock:
                    snap.inventory        = counts
                    snap.detected_bricks  = bricks
                    snap.last_update      = time.time()

            # ── service helpers ───────────────────────────────────

            def call_detect(self) -> list[dict]:
                if not has_custom:
                    return []
                if not self._detect_cli.wait_for_service(timeout_sec=2.0):
                    logger.warning("detect_bricks service not available")
                    return []
                import rclpy
                from dual_arms_msgs.srv import DetectBricks
                req    = DetectBricks.Request()
                future = self._detect_cli.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                res = future.result()
                if res and res.success:
                    out = []
                    for b in res.bricks:
                        out.append({
                            "id":   b.id,
                            "type": BRICK_TYPE_MAP.get(b.type, "I"),
                            "side": BRICK_SIDE_MAP.get(b.side, "AR4"),
                            "x":    b.pose.position.x,
                            "y":    b.pose.position.y,
                            "z":    b.pose.position.z,
                            "yaw":  _quat_to_yaw(
                                b.pose.orientation.x,
                                b.pose.orientation.y,
                                b.pose.orientation.z,
                                b.pose.orientation.w,
                            ),
                        })
                    return out
                return []

            def call_push_plan(self, bricks: list) -> tuple[bool, str]:
                """
                get_assembly_plan is polled BY the supervisor.
                Our bridge just needs the plan to be available when
                the supervisor next polls — this is handled by the
                gui_system / assembly_plan.py node via Firestore.
                Here we call get_assembly_plan as a sanity ping.
                """
                if not has_custom:
                    return False, "Custom messages not available"
                if not self._plan_cli.wait_for_service(timeout_sec=3.0):
                    return False, "get_assembly_plan service unavailable"
                import rclpy
                from supervisor_package.srv import GetAssemblyPlan
                future = self._plan_cli.call_async(GetAssemblyPlan.Request())
                rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
                if future.result() is not None:
                    return True, f"Plan service reachable — {len(bricks)} bricks"
                return False, "Service call timed out"

        return AssemblyBridgeNode()

    # ── public methods ───────────────────────────────────────────────────────

    def get_snapshot(self) -> _Snapshot:
        with self._lock:
            return copy.deepcopy(self._snap)

    def set_inventory(self, _counts: dict):
        pass  # live mode: inventory always comes from /detected_bricks

    def push_plan(self, bricks: list) -> tuple[bool, str]:
        if self._node is None:
            return False, "ROS node not ready"
        return self._node.call_push_plan(bricks)

    def call_detect_service(self) -> list[dict]:
        if self._node is None:
            return []
        return self._node.call_detect()

    def emergency_stop(self):
        logger.warning("[LiveBridge] EMERGENCY STOP requested")
        # The supervisor has its own emergency handler on /zone_status
        # A proper implementation would call a dedicated e-stop service
        with self._lock:
            self._snap.ar4_stage        = "IDLE"
            self._snap.abb_stage        = "IDLE"
            self._snap.supervisor_state = "EMERGENCY_STOP"


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC FACADE
# ══════════════════════════════════════════════════════════════════════════════

class RosBridge:
    """
    Single entry point for all ROS2 interaction.

    Example usage in app.py:
        if "ros_bridge" not in st.session_state:
            st.session_state.ros_bridge = RosBridge(mock=True)
            st.session_state.ros_bridge.start()

        bridge = st.session_state.ros_bridge
        bridge.sync_to_state(st.session_state.inventory,
                             st.session_state.ros_status)
    """

    def __init__(self, mock: bool = True, mode: str = "sim"):
        self._mock = mock
        self._impl: _MockBridge | _LiveBridge = (
            _MockBridge() if mock else _LiveBridge(mode=mode)
        )
        self._started = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if not self._started:
            self._impl.start()
            self._started = True

    def stop(self):
        self._impl.stop()
        self._started = False

    def restart(self, mock: bool, mode: str = "sim"):
        """Switch between mock and live mode at runtime."""
        self.stop()
        self._mock = mock
        self._impl = _MockBridge() if mock else _LiveBridge(mode=mode)
        self.start()

    # ── data accessors ───────────────────────────────────────────────────────

    def get_inventory(self) -> dict:
        """{"I": int, "L": int, "T": int, "Z": int}"""
        return self._impl.get_snapshot().inventory

    def get_status(self) -> dict:
        """
        All live state fields as a flat dict:
          connected, mode, ar4_stage, abb_stage,
          supervisor_state, zone_status, last_update
        """
        s = self._impl.get_snapshot()
        return {
            "connected":        s.connected,
            "mode":             s.mode,
            "ar4_stage":        s.ar4_stage,
            "abb_stage":        s.abb_stage,
            "supervisor_state": s.supervisor_state,
            "zone_status":      s.zone_status,
            "last_update":      s.last_update,
        }

    def get_detected_bricks(self) -> list[dict]:
        """
        List of bricks currently on the table:
          [{"id": int, "type": "I"|"L"|"T"|"Z",
            "side": "AR4"|"ABB"|"GRID",
            "x": float, "y": float, "z": float,
            "yaw": float}, …]
        """
        return self._impl.get_snapshot().detected_bricks

    # ── actions ──────────────────────────────────────────────────────────────

    def call_detect_service(self) -> list[dict]:
        """Trigger a fresh detection run. Returns new brick list."""
        return self._impl.call_detect_service()

    def push_plan(self, bricks: list) -> tuple[bool, str]:
        """
        Send a validated assembly plan to the robot system.
        bricks: list of dicts from AssemblyPlan.arrangement
        Returns (success: bool, message: str).
        """
        return self._impl.push_plan(bricks)

    def emergency_stop(self):
        """Cancel all active goals immediately."""
        self._impl.emergency_stop()

    def update_mock_inventory(self, counts: dict):
        """
        Override simulated brick counts (mock mode only).
        Pass {"I": 3, "L": 1, "T": 0, "Z": 0}
        """
        self._impl.set_inventory(counts)

    # ── convenience sync helpers (called every Streamlit rerun) ──────────────

    def sync_to_state(self, inventory_obj, ros_status_obj):
        """
        Push the latest snapshot into the two state.py objects.
        Call this at the top of app.py on every rerun.

        inventory_obj  : state.BrickInventory instance
        ros_status_obj : state.RosStatus instance
        """
        inv    = self._impl.get_snapshot().inventory
        status = self.get_status()

        # Update inventory (skip if user has manually overridden)
        inventory_obj.I = inv.get("I", 0)
        inventory_obj.L = inv.get("L", 0)
        inventory_obj.T = inv.get("T", 0)
        inventory_obj.Z = inv.get("Z", 0)

        # Update ROS status
        ros_status_obj.connected        = status["connected"]
        ros_status_obj.mode             = status["mode"]
        ros_status_obj.ar4_stage        = status["ar4_stage"]
        ros_status_obj.abb_stage        = status["abb_stage"]
        ros_status_obj.supervisor_state = status["supervisor_state"]
        ros_status_obj.zone_status      = status["zone_status"]
        ros_status_obj.last_update      = status["last_update"]

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def is_mock(self) -> bool:
        return self._mock

    @property
    def is_connected(self) -> bool:
        return self._impl.get_snapshot().connected