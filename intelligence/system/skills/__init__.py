"""
skills/
=======
LangChain tool functions for the robotic assembly skill library.

Each skill maps directly to a real supervisor_node.py action:
  pick()      → ExecuteTask(task_type="PICK")     on ar4_control / abb_control
  place()     → ExecuteTask(task_type="PLACE")    on ar4_control / abb_control
  handover()  → ExecuteTask(task_type="INTERMEDIATE_GIVE") + "INTERMEDIATE_TAKE"
  detect()    → DetectBricks service              on detect_bricks
  home()      → ExecuteTask(task_type="HOME")     on ar4_control / abb_control
  grasp()     → GetGrasp service                  on grasp/get_grasp_point
  status()    → reads live ROS bridge snapshot

Import all tools for the agent:
    from skills import ALL_TOOLS
"""

from skills.pick     import pick_brick
from skills.place    import place_brick
from skills.handover import handover_brick
from skills.detect   import detect_bricks
from skills.home     import home_arm
from skills.grasp    import get_grasp_point
from skills.status   import get_system_status

ALL_TOOLS = [
    pick_brick,
    place_brick,
    handover_brick,
    detect_bricks,
    home_arm,
    get_grasp_point,
    get_system_status,
]

__all__ = ["ALL_TOOLS",
           "pick_brick", "place_brick", "handover_brick",
           "detect_bricks", "home_arm", "get_grasp_point",
           "get_system_status"]