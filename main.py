import os
import json
import cv2

import gymnasium as gym
import mani_skill.envs
import numpy as np

from robot_controller import RobotController

from skills.pick import PickSkill
from skills.place import PlaceSkill

from perception.qwen_vlm import QwenVLM


# ============================================================
# Environment
# ============================================================

env = gym.make(
    "PickCube-v1",
    num_envs=1,
    obs_mode="rgb",
    control_mode="pd_ee_delta_pose",
    render_mode="human",
)

obs, info = env.reset()


# ============================================================
# Components
# ============================================================

robot = RobotController(env)

pick_skill = PickSkill(
    env,
    robot
)

place_skill = PlaceSkill(
    env,
    robot
)

vlm = QwenVLM()


os.makedirs(
    "images",
    exist_ok=True
)


# ============================================================
# Camera
# ============================================================

def get_current_observation():

    action = np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            robot.gripper,
        ],
        dtype=np.float32,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    return obs


def save_camera_image(
    obs,
    path
):

    sensor_data = obs[
        "sensor_data"
    ]

    camera_name = next(
        iter(sensor_data)
    )

    rgb = sensor_data[
        camera_name
    ]["rgb"]

    rgb = (
        rgb[0]
        .detach()
        .cpu()
        .numpy()
    )

    bgr = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        path,
        bgr
    )


# ============================================================
# Tool Router
# ============================================================

def execute_tool(
    decision
):

    function_name = decision.get(
        "function"
    )

    target = decision.get(
        "target"
    )


    # --------------------------------------------------------
    # PICK
    # --------------------------------------------------------

    if function_name == "pick":

        print()
        print(
            f"[Agent] CALL -> "
            f"pick({target})"
        )

        return pick_skill.execute(
            target
        )


    # --------------------------------------------------------
    # PLACE
    # --------------------------------------------------------

    if function_name == "place":

        print()
        print(
            f"[Agent] CALL -> "
            f"place({target})"
        )

        return place_skill.execute(
            target
        )


    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    if function_name == "finish":

        print()
        print(
            "[Agent] CALL -> finish()"
        )

        return {
            "success": True,
            "task_finished": True
        }


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return {
        "success": False,
        "error": (
            f"Unknown tool: "
            f"{function_name}"
        )
    }


# ============================================================
# User Goal
# ============================================================

print()
print("=" * 60)
print("VLM ROBOT AGENT")
print("=" * 60)


instruction = input(
    "\nRobot instruction: "
)


# ============================================================
# Agent Memory
# ============================================================

history = []


# ============================================================
# Agent Loop
# ============================================================

MAX_AGENT_STEPS = 6


for step in range(
    1,
    MAX_AGENT_STEPS + 1
):

    print()
    print()
    print("#" * 60)

    print(
        f"AGENT STEP {step}"
    )

    print("#" * 60)


    # ========================================================
    # OBSERVE
    # ========================================================

    print()
    print(
        "[Agent] Observing world..."
    )


    obs = get_current_observation()


    image_path = (
        f"images/"
        f"agent_step_{step}.png"
    )


    save_camera_image(
        obs,
        image_path
    )


    print(
        "[Agent] Observation saved:",
        image_path
    )


    # ========================================================
    # THINK
    # ========================================================

    print()
    print(
        "[Agent] Asking Qwen..."
    )


    try:

        decision = vlm.decide(
            image_path,
            instruction,
            history
        )

    except Exception as e:

        print()
        print(
            "[Agent] VLM ERROR:"
        )

        print(e)

        break


    print()
    print(
        "[Agent] Decision:"
    )

    print(
        json.dumps(
            decision,
            ensure_ascii=False,
            indent=2
        )
    )


    # ========================================================
    # ACT
    # ========================================================

    result = execute_tool(
        decision
    )


    print()
    print(
        "[Agent] Tool result:"
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )


    # ========================================================
    # MEMORY
    # ========================================================

    history.append(
        {
            "step": step,

            "decision": decision,

            "tool_result": result,
        }
    )


    # ========================================================
    # FINISH
    # ========================================================

    if result.get(
        "task_finished",
        False
    ):

        print()
        print("=" * 60)

        print(
            "AGENT TASK COMPLETE"
        )

        print("=" * 60)

        break


else:

    print()
    print("=" * 60)

    print(
        "MAX AGENT STEPS REACHED"
    )

    print("=" * 60)


# ============================================================
# History
# ============================================================

print()
print()
print("=" * 60)

print(
    "FINAL AGENT HISTORY"
)

print("=" * 60)


print(
    json.dumps(
        history,
        ensure_ascii=False,
        indent=2
    )
)


input(
    "\nPress Enter to exit..."
)


env.close()