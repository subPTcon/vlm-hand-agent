import time
import numpy as np


class PlaceSkill:

    def __init__(self, env, robot):

        self.env = env
        self.robot = robot
        self.u = env.unwrapped


    # ========================================================
    # State
    # ========================================================

    def get_tcp_position(self):

        return (
            self.u.agent.tcp.pose.p[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )


    def get_cube_position(self):

        return (
            self.u.cube.pose.p[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )


    def get_goal_position(self):

        return (
            self.u.goal_site.pose.p[0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )


    def is_grasping(self):

        result = self.u.agent.is_grasping(
            self.u.cube
        )

        return bool(
            result[0]
            .detach()
            .cpu()
            .item()
        )


    # ========================================================
    # Move cube toward goal
    # ========================================================

    def move_to_goal(self):

        print()
        print("[PlaceSkill] Moving cube to goal...")

        # goal_thresh = 0.025
        # 我们控制得更精确一点
        POSITION_THRESHOLD = 0.012

        for i in range(200):

            # ------------------------------------------------
            # Make sure object is still held
            # ------------------------------------------------

            if not self.is_grasping():

                print(
                    "[PlaceSkill] ERROR: grasp lost."
                )

                return False


            cube = self.get_cube_position()
            goal = self.get_goal_position()

            error = goal - cube

            dx = error[0]
            dy = error[1]
            dz = error[2]

            distance = np.linalg.norm(error)


            print(
                f"[PlaceSkill] "
                f"dx={dx:+.4f} "
                f"dy={dy:+.4f} "
                f"dz={dz:+.4f} "
                f"distance={distance:.4f}"
            )


            # ------------------------------------------------
            # Goal reached
            # ------------------------------------------------

            if distance < POSITION_THRESHOLD:

                print()
                print(
                    "[PlaceSkill] GOAL REACHED."
                )

                return True


            # ------------------------------------------------
            # Build action
            # ------------------------------------------------

            action = np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    self.robot.gripper,
                ],
                dtype=np.float32,
            )


            # ------------------------------------------------
            # Move largest-error axis first
            # ------------------------------------------------

            axis = np.argmax(
                np.abs(error)
            )


            axis_error = error[axis]


            magnitude = min(
                0.04,
                max(
                    0.008,
                    abs(axis_error) * 2.0
                )
            )


            action[axis] = (
                magnitude
                if axis_error > 0
                else -magnitude
            )


            self.robot._execute(
                action,
                steps=4
            )

            self.robot.stop(2)

            time.sleep(0.03)


        print(
            "[PlaceSkill] "
            "Failed to reach goal."
        )

        return False


    # ========================================================
    # Verify
    # ========================================================

    def verify_goal(self):

        cube = self.get_cube_position()
        goal = self.get_goal_position()

        distance = np.linalg.norm(
            cube - goal
        )


        print()
        print(
            "[PlaceSkill] Cube:",
            np.round(cube, 4)
        )

        print(
            "[PlaceSkill] Goal:",
            np.round(goal, 4)
        )

        print(
            "[PlaceSkill] Distance:",
            round(float(distance), 4)
        )

        print(
            "[PlaceSkill] Goal threshold:",
            self.u.goal_thresh
        )


        return (
            distance
            <
            self.u.goal_thresh
        )


    # ========================================================
    # Public API
    # ========================================================

    def execute(
        self,
        target="green goal"
    ):

        print()
        print("=" * 60)

        print(
            f"[PlaceSkill] "
            f"Executing PLACE({target})"
        )

        print("=" * 60)


        # ----------------------------------------------------
        # Must already hold cube
        # ----------------------------------------------------

        if not self.is_grasping():

            return {
                "success": False,
                "error": (
                    "Robot is not holding the cube"
                )
            }


        # ----------------------------------------------------
        # Move
        # ----------------------------------------------------

        if not self.move_to_goal():

            return {
                "success": False,
                "error": (
                    "Failed to move cube to goal"
                )
            }


        # ----------------------------------------------------
        # Verify while still holding
        # ----------------------------------------------------

        if not self.verify_goal():

            return {
                "success": False,
                "error": (
                    "Cube did not enter goal region"
                )
            }


        print()
        print(
            "[PlaceSkill] "
            "PLACE VERIFIED."
        )


        # ----------------------------------------------------
        # Release
        # ----------------------------------------------------

        print(
            "[PlaceSkill] "
            "Opening gripper..."
        )

        self.robot.open_gripper()

        self.robot.stop(10)

        time.sleep(0.2)


        return {
            "success": True,
            "target": target
        }