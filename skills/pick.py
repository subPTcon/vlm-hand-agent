import time
import numpy as np


class PickSkill:

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
    # Precise XY alignment
    # ========================================================

    def align_xy(self):

        print()
        print("[PickSkill] Aligning XY...")

        XY_THRESHOLD = 0.0015

        for i in range(100):

            tcp = self.get_tcp_position()
            cube = self.get_cube_position()

            dx = cube[0] - tcp[0]
            dy = cube[1] - tcp[1]

            print(
                f"[PickSkill] align "
                f"dx={dx:+.4f} "
                f"dy={dy:+.4f}"
            )

            if (
                abs(dx) < XY_THRESHOLD
                and
                abs(dy) < XY_THRESHOLD
            ):

                print(
                    "[PickSkill] XY aligned."
                )

                return True


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


            if abs(dx) >= abs(dy):

                magnitude = min(
                    0.04,
                    max(
                        0.008,
                        abs(dx) * 2.0
                    )
                )

                action[0] = (
                    magnitude
                    if dx > 0
                    else -magnitude
                )

            else:

                magnitude = min(
                    0.04,
                    max(
                        0.008,
                        abs(dy) * 2.0
                    )
                )

                action[1] = (
                    magnitude
                    if dy > 0
                    else -magnitude
                )


            self.robot._execute(
                action,
                steps=4
            )

            self.robot.stop(2)

            time.sleep(0.03)


        return False


    # ========================================================
    # Descend
    # ========================================================

    def descend(self):

        print()
        print(
            "[PickSkill] Descending..."
        )

        TARGET_Z_OFFSET = 0.003

        Z_THRESHOLD = 0.003


        for i in range(100):

            tcp = self.get_tcp_position()
            cube = self.get_cube_position()

            target_z = (
                cube[2]
                +
                TARGET_Z_OFFSET
            )

            dz = target_z - tcp[2]


            print(
                f"[PickSkill] descend "
                f"tcp_z={tcp[2]:.4f} "
                f"target_z={target_z:.4f} "
                f"dz={dz:+.4f}"
            )


            if abs(dz) < Z_THRESHOLD:

                print(
                    "[PickSkill] "
                    "grasp height reached."
                )

                return True


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


            magnitude = min(
                0.04,
                max(
                    0.006,
                    abs(dz) * 1.5
                )
            )


            action[2] = (
                magnitude
                if dz > 0
                else -magnitude
            )


            self.robot._execute(
                action,
                steps=4
            )

            self.robot.stop(2)

            time.sleep(0.03)


        return False


    # ========================================================
    # Lift
    # ========================================================

    def lift(self):

        print()
        print(
            "[PickSkill] Lifting..."
        )

        initial_z = (
            self.get_cube_position()[2]
        )


        for i in range(20):

            if not self.is_grasping():

                print(
                    "[PickSkill] "
                    "grasp lost."
                )

                return False


            action = np.array(
                [
                    0.0,
                    0.0,
                    0.04,
                    0.0,
                    0.0,
                    0.0,
                    self.robot.gripper,
                ],
                dtype=np.float32,
            )


            self.robot._execute(
                action,
                steps=4
            )

            self.robot.stop(2)


            cube_z = (
                self.get_cube_position()[2]
            )


            print(
                f"[PickSkill] "
                f"cube_z={cube_z:.4f}"
            )


            if (
                cube_z
                >
                initial_z + 0.08
            ):

                print(
                    "[PickSkill] "
                    "Cube lifted."
                )

                return True


        return False


    # ========================================================
    # Public Skill
    # ========================================================

    def execute(self, target):

        print()
        print("=" * 60)

        print(
            f"[PickSkill] "
            f"Executing PICK({target})"
        )

        print("=" * 60)


        # Open gripper

        self.robot.open_gripper()

        self.robot.stop(5)


        # Align

        if not self.align_xy():

            return {
                "success": False,
                "error": "XY alignment failed"
            }


        # Descend

        if not self.descend():

            return {
                "success": False,
                "error": "Descend failed"
            }


        # Close

        print()
        print(
            "[PickSkill] "
            "Closing gripper..."
        )

        self.robot.close_gripper()

        self.robot.stop(10)

        time.sleep(0.3)


        # Real physics verification

        if not self.is_grasping():

            return {
                "success": False,
                "error": "Real grasp verification failed"
            }


        print(
            "[PickSkill] "
            "REAL GRASP CONFIRMED"
        )


        # Lift

        if not self.lift():

            return {
                "success": False,
                "error": "Lift failed"
            }


        return {
            "success": True,
            "target": target
        }