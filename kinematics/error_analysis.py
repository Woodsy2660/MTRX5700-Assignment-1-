# error_analysis.py 
# Loads ground truth TCP poses from a CSV file and compares them against the FK model output to compute position and orientation errors and the full transformation error.

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


class ErrorAnalyser:
    # Takes an ArmKinematics instance and evaluates FK accuracy against ground truth data

    def __init__(self, arm):
        self.arm = arm
        self.ground_truth = []  # from csv
        self.results = []       # validation results


    def load_csv(self, filepath):
        # Load ground truth poses and joint angles from CSV

        df = pd.read_csv(filepath)
        self.ground_truth = []

        for i, row in df.iterrows():
            entry = {
                "config": i + 1,
                "position_gt": np.array([row["X (m)"], row["Y (m)"], row["Z (m)"]]),
                "orientation_gt": np.array([row["RX (rad)"], row["RY (rad)"], row["RZ (rad)"]]),
                "joint_angles": np.radians([
                    row["Base (deg)"],
                    row["Shoulder (deg)"],
                    row["Elbow (deg)"],
                    row["Wrist1 (deg)"],
                    row["Wrist2 (deg)"],
                    row["Wrist3 (deg)"]
                ])
            }
            self.ground_truth.append(entry)

        print(f"Loaded {len(self.ground_truth)} configurations from '{filepath}'")


    def validate_model(self):
        # Run FK for each configuration and compute errors against ground truth

        if not self.ground_truth:
            print("No ground truth data loaded. Call load_csv() first.")
            return

        self.results = []

        for data in self.ground_truth:
            T_model, _, _ = self.arm.forward_kinematics(data["joint_angles"])
            T_gt = self.build_gt_transform(data["position_gt"], data["orientation_gt"])
            errors = self.calculate_errors(T_model, T_gt)

            result = {}
            result.update(data)
            result.update(errors)
            result["T_model"] = T_model
            result["T_gt"] = T_gt
            self.results.append(result)

        return self.results


    def print_results(self):
        # Print errors for each configuration and overall summary like mean and max

        if not self.results:
            print("No results to print. Call validate_model() first.")
            return

        print("\n" + "=" * 65)
        print("  Error Analysis - Per-Configuration Results")
        print("=" * 65)

        for r in self.results:

            e_pos  = r["e_pos"]
            e_rot  = r["e_rot"]
            t_err  = r["t_err"]
            R_err  = r["R_err"]
            T_err  = r["T_err"]
            pdiff  = r["pos_diff"]

            print(f"\n  -- Configuration {r['config']} --")

            # Position error
            print(f"  Position error  : {e_pos:.6f} m")
            print(f"    dX = {pdiff[0]:+.6f} m")
            print(f"    dY = {pdiff[1]:+.6f} m")
            print(f"    dZ = {pdiff[2]:+.6f} m")


            # Orientation error
            rx_e, ry_e, rz_e = self.rotation_matrix_to_axis_angle(R_err)
            print(f"  Orientation error  : {e_rot:.6f} rad")
            print(f"    RX = {rx_e:+.6f} rad")
            print(f"    RY = {ry_e:+.6f} rad")
            print(f"    RZ = {rz_e:+.6f} rad")

            # Full transformation error matrix
            print(f"  Full T_err (T_gt_inv * T_model):")
            for row in T_err:
                print(f"    [ {row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.6f}  {row[3]:10.6f} ]")

        # Summary stats
        all_e_pos = [r["e_pos"]   for r in self.results]
        all_e_rot = [r["e_rot"]   for r in self.results]

        print("\n" + "=" * 65)
        print("  Summary")
        print("=" * 65)
        print(f"  {'Metric':<38} {'Mean':>10}  {'Max':>10}")
        print("  " + "-" * 60)
        print(f"  {'Position error (m)':<38} {np.mean(all_e_pos):>10.6f}  {np.max(all_e_pos):>10.6f}")
        print(f"  {'Orientation error (rad)':<38} {np.mean(all_e_rot):>10.6f}  {np.max(all_e_rot):>10.6f}")
        print("=" * 65 + "\n")



    def build_gt_transform(self, position, orientation):
        # Build a 4x4 homogeneous transform from position vector and axis-angle orientation

        R = self.axis_angle_to_rotation_matrix(orientation[0], orientation[1], orientation[2])
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = position
        return T


    def calculate_errors(self, T_model, T_gt):
        # Calculate position and orientation errors as well as full transformation error between T_model and T_gt. 

        # Position Error (e = || p_model - p_gt || )
        p_model = T_model[0:3, 3]
        p_gt    = T_gt[0:3, 3]
        pos_diff = p_model - p_gt
        e_pos = np.linalg.norm(pos_diff)


        # Full Transformation Error 
        T_err  = np.linalg.inv(T_gt) @ T_model
        t_err  = T_err[0:3, 3]
        R_err  = T_err[0:3, 0:3]
        e_trans = np.linalg.norm(t_err) # translational


        # Orientation error using the formula arccos((Tr(R_err) - 1) / 2)
        cos_angle = (np.trace(R_err) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        e_rot = float(np.arccos(cos_angle))

        return {
            "e_pos":    e_pos,
            "e_translational":  e_trans,
            "e_rot":    e_rot,
            "t_err":    t_err,
            "R_err":    R_err,
            "T_err":    T_err,
            "pos_diff": pos_diff
        }


    def axis_angle_to_rotation_matrix(self, rx, ry, rz):
        # Convert axis-angle vector (rx, ry, rz) to 3x3 rotation matrix
        theta = np.sqrt(rx**2 + ry**2 + rz**2)

        if theta < 1e-9:
            return np.eye(3)

        # Unit axis vector
        kx = rx / theta
        ky = ry / theta
        kz = rz / theta

        c = np.cos(theta)
        s = np.sin(theta)
        t = 1.0 - c

        R = np.array([
            [t*kx*kx + c,    t*kx*ky - s*kz, t*kx*kz + s*ky],
            [t*kx*ky + s*kz, t*ky*ky + c,    t*ky*kz - s*kx],
            [t*kx*kz - s*ky, t*ky*kz + s*kx, t*kz*kz + c   ]
        ])

        return R


    def rotation_matrix_to_axis_angle(self, R):
        # Convert 3x3 rotation matrix to axis-angle vector (rx, ry, rz)
        cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        # If no rotation
        if abs(theta) < 1e-9:
            return 0.0, 0.0, 0.0

        # If there is 180 degree rotation present
        if abs(theta - np.pi) < 1e-6:
            rx = np.sqrt(max((R[0, 0] + 1) / 2, 0.0))
            ry = np.sqrt(max((R[1, 1] + 1) / 2, 0.0))
            rz = np.sqrt(max((R[2, 2] + 1) / 2, 0.0))
            if R[0, 1] < 0:
                ry = -ry
            if R[0, 2] < 0:
                rz = -rz
            return rx * theta, ry * theta, rz * theta

        # General case
        factor = theta / (2.0 * np.sin(theta))
        rx = (R[2, 1] - R[1, 2]) * factor
        ry = (R[0, 2] - R[2, 0]) * factor
        rz = (R[1, 0] - R[0, 1]) * factor

        return rx, ry, rz