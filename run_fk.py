### RUNNING INSTRUCTIONS ##
## python run_fk.py robots/robotname.txt robots/robotinputname.txt

import sys
import math
import numpy as np
from kinematics import load_robot, plot_robot


def load_joint_inputs(input_file):
    """Load joint angles from an input file.

    Expected format:
        <RobotName>
        <q1,q2,q3,q4,q5,q6>

    Supports math expressions e.g. pi, pi/2, -pi
    """
    MATH_ENV = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}

    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines:
        if line.startswith('<') and line.endswith('>') and ',' in line:
            values = line[1:-1].split(',')
            return np.array([float(eval(v.strip(), {"__builtins__": {}}, MATH_ENV)) for v in values])

    raise ValueError(f"Could not find joint angles in format <q1,...,qn> in '{input_file}'")


def main():
    if len(sys.argv) < 3:
        print("Usage: python run_fk.py <robot_description_file> <input_file>")
        print("  Example: python run_fk.py robots/ur5e.txt robots/ur5einput.txt")
        sys.exit(1)

    robot_file = sys.argv[1]
    input_file = sys.argv[2]

    # Load robot and joint configuration
    arm, robot_name, q_dot_from_file = load_robot(robot_file)
    q = load_joint_inputs(input_file)

    # Use q_dot from robot file if available
    q_dot = q_dot_from_file

    # Print DH table
    arm._dh_table.print_table(robot_name)

    # Compute forward kinematics
    T, R, p = arm.forward_kinematics(q)

    # Get joint information
    num_joints = arm._dh_table.num_joints()
    joint_types = arm._dh_table._joint_types
    joint_types_str = ", ".join(joint_types)

    # Convert joint angles to degrees
    q_deg = np.degrees(q)

    # Print robot information
    print(f"Robot: {robot_name}")
    print(f"Joints: {num_joints} ({joint_types_str})")
    print(f"Joint configuration: q = {q} rad")
    print(f"Joint configuration (deg): {q_deg}")
    print()
    print("=" * 60)
    print("Forward Kinematics Results")
    print("=" * 60)
    print()
    print("End-effector position (m):")
    print(f"  x = {p[0]:.6f}")
    print(f"  y = {p[1]:.6f}")
    print(f"  z = {p[2]:.6f}")
    print()
    print("End-effector rotation matrix:")
    for row in R:
        print(f"  [{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f}]")
    print()
    print("Full homogeneous transform (T_0_n):")
    for row in T:
        print(f"  [{row[0]:10.6f} {row[1]:10.6f} {row[2]:10.6f} {row[3]:12.6f}]")
    print("=" * 60)
    print()

    # Compute differential kinematics (Jacobian)
    J, sigma, v_ee = arm.differential_kinematics(q, q_dot)

    print("=" * 60)
    print("Differential Kinematics Results")
    print("=" * 60)
    print()
    print("Jacobian matrix J (6xn):")
    print("  [Linear velocity contributions]")
    for i in range(3):
        row_str = "  ["
        for j in range(J.shape[1]):
            row_str += f"{J[i, j]:10.6f} "
        row_str += "]"
        print(row_str)
    print("  [Angular velocity contributions]")
    for i in range(3, 6):
        row_str = "  ["
        for j in range(J.shape[1]):
            row_str += f"{J[i, j]:10.6f} "
        row_str += "]"
        print(row_str)
    print()
    print("Singular values (from SVD):")
    sigma_str = "  ["
    for s in sigma:
        sigma_str += f"{s:.6f} "
    sigma_str += "]"
    print(sigma_str)
    print()

    if v_ee is not None:
        print("Joint velocities (rad/s or m/s):")
        q_dot_str = "  ["
        for qd in q_dot:
            q_dot_str += f"{qd:.6f} "
        q_dot_str += "]"
        print(q_dot_str)
        print()
        print("End-effector velocity (m/s and rad/s):")
        print(f"  Linear velocity:")
        print(f"    vx = {v_ee[0]:.6f} m/s")
        print(f"    vy = {v_ee[1]:.6f} m/s")
        print(f"    vz = {v_ee[2]:.6f} m/s")
        print(f"  Angular velocity:")
        print(f"    wx = {v_ee[3]:.6f} rad/s")
        print(f"    wy = {v_ee[4]:.6f} rad/s")
        print(f"    wz = {v_ee[5]:.6f} rad/s")
    else:
        print("No joint velocities provided - end-effector velocity not computed.")
    print("=" * 60)
    print()
    print("Generating 3D visualization...")

    frames = arm.all_frames(q)
    plot_robot(frames, q=q, robot_name=robot_name)


if __name__ == "__main__":
    main()