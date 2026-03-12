### RUNNING INSRUCTIONS##

## python run_fk.py robot/robotname.txt robot/robotinputname.txt

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
    arm, robot_name = load_robot(robot_file)
    q = load_joint_inputs(input_file)

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
    print("Generating 3D visualization...")

    frames = arm.all_frames(q)
    plot_robot(frames, q=q, robot_name=robot_name)


if __name__ == "__main__":
    main()