"""
Main entry point for Forward Kinematics visualization.

Usage:
    python run_fk.py <robot_file.txt>                           # Zero configuration
    python run_fk.py <robot_file.txt> 0 1.57 0 0 0 0           # With joint angles (radians)
    python run_fk.py robots/ur5e.txt 0 -1.57 0 -1.57 0 0      # Example with UR5e

This script:
1. Parses a robot definition file (e.g., robots/ur5e.txt)
2. Constructs the DH table
3. Computes forward kinematics for given joint configuration
4. Visualizes the robot in 3D with coordinate frames
"""

import sys
import numpy as np
from kinematics.dh_table import DHTable, parse_robot_file
from kinematics.arm_kinematics import ArmKinematics
from kinematics.arm_visualiser import plot_robot


def main():
    # Check if robot file is provided
    if len(sys.argv) < 2:
        print("Error: Robot definition file required.")
        print("\nUsage:")
        print("  python run_fk.py <robot_file.txt>")
        print("  python run_fk.py <robot_file.txt> q1 q2 q3 ... qn  (radians)")
        print("\nExample:")
        print("  python run_fk.py robots/ur5e.txt")
        print("  python run_fk.py robots/ur5e.txt 0 -1.57 0 -1.57 0 0")
        sys.exit(1)

    robot_file = sys.argv[1]

    # Parse the robot definition file
    try:
        name, joint_types, dh_params = parse_robot_file(robot_file)
        print(f"\nRobot: {name}")
        print(f"Joints: {len(joint_types)} ({', '.join(joint_types)})")
    except FileNotFoundError:
        print(f"Error: File '{robot_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing robot file: {e}")
        sys.exit(1)

    # Create ArmKinematics instance
    arm = ArmKinematics(dh_params, joint_types)
    n_joints = len(joint_types)

    # Parse joint angles if provided
    if len(sys.argv) == n_joints + 2:
        # User provided joint angles
        try:
            q = np.array([float(x) for x in sys.argv[2:]])
            print(f"Joint configuration: q = {q} rad")
            print(f"Joint configuration (deg): {np.degrees(q)}")
        except ValueError:
            print("Error: All joint angles must be numeric values in radians.")
            print(f"Usage: python run_fk.py {robot_file} q1 q2 ... q{n_joints}")
            sys.exit(1)
    elif len(sys.argv) == 2:
        # Default to zero configuration
        q = np.zeros(n_joints)
        print(f"No joint angles provided - using zero configuration.")
        print(f"Usage: python run_fk.py {robot_file} q1 q2 ... q{n_joints}  (radians)")
    else:
        print(f"Error: Expected {n_joints} joint angles, got {len(sys.argv) - 2}")
        print(f"Usage: python run_fk.py {robot_file} q1 q2 ... q{n_joints}")
        sys.exit(1)

    # Compute forward kinematics
    T, R, p = arm.forward_kinematics(q)

    print("\n" + "="*60)
    print("Forward Kinematics Results")
    print("="*60)
    print(f"\nEnd-effector position (m):")
    print(f"  x = {p[0]:.6f}")
    print(f"  y = {p[1]:.6f}")
    print(f"  z = {p[2]:.6f}")

    print(f"\nEnd-effector rotation matrix:")
    for row in R:
        print(f"  [{row[0]:9.6f}  {row[1]:9.6f}  {row[2]:9.6f}]")

    print(f"\nFull homogeneous transform (T_0_n):")
    for row in T:
        print(f"  [{row[0]:9.6f}  {row[1]:9.6f}  {row[2]:9.6f}  {row[3]:9.6f}]")
    print("="*60 + "\n")

    # Visualize the robot
    print("Generating 3D visualization...")
    plot_robot(arm, q, robot_name=name, show=True)


if __name__ == '__main__':
    main()
