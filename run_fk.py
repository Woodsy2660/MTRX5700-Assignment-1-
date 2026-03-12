import numpy as np
from kinematics import load_robot, plot_robot

# One call wires parser -> DHTable -> ArmKinematics
arm, robot_name = load_robot("robots/ur5e.txt")

q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 0.0])

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
plot_robot(frames, q=q)
