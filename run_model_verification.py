"""
run_model_verification.py  —  Validates FK model accuracy against ground-truth data of pose from csv file.

Commandline input:
    python run_model_verification.py <robot_file> <ground_truth_csv>

Example:
    python run_model_verification.py robots/ur5e.txt robot_positions_data.csv

The ground-truth CSV must contain these columns:
    X (m), Y (m), Z (m), RX (rad), RY (rad), RZ (rad), Base (deg), Shoulder (deg), Elbow (deg), Wrist1 (deg), Wrist2 (deg), Wrist3 (deg)

"""

import sys
from kinematics import load_robot, ErrorAnalyser


def main():
    if len(sys.argv) != 3:
        print("How to test: python run_model_verification.py <robot_file> <ground_truth_csv>")
        sys.exit(1)

    robot_file = sys.argv[1]
    csv_file   = sys.argv[2]

    # Load robot
    arm, robot_name, _ = load_robot(robot_file)
    print(f"\nRobot loaded : {robot_name}")
    arm._dh_table.print_table(robot_name)

    # Run error analysis
    analyser = ErrorAnalyser(arm)
    analyser.load_csv(csv_file)
    analyser.validate_model()
    analyser.print_results()


if __name__ == "__main__":
    main()