# kinematics/__init__.py

from .dh_table import DHTable
from .arm_kinematics import ArmKinematics
from .arm_visualiser import plot_robot


def load_robot(filepath: str, t_tool=None):
    """
    Convenience function that wires the full pipeline together.
    Parses the robot definition file, builds a DHTable, and
    returns a ready-to-use ArmKinematics instance and robot name.
    """
    # Step 1 — parse the file into raw data
    dh_table, robot_name = DHTable.from_file(filepath)

    # Step 2 — pass the DHTable into ArmKinematics
    arm = ArmKinematics(dh_table, t_tool)

    return arm, robot_name