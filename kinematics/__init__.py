# kinematics/__init__.py

from .dh_table import DHTable
from .arm_kinematics import ArmKinematics
from .arm_visualiser import plot_robot
from .error_analysis import ErrorAnalyser


def load_robot(filepath: str, t_tool=None):
    """
    Convenience function that wires the full pipeline together.
    Parses the robot definition file, builds a DHTable, and
    returns a ready-to-use ArmKinematics instance, robot name, and joint velocities.
    """
    dh_table, robot_name, q_dot = DHTable.from_file(filepath)
    arm = ArmKinematics(dh_table, t_tool)
    return arm, robot_name, q_dot