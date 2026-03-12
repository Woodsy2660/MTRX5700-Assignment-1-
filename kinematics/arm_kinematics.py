"""
Computes forward kinematics, Jacobian matrix determining the linear and angular
velocity contribution of each joint, analyses singular configurations using Singular
Value Decomposition (SVD) by examining rank deficiency and presence of very small singular
values, Numerical Inverse Kinematics (IK) implemented here.

This file composes 'dh_table.py'
"""

from typing import List, Tuple
import numpy as np

from .dh_table import DHTable

class ArmKinematics:
    """
    Computes forward kinematics for a serial manipulator

    Takes a DHTable instance at construction, exposes methods to compute
    the full kinematic chain.

    Parameters are: dh_table, t_tool
    """

    def __init__(
        self,
        dh_table: DHTable,
        t_tool: np.ndarray = None,
    ) -> None:

        self._dh_table = dh_table  # Store the DHTable instance

        if t_tool is None:
            self.t_tool = np.eye(4)  # Store tool transform, if none provided use the 4x4 identity
        else:
            self.t_tool = np.asarray(t_tool, dtype=float)
        
    def forward_kinematics(
        self, q: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns T (homogeneous transform from base to end-effector),
        R (rotation matrix (top-left block of T)), p (position vector (top-right column of T))
        """

        frames = self.all_frames(q)  # Calls all_frames and returns a list of cumulative transforms 0T1, 0T2, ..., 0TN

        T = np.dot(frames[-1], self.t_tool)  # Apply the tool transform to last link frame

        R = T[:3, :3]  # Slices rotation matrix top left 3x3 block of T
        p = T[:3, 3]  # Slices position vector top right 3 values

        return T, R, p


    def all_frames(self, q: List[float]) -> List[np.ndarray]:
        """
        Compute all cumulative transforms from base to each joint.
        Returns a list of 4x4 homogeneous transforms.
        """

        T = np.eye(4)  # Creates identity matrix, base frame

        frames = []  # Create empty list to append to

        for i in range(self._dh_table.num_joints()):  # Iterate over number of joints

            A_i = self._dh_table.get_transform(i, q[i])  # Each row is read in DH-table building out each A-matrix

            T = T @ A_i  # Multiply the running transform by A_i

            frames.append(T.copy())  # Append a copy of the current cumulative transform

        return frames








    
