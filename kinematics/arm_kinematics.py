"""
Computes forward kinematics, Jacobian matrix determining the linear and angular
velocity contribution of each joint, analyses singular configurations using Singular
Value Decomposition (SVD) by examining rank deficiency and presence of very small singular
values, Numerical Inverse Kinematics (IK) implemented here.

This file composes 'dh_table.py'
"""

from typing import List, Tuple
import numpy as np
import sympy as sp

from .dh_table import DHTable


class ArmKinematics:
    """
    Computes forward kinematics for a serial manipulator

    Takes DH parameters and joint types at construction, builds
    a DHTable internally, exposes methods to compute the full kinematic chain.

    Parameters are: dh_params, joint_types, t_tool
    """

    def __init__(
        self,
        dh_params: np.ndarray,
        joint_types: List[str],
        t_tool: np.ndarray = None,
    ) -> None:

        self._dh_table = DHTable(dh_params, joint_types)  # call dh_table class in composition

        if t_tool is None:
            self.t_tool = np.eye(4)  # store tool transform, if none provided use the 4x4 identity
        else:
            self.t_tool = np.asarray(t_tool, dtype=float)

    def forward_kinematics(
        self, q: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns T (homogeneous transform from base to end-effector),
        R (rotation matrix - top-left 3x3 block of T),
        p (position vector - top-right column of T)
        """
        frames = self.all_frames(q)  # get all frame transforms

        # The last frame is the end-effector position
        T_base_to_ee = frames[-1] @ self.t_tool  # Apply tool transform

        R = T_base_to_ee[:3, :3]  # Extract rotation matrix
        p = T_base_to_ee[:3, 3]   # Extract position vector

        return T_base_to_ee, R, p

    def all_frames(self, q: List[float]) -> List[np.ndarray]:
        """
        Compute all cumulative frame transforms from base to each joint.

        Parameters
        ----------
        q : List of joint values (angles for revolute, distances for prismatic)

        Returns
        -------
        frames : List of 4x4 numpy arrays, T_0_1, T_0_2, ..., T_0_n
        """
        T = np.eye(4)  # Initialize as base frame (identity)
        frames = []

        # Iterate over each joint
        for i in range(self._dh_table.num_joints()):
            # Get the symbolic A matrix for this joint
            A_sym = self._dh_table.A_matrices[i]

            # Substitute the joint variable with actual value
            q_sym = sp.Symbol(f"q{i+1}")
            A_numeric = np.array(A_sym.subs(q_sym, q[i]), dtype=float)

            # Accumulate the transform
            T = T @ A_numeric
            frames.append(T.copy())

        return frames
