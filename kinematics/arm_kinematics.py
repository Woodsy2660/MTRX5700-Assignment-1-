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
        

    def differential_kinematics(
        self, q: List[float], q_dot: List[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """
        Computes the geometric Jacobian J(q) expressed in the base frame.

        The Jacobian maps joint velocities to end-effector twist:

        Re-using the all_frames function and subsituting into V = J(q) * q_dot

        where V = [v; w] is a 6 vector stacking linear velocity (v) on top of angular velocity (w)

        Parameters:
            q: Joint positions (radians for revolute, meters for prismatic)
            q_dot: Joint velocities (rad/s for revolute, m/s for prismatic). Optional.

        Returns:
            J: Jacobian matrix (6 x n)
            sigma: Singular values from SVD
            v_ee: End-effector velocity (6x1) if q_dot provided, else None
        """

        frames = self.all_frames(q)
        n = self._dh_table.num_joints()

        o_n = frames[-1][:3, 3]  # end effector origin - position coloumn of the last frame

        J = np.zeros((6, n))  # pre allocate jacobian (6 x N), each coloumn will be filled in the loop below

        for i in range(n):
            if i == 0:
                # Frame 0 is the base frame — always the identity.
                z_prev = np.array([0.0, 0.0, 1.0])
                o_prev = np.zeros(3)
            else:
                # Frame i-1 is frames[i-1].
                z_prev = frames[i - 1][:3, 2]      # Third column of the rotation block [:3, :3] is the z-axis.
                o_prev = frames[i - 1][:3, 3]      # Position column [:3, 3] is the origin.

            if self._dh_table._joint_types[i] == "R":
                # Revolute joint — linear velocity from cross product,
                # angular velocity is the joint axis direction z_{i-1}.
                J[:3, i] = np.cross(z_prev, o_n - o_prev)  # linear velocity
                J[3:, i] = z_prev                           # angular velocity

            else:
                # Prismatic joint — pure linear motion along z_{i-1},
                # no angular velocity contribution.
                J[:3, i] = z_prev   # linear velocity
                J[3:, i] = 0.0      # angular velocity


        # np.linalg.svd returns (U, sigma, Vt).
        # We only need sigma, full_matrices=False avoids allocating the full U and Vt for speed.
        _, sigma, _ = np.linalg.svd(J, full_matrices=False)

        if sigma[-1] < 1e-4:
            print(
                f"[WARNING] Near-singular configuration detected. "
                f"sigma_min = {sigma[-1]:.2e}  (threshold 1e-4)"
            )

        # Compute end-effector velocity if joint velocities are provided
        if q_dot is not None:
            q_dot_array = np.array(q_dot)
            v_ee = J @ q_dot_array
        else:
            v_ee = None

        return J, sigma, v_ee

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








    
