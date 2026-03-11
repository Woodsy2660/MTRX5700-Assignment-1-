"Creates and Stores DH parameters class and generates homogeneous transfroms"

"""
dh_table.py
-----------
Stores standard DH parameters and generates individual A-matrices.

Column order for dh_params: [a, alpha, d, theta]
Units: meters for a and d, radians for alpha and theta.
"""

from typing import List
import numpy as np


class DHTable:

    def __init__(self, dh_params: np.ndarray, joint_types: List[str]) -> None:
        """Stores DH parameters for a serial manipulator and computes single-joint
        A-matrices using DH formula (Spong Eq)"""
        dh_array = np.asarray(dh_params, dtype=float)



    def num_joints(self) -> int:
        """Number of joints in the table"""
        return self._dh_params.shape[0]


    def get_transform(self, joint_index: int, q_i: float) -> np.ndarray:
        """returns the 4x4 A-matrix forr a single joint given its joint 
        variable q_i, handles this for 'R' (revolute) -> q_i is theta 
        and 'P' (prismatic) -> q_i is d"""

        
        

