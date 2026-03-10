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
        

