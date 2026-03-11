"Computes forward kinematics, Jacobian matrix determining the linear and angular 
"velocity contribution of each joint, analyses singular configurations using Singular
"Value Decomposition (SVD) by examining rank deficiency and presence of very small singular
"vlaues, Numerical Inverse Kinematics (IK) implemented here" 

"this file composes 'dh_table.py'"


from typing import List, Tuple 
import numpy as np

frrom dh_table import DHTable """import DHTable class"""

class ArmKinematics:
    """
    Computes forward kinematics for a serial manipulator

    Takes DH parameters and joint typees at constrruction, builds
    a DHTable internally, exposes methods to compute the full kinematic chain.

    Parameters are: dh_params, joint_types, t_tool
    """
    
    def __init__(
        self,
        dh_params: np.ndarray,
        joint_types: List[str],
        t_tool: np.ndarry = None,
    ) -> None:

        self._dh_table = DHTable(dh_params, joint_types) """call dh_table class in composition"""


        if t_tool is None:
            self.t_tool = np.eye(4)  """store tool transform, if none provided use the 4x4 identity, meaning end effector will be last link frame"""
        else 
            self.t_tool = np.asarray(t_tool, dtype=float)
        
    def forward_kinematics(
        self, q: List[float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: """returns T (homogeneous transform from base to end-effector),
    R (rotation matrix (top-left block of T), p (position vector (top-right column of T))"""

    frames = self.all_frames(q) """intialises frames"""


    def all_frames(self, q: List[float]) -> List[np.ndarray]:

        T = np.eye(4) """creates identity matrix, base frame"""

        frames = [] """create empty array to append to"""

        for i in range(self._dh_table.num_joints): """iterate over number of joints"""

            A_i = self._dh_table.get_transform(i, q_[i]) """delegates to DHTable for the single






    
