
# Forward Kinematics Calculation Implementation using Python


import numpy as np
from numpy import cos, sin, pi

# DH Parameters 
DH_PARAMS = {
    'd': [0.000, 0.000, 0.000, 0.127, 0.100, 0.100],  # meters
    'a': [0.163, -0.425, -0.392, 0.000, 0.000, 0.000],  # meters
    'alpha': [pi/2, 0, 0, pi/2, -pi/2, 0]  # radians
}


def dh_transform(theta, d, a, alpha):
    """
    Creates a single homogeneous transformation matrix using DH parameters.

    Returns:
    - 4x4 homogeneous transformation matrix T
    """
    ct = cos(theta)
    st = sin(theta)
    ca = cos(alpha)
    sa = sin(alpha)
    
    # Put T = A for simplicity here

    T = np.array([
        [ct,  -st*ca,   st*sa,  a*ct],
        [st,   ct*ca,  -ct*sa,  a*st],
        [0,    sa,      ca,     d   ],
        [0,    0,       0,      1   ]
    ])

    
    return T



def forward_kinematics(joint_angles):
    """
    Computes forward kinematics 
    
    Parameters:
    - joint_angles: array of 6 joint angles [q1, q2, q3, q4, q5, q6] in radians
    
    Returns:
    - T_0_6: 4x4 transformation matrix from base to end-effector
    """
    q = joint_angles
    d = DH_PARAMS['d']
    a = DH_PARAMS['a']
    alpha = DH_PARAMS['alpha']
    
    # Initialise as identity matrix
    T_0_6 = np.eye(4)
    
    # Multiply transformation matrices for each joint
    for i in range(6):
        T_i = dh_transform(q[i], d[i], a[i], alpha[i])
        T_0_6 = T_0_6 @ T_i     # Matrix multiplication
    
    return T_0_6



def extract_position(T):
    """Extract position vector from transformation matrix."""
    return T[0:3, 3]

def extract_rotation_matrix(T):
    """Extract 3x3 rotation matrix from transformation matrix."""
    return T[0:3, 0:3]

def rotation_matrix_to_axis_angle(R):
    """
    Convert rotation matrix R to axis-angle representation.
    
    """
    # Calculate rotation angle
    theta = np.arccos((np.trace(R) - 1) / 2)
    
    # Handle special case: no rotation
    if abs(theta) < 1e-6:
        return 0, 0, 0
    
    # Handle special case: 180 degree rotation
    if abs(theta - pi) < 1e-6:
        # Find the axis (eigenvector with eigenvalue 1)
        rx = np.sqrt((R[0,0] + 1) / 2)
        ry = np.sqrt((R[1,1] + 1) / 2)
        rz = np.sqrt((R[2,2] + 1) / 2)
        
        # Determine signs
        if R[0,1] < 0: ry = -ry
        if R[0,2] < 0: rz = -rz
        
        return rx * theta, ry * theta, rz * theta
    
    # General case
    rx = (R[2,1] - R[1,2]) / (2 * sin(theta)) * theta
    ry = (R[0,2] - R[2,0]) / (2 * sin(theta)) * theta
    rz = (R[1,0] - R[0,1]) / (2 * sin(theta)) * theta
    
    return rx, ry, rz   # axis-angle vectors (radians)



def get_tcp_pose(joint_angles):
    """
    Get TCP position and orientation from joint angles (array [q1, q2, q3, q4, q5, q6] )
    
    Returns:
    - position: [x, y, z] in meters
    - orientation: [rx, ry, rz] in radians (axis-angle)
    """
    T = forward_kinematics(joint_angles)
    
    position = extract_position(T)
    R = extract_rotation_matrix(T)
    orientation = rotation_matrix_to_axis_angle(R)
    
    return position, orientation


# Tool Position Test 1
q_test1 = np.array([0, -pi/2, 0, -pi/2, 0, 0])

# Ground truth for test set 1 as per photo:
pos_1 = np.array([-8.58, -232.24, 679.49])                # mm
orientation_1 = np.array([0.043, 2.239, -2.242])          # rad



position, orientation = get_tcp_pose(q_test1)

print("Joint angles (rad):", q_test1)
print("TCP Position found (m):", position)
print("TCP Orientation found (rad):", orientation)

print("TCP Position ground truth (m):", (pos_1/1000))
print("TCP Orientation ground truth (rad):", orientation_1)


# Print full transformation matrix for joint
T = forward_kinematics(q_test1)
print("\nTransformation matrix T_0_6:")
print(T)


