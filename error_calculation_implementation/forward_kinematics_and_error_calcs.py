
# Forward Kinematics Calculation and Error Calculation Implementation using Python


import numpy as np
from numpy import cos, sin, pi
import pandas as pd



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
    Computes forward kinematics to get the final transformation matrix from base to end effector
    
    Parameters:
    joint_angles: array of 6 joint angles [q1, q2, q3, q4, q5, q6] in radians

    """
    q = joint_angles
    d = DH_PARAMS['d']
    a = DH_PARAMS['a']
    alpha = DH_PARAMS['alpha']
    

    T_0_6 = np.eye(4)
    
    # Multiply transformation matrices for each joint
    for i in range(6):
        T_i = dh_transform(q[i], d[i], a[i], alpha[i])
        T_0_6 = T_0_6 @ T_i     
    
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
    

    # If no rotation
    if abs(theta) < 1e-6:
        return 0, 0, 0
    
    # If 180 degree rotation
    if abs(theta - pi) < 1e-6:

        # Find the axis (eigenvector with eigenvalue 1)
        rx = np.sqrt((R[0,0] + 1) / 2)
        ry = np.sqrt((R[1,1] + 1) / 2)
        rz = np.sqrt((R[2,2] + 1) / 2)
        
        # Determine signs
        if R[0,1] < 0: ry = -ry
        if R[0,2] < 0: rz = -rz
        
        return rx * theta, ry * theta, rz * theta
    


    # General calc
    rx = (R[2,1] - R[1,2]) / (2 * sin(theta)) * theta
    ry = (R[0,2] - R[2,0]) / (2 * sin(theta)) * theta
    rz = (R[1,0] - R[0,1]) / (2 * sin(theta)) * theta
    
    return rx, ry, rz   # axis-angle vectors (radians)



def get_tcp_pose(joint_angles):
    """
    Get TCP position and orientation from joint angles ( [q1, q2, q3, q4, q5, q6] )
    """

    T = forward_kinematics(joint_angles)
    
    position = extract_position(T)
    R = extract_rotation_matrix(T)
    orientation = rotation_matrix_to_axis_angle(R)
    
    return position, orientation



def axis_angle_to_rotation_matrix(rx, ry, rz):
    """ Convert axis-angle rotation vector to 3x3 rotation matrix. """
    theta = np.sqrt(rx**2 + ry**2 + rz**2)
    if theta < 1e-6:
        return np.eye(3)
    
    kx, ky, kz = rx / theta, ry / theta, rz / theta
    c, s = cos(theta), sin(theta)
    t = 1 - c

    return np.array([
        [t*kx*kx + c,    t*kx*ky - s*kz, t*kx*kz + s*ky],
        [t*kx*ky + s*kz, t*ky*ky + c,    t*ky*kz - s*kx],
        [t*kx*kz - s*ky, t*ky*kz + s*kx, t*kz*kz + c   ]
    ])



def calculate_Tmatrix_errors(T_model, T_gt):
    """
    Compute translational and rotational errors 
    """
    
    T_gt_inv = np.linalg.inv(T_gt)
    T_err = T_gt_inv @ T_model


    t_err = T_err[0:3, 3]
    R_err = T_err[0:3, 0:3]

    # Translational error: magnitude of t_err
    e_trans = np.linalg.norm(t_err)

    # Rotational error: arccos((Tr(R_err) - 1) / 2)
    cos_theta = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    e_rot = np.arccos(cos_theta)

    return e_trans, e_rot, t_err, R_err


def load_ground_truth_csv(filepath):
    """
    Load ground truth data from CSV with columns
    """
    df = pd.read_csv(filepath)
    
    ground_truth_data = []
    for i, row in df.iterrows():
        entry = {
            'config': i + 1,

            # Convert position to meters 
            'position_gt': np.array([
                row['X (mm)'] / 1000.0,
                row['Y (mm)'] / 1000.0,
                row['Z (mm)'] / 1000.0
            ]),

            'orientation_gt': np.array([
                row['RX (rad)'],
                row['RY (rad)'],
                row['RZ (rad)']
            ]),

            'joint_angles': np.radians([
                row['Base (deg)'],
                row['Shoulder (deg)'],
                row['Elbow (deg)'],
                row['Wrist1 (deg)'],
                row['Wrist2 (deg)'],
                row['Wrist3 (deg)']
            ])
        }
        ground_truth_data.append(entry)
    
    return ground_truth_data



def build_transformation_matrix_from_csv(position, orientation):
    """Build 4x4 T_gt from position [x,y,z] and axis-angle [rx,ry,rz]."""
    R = axis_angle_to_rotation_matrix(orientation[0], orientation[1], orientation[2])
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = position
    return T


if __name__ == "__main__":

    # File of data collection (all configurations)
    filepath = "robot_positions_theflyingdutchman.csv"

    ground_truth_data = load_ground_truth_csv(filepath)

    # Evaluate all configurations (poses) from data collection
    errors_position = []
    errors_orientation = []


    for data in ground_truth_data:
        q = data['joint_angles']
        pos_gt = data['position_gt']
        orient_gt = data['orientation_gt']

        T_model = forward_kinematics(q)
        T_gt = build_transformation_matrix_from_csv(pos_gt, orient_gt)   

        e_trans, e_rot, t_err, R_err = calculate_Tmatrix_errors(T_model, T_gt)  
        errors_position.append(e_trans)
        errors_orientation.append(e_rot)

        print(f"Configuration {data['config']}:")
        print(f"  Position error: {e_trans*1000:.2f} mm")
        print(f"    X error: {t_err[0]*1000:.2f} mm")
        print(f"    Y error: {t_err[1]*1000:.2f} mm")
        print(f"    Z error: {t_err[2]*1000:.2f} mm")


        print(f"  Rotation error: {np.degrees(e_rot):.2f} degrees")


        rx_err, ry_err, rz_err = rotation_matrix_to_axis_angle(R_err)
        print(f"    RX error: {np.degrees(rx_err):.2f} deg")
        print(f"    RY error: {np.degrees(ry_err):.2f} deg")
        print(f"    RZ error: {np.degrees(rz_err):.2f} deg")

        print("\n")
            


    # Summary statistics
    print("\n=== SUMMARY ===")
    print(f"Mean position error: {np.mean(errors_position)*1000:.2f} mm")
    print(f"Max position error: {np.max(errors_position)*1000:.2f} mm")
    print(f"Mean orientation error: {np.degrees(np.mean(errors_orientation)):.2f} deg")
    print(f"Max orientation error: {np.degrees(np.max(errors_orientation)):.2f} deg")





