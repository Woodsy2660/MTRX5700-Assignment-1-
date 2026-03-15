# MTRX5700-Assignment-1-
This repo contains the codebase for the kinematics assignment 1.

Please use run_fk.py to run forward kinematics and Jacobian solvers, outputting DH table used, Forward Kinematics results, Jacobian results, and the Visualised Arm. Please use run_model_verification.py to calculate and display errors of kinematics model v.s. ground truth values. See the two scripts' header comments for command line running instructions. 

To test different configurations of the robot arm, enter the desired joint angle for each joint across the arm in the robotinput.txt file and run as per instruction in run_fk.py script header comment. 

E.g., if using UR5e, the order of joint angles input is <base, shoulder, elbow, wrist1, wrist2, wrist3> in ur5einput.txt.

