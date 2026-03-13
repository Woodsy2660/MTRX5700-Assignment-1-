# imports
import re
import math
from typing import List, Tuple
import numpy as np

# Parser function that reads the robot text file - currently no error handling can add later

def parse_robot_file(filepath: str) -> Tuple[str, List[str], np.ndarray, np.ndarray]:

    MATH_ENV = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}

    def _eval(token: str) -> float:
        token = token.strip()
        if re.fullmatch(r'q\d+', token):
            return float('nan')
        return float(eval(token, {"__builtins__": {}}, MATH_ENV))

    with open(filepath) as fh:
        lines = [re.sub(r'//.*', '', l).strip() for l in fh]
    lines = [l for l in lines if l]

    name        = re.match(r'<(.+)>', lines[0]).group(1).strip()
    joint_types = re.search(r'\[([^\]]+)\]', lines[1]).group(1).replace(' ', '').split(',')

    # Separate DH parameter lines from q_dot line
    dh_lines = []
    q_dot = None

    for l in lines[2:]:
        if 'q_dot_values' in l:
            # Extract joint velocities
            values_str = re.search(r'\[([^\]]+)\]', l).group(1)
            q_dot = np.array([_eval(t) for t in values_str.split(',')])
        else:
            dh_lines.append(l)

    dh_params = np.array([
        [_eval(t) for t in re.search(r'\[([^\]]+)\]', l).group(1).split(',')]
        for l in dh_lines
    ])

    # Convert d and a from millimeters to meters
    # DH params are in order: [theta, d, a, alpha]
    # Convert column 1 (d) and column 2 (a) from mm to m
    dh_params[:, 1] /= 1000.0  # d parameter
    dh_params[:, 2] /= 1000.0  # a parameter

    return name, joint_types, dh_params, q_dot