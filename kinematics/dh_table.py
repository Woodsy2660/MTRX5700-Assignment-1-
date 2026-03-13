# imports
import re
import math
from typing import List, Tuple
import numpy as np
import sympy as sp

from .robot_parser import parse_robot_file 

class DHTable:

    def __init__(self, dh_params: np.ndarray, joint_types: List[str]) -> None:
        self._dh_params = np.asarray(dh_params, dtype=float) ## takes in the DH table extracted from the text reader function
        self._joint_types = [jt.upper() for jt in joint_types] #creates a list of the join types from the reader and adds to class


        self.A_matrices = [self._compute_sym_transform(i) for i in range(self.num_joints())] ## compute and store all A matrices on construction of class


    @classmethod
    def from_file(cls, filepath: str) -> Tuple['DHTable', str, np.ndarray]: ## this is the first thing that runs to instantiate the class
        name, joint_types, dh_params, q_dot = parse_robot_file(filepath) ## uses the parsing function described above
        return cls(dh_params, joint_types), name, q_dot ## returns extracted dh_params and joint types along with robot name and q_dot

    def num_joints(self) -> int:
        return self._dh_params.shape[0] ## literally just returns the # joints, used later in for loops for creating A matrix tables

    def _get_sym_params(self, joint_index: int) -> tuple: ## this is used to create clean outputs for tables, using cos and sin and pi etc
        row = self._dh_params[joint_index] ## gathering data needed from class attributes
        jtype = self._joint_types[joint_index]
        joint_num = joint_index + 1
        q_sym = sp.Symbol(f"q{joint_num}")

        def _to_sym(val: float) -> sp.Expr: ## This is all sympi code to ensure the variables work
            if math.isnan(val):
                return q_sym
            # Recognise common multiples of pi for clean display
            for num, den in [(1,1),(1,2),(1,3),(1,4),(1,6),(2,3),(3,4),(5,6),
                              (-1,1),(-1,2),(-1,3),(-1,4),(-1,6),(-2,3),(-3,4)]:
                if abs(val - num * math.pi / den) < 1e-10:
                    return sp.Rational(num, den) * sp.pi
            return sp.nsimplify(val, rational=False, tolerance=1e-9)

        theta_raw, d_raw, a_raw, alpha_raw = row
        if jtype == 'R':
            theta, d = q_sym, _to_sym(d_raw)
        else:
            theta, d = _to_sym(theta_raw), q_sym
        return theta, d, _to_sym(a_raw), _to_sym(alpha_raw)

    ################################ TRANSFORM MATRICIES CODE ###############################

    def _compute_sym_transform(self, joint_index: int) -> sp.Matrix: ## Function that creates the A matrix for a given joint (used by joint_index)
        theta, d, a, alpha = self._get_sym_params(joint_index)
        ct, st = sp.cos(theta), sp.sin(theta)
        ca, sa = sp.cos(alpha), sp.sin(alpha)

        ## A matrix based on mathematical theory from lecture
        A = sp.Matrix([
            [ct,  -st*ca,   st*sa,  a*ct],
            [st,   ct*ca,  -ct*sa,  a*st],
            [ 0,   sa,       ca,    d   ],
            [ 0,   0,        0,     1   ],
        ])
        return sp.trigsimp(A)

    def get_transform(self, joint_index: int, q_i: float) -> np.ndarray:
        """
        Returns the 4x4 numeric A-matrix for joint_index given live joint value q_i.
        Substitutes q_i into whichever parameter is stored as nan (the joint variable).
        """
        row = self._dh_params[joint_index].copy()

        # Find which column holds nan — that is the joint variable slot
        # np.isnan returns a boolean array, argmax finds the index of the first True
        var_col = int(np.argmax(np.isnan(row)))
        row[var_col] = q_i

        # Unpack in the order your parser stores them: [theta, d, a, alpha]
        theta, d, a, alpha = row

        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)

        return np.array([
            [ct, -st * ca,  st * sa,  a * ct],
            [st,  ct * ca, -ct * sa,  a * st],
            [0.0,      sa,       ca,       d],
            [0.0,     0.0,      0.0,     1.0],
        ])

    ############# EXECUTING CLASS FUNCTIONS AND PRINTING ALL INFO #################

    def print_table(self, robot_name: str = "") -> None:
        header = f"DH Table — {robot_name}" if robot_name else "DH Table"
        print("\n" + "=" * 62)
        print(f"  {header}")
        print("=" * 62)
        print(f"  {'Joint':<8} {'Type':<8} {'theta (rad)':<16} {'d (m)':<14} {'a (m)':<14} {'alpha (rad)'}")
        print("  " + "-" * 58)
        for i, (row, jt) in enumerate(zip(self._dh_params, self._joint_types), 1):
            theta_str = f"q{i}" if math.isnan(row[0]) else f"{row[0]:.4f}"
            d_str     = f"q{i}" if math.isnan(row[1]) else f"{row[1]:.4f}"
            a_str     = f"{row[2]:.4f}"
            alpha_str = f"{row[3]:.4f}"
            print(f"  J{i:<7} {jt:<8} {theta_str:<16} {d_str:<14} {a_str:<14} {alpha_str}")
        print("=" * 62 + "\n")

    def print_all_transforms(self, robot_name: str = "") -> None:
        W = 72
        print(f"\n{'='*W}")
        title = f"  Individual A-Matrices — {robot_name}" if robot_name else "  Individual A-Matrices"
        print(title)
        print(f"{'='*W}")

        for i, A in enumerate(self.A_matrices):
            jtype = self._joint_types[i]
            print(f"\n  A{i+1}  (J{i+1}, {'Revolute' if jtype=='R' else 'Prismatic'}):")
            self._print_sym_matrix(A)

        print(f"\n{'='*W}")
        print("  T_0n  (Base → End-Effector):")
        self._print_sym_matrix(self.T)
        print(f"{'='*W}\n")

    @staticmethod
    def _print_sym_matrix(M: sp.Matrix) -> None:
        """Print a SymPy matrix with aligned columns."""
        rows, cols = M.shape
        str_grid = [[str(sp.nsimplify(M[r, c])) for c in range(cols)] for r in range(rows)]
        col_widths = [max(len(str_grid[r][c]) for r in range(rows)) for c in range(cols)]
        for row in str_grid:
            cells = "  ".join(cell.rjust(col_widths[c]) for c, cell in enumerate(row))
            print(f"  [ {cells} ]")




        