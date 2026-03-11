"""
Plots links and coordinate frames for a serial manipulator.
Takes a list of 4x4 homogeneous transforms (one per joint) and draws the robot in 3D.

Currently uses hardcoded UR5e DH parameters at q=0 to compute frames internally.
TODO: replace _compute_frames_hardcoded() call with arm_kinematics.all_frames(q)
      once arm_kinematics.py is working.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ── Hardcoded UR5e DH params [theta_offset, d, a, alpha] in metres ──────────
# Order matches dh_table.py: row = [theta, d, a, alpha]
# Variable joints have theta_offset=0 (q is added at call time)
_UR5E_DH = np.array([
    [0,  0.1625,  0.0,     np.pi/2],
    [0,  0.0,    -0.425,   0.0    ],
    [0,  0.0,    -0.3922,  0.0    ],
    [0,  0.1333,  0.0,     np.pi/2],
    [0,  0.0997,  0.0,    -np.pi/2],
    [0,  0.0996,  0.0,     0.0    ],
])


def _single_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """Standard DH homogeneous transform for one joint."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d     ],
        [0,   0,        0,       1     ],
    ])


def _compute_frames(q: np.ndarray) -> list:
    """
    Compute all cumulative base-to-joint transforms for given joint angles q.
    Returns list of 4x4 arrays: [T_0_1, T_0_2, ..., T_0_6]

    TODO: replace this with arm_kinematics.ArmKinematics.all_frames(q)
          once arm_kinematics.py is complete.
    """
    T = np.eye(4)
    frames = []
    for i, row in enumerate(_UR5E_DH):
        theta_offset, d, a, alpha = row
        theta = theta_offset + q[i]
        A = _single_transform(theta, d, a, alpha)
        T = T @ A
        frames.append(T.copy())
    return frames


def plot_robot(q: np.ndarray = None, ax: plt.Axes = None, show: bool = True) -> plt.Axes:
    """
    Plot the UR5e robot arm for a given joint configuration.

    Parameters
    ----------
    q    : joint angles in radians, shape (6,). Defaults to zero config.
    ax   : existing matplotlib 3D axes to draw on. Creates new figure if None.
    show : call plt.show() at the end if True.

    Returns
    -------
    ax : the matplotlib 3D axes object
    """
    if q is None:
        q = np.zeros(6)
    q = np.asarray(q, dtype=float)

    frames = _compute_frames(q)

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

    # ── Collect joint origins (base + one per joint) ─────────────────────────
    origins = [np.zeros(3)]          # base frame origin
    for T in frames:
        origins.append(T[:3, 3])
    origins = np.array(origins)      # shape (7, 3)

    # ── Draw links ───────────────────────────────────────────────────────────
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2],
            'o-', color='steelblue', linewidth=3, markersize=6,
            zorder=3, label='Links')

    # ── Draw coordinate frames at each joint ─────────────────────────────────
    axis_len = 0.05   # 50 mm arrow length
    colors = {'x': 'red', 'y': 'green', 'z': 'blue'}

    # Base frame
    base_T = np.eye(4)
    _draw_frame(ax, base_T, axis_len, colors, label_prefix='0')

    for i, T in enumerate(frames):
        _draw_frame(ax, T, axis_len, colors)

    # ── Formatting ───────────────────────────────────────────────────────────
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    q_str = ', '.join(f'{np.degrees(qi):.1f}°' for qi in q)
    ax.set_title(f'UR5e  |  q = [{q_str}]')

    _set_equal_axes(ax, origins)

    # Legend proxy for axes colours
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red',      lw=2, label='x-axis'),
        Line2D([0], [0], color='green',    lw=2, label='y-axis'),
        Line2D([0], [0], color='blue',     lw=2, label='z-axis'),
        Line2D([0], [0], color='steelblue', lw=3, label='link'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def _draw_frame(ax, T: np.ndarray, length: float, colors: dict,
                label_prefix: str = '') -> None:
    """Draw x/y/z axes of a frame given its 4x4 transform."""
    origin = T[:3, 3]
    for col, (axis_name, color) in enumerate(colors.items()):
        direction = T[:3, col] * length
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color=color, linewidth=1.5, arrow_length_ratio=0.3,
        )


def _set_equal_axes(ax, points: np.ndarray) -> None:
    """Force equal aspect ratio on a 3D matplotlib axes."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centre = (mins + maxs) / 2
    half_range = max((maxs - mins).max() / 2, 0.3)

    ax.set_xlim(centre[0] - half_range, centre[0] + half_range)
    ax.set_ylim(centre[1] - half_range, centre[1] + half_range)
    ax.set_zlim(0, centre[2] + half_range)


# ── Run directly to preview zero config ──────────────────────────────────────
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 7:
        q_input = np.array([float(x) for x in sys.argv[1:]])
    else:
        q_input = np.zeros(6)
        print("No q provided — showing zero configuration.")
        print("Usage: python arm_visualiser.py q1 q2 q3 q4 q5 q6  (radians)")

    plot_robot(q_input)
