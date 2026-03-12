"""
Plots links and coordinate frames for a serial manipulator.
Takes a list of 4x4 homogeneous transforms (one per joint) and draws the robot in 3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_robot(frames, q: np.ndarray = None, ax: plt.Axes = None, show: bool = True, robot_name: str = "Robot") -> plt.Axes:
    """
    Plot the robot arm for a given joint configuration.

    Parameters
    ----------
    frames     : list of 4x4 homogeneous transforms (one per joint)
    q          : joint angles in radians, shape (n,). Defaults to zero config.
    ax         : existing matplotlib 3D axes to draw on. Creates new figure if None.
    show       : call plt.show() at the end if True.
    robot_name : name of the robot to display in the title.

    Returns
    -------
    ax : the matplotlib 3D axes object
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

    # ── Collect joint origins (base + one per joint) ─────────────────────────
    origins = [np.zeros(3)]          # base frame origin
    for T in frames:
        origins.append(T[:3, 3])
    origins = np.array(origins)      # shape (n+1, 3)

    # ── Draw links ───────────────────────────────────────────────────────────
    ax.plot(origins[:, 0], origins[:, 1], origins[:, 2],
            'o-', color='steelblue', linewidth=3, markersize=6,
            zorder=3, label='Links')

    # ── Draw coordinate frames at each joint ─────────────────────────────────
    span = max((origins.max(axis=0) - origins.min(axis=0)).max(), 1e-3)
    axis_len = span * 0.12   # 12% of robot span
    colors = {'x': 'red', 'y': 'green', 'z': 'blue'}

    # Base frame
    base_T = np.eye(4)
    _draw_frame(ax, base_T, axis_len, colors, label_prefix='0')

    for i, T in enumerate(frames):
        _draw_frame(ax, T, axis_len, colors)

    # ── Formatting ───────────────────────────────────────────────────────────
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if q is not None:
        q_str = ', '.join(f'{np.degrees(qi):.1f}°' for qi in q)
        ax.set_title(f'{robot_name}  |  q = [{q_str}]')
    else:
        ax.set_title(f'{robot_name} Kinematic Chain')

    _set_equal_axes(ax, origins)

    # Legend proxy for axes colours
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red',       lw=2, label='x-axis'),
        Line2D([0], [0], color='green',     lw=2, label='y-axis'),
        Line2D([0], [0], color='blue',      lw=2, label='z-axis'),
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
    ax.set_zlim(centre[2] - half_range, centre[2] + half_range)