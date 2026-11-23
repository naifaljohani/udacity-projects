"""Generate the 3D Motion Planning GIFs for the README.

1. motion_planning_astar.gif  - the project's own A* + collinearity pruning
   (planning_utils_with_Bresenham.py) run on colliders.csv.
2. motion_planning_flight.gif - replay of the real telemetry log (Logs/TLog.txt)
   recorded while the drone flew the planned route in the simulator.

Usage:
    python make_motion_planning_gifs.py <path-to-extracted-FCND-Motion-Planning> <out-dir>
"""
import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

PROJECT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, PROJECT)
from planning_utils_with_Bresenham import create_grid, a_star, heuristic, prune_path  # noqa: E402

TARGET_ALTITUDE, SAFETY_DISTANCE = 5, 5
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
BG, OBST, FREE = "#0f172a", "#475569", "#1e293b"

# ---------------------------------------------------------------- data
data = np.loadtxt(os.path.join(PROJECT, "colliders.csv"), delimiter=",", skiprows=2)
grid, n_off, e_off = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

rows = [l.strip().split(",") for l in open(os.path.join(PROJECT, "Logs", "TLog.txt"))]
lp = np.array([[float(r[1]), float(r[2]), float(r[3]), float(r[4])]
               for r in rows if r[0] == "MsgID.LOCAL_POSITION"])
# keep the actual mission: from the take-off command until touchdown, drop the
# one-off spawn glitches (|down| > 20 m) that appear before take-off
lp = lp[np.abs(lp[:, 3]) < 20]
airborne = np.where(-lp[:, 3] > 1.0)[0]
t_up, t_down = lp[airborne[0], 0], lp[airborne[-1], 0]
lp = lp[(lp[:, 0] >= t_up - 3) & (lp[:, 0] <= t_down + 6)]

start_local = lp[0, 1:3]
goal_local = lp[-1, 1:3]
grid_start = (int(round(start_local[0])) - n_off, int(round(start_local[1])) - e_off)
grid_goal = (int(round(goal_local[0])) - n_off, int(round(goal_local[1])) - e_off)

print("Running A* from", grid_start, "to", grid_goal, "...")
path, cost = a_star(grid, heuristic, grid_start, grid_goal)
pruned = prune_path(path)
print(f"A* path: {len(path)} cells, cost {cost:.1f}; pruned to {len(pruned)} waypoints")

path = np.array(path) + [n_off, e_off]
pruned = np.array(pruned) + [n_off, e_off]

# view window around the mission
pad = 60
allpts = np.vstack([path, lp[:, 1:3]])
nmin, nmax = allpts[:, 0].min() - pad, allpts[:, 0].max() + pad
emin, emax = allpts[:, 1].min() - pad, allpts[:, 1].max() + pad


def base_axes(title):
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=80)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(FREE)
    cmap = matplotlib.colors.ListedColormap([FREE, OBST])
    ax.imshow(grid, origin="lower", cmap=cmap,
              extent=[e_off, e_off + grid.shape[1], n_off, n_off + grid.shape[0]])
    ax.set_xlim(emin, emax)
    ax.set_ylim(nmin, nmax)
    ax.set_xlabel("East (m)", color="#cbd5e1")
    ax.set_ylabel("North (m)", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    for s in ax.spines.values():
        s.set_color("#334155")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    ax.plot(start_local[1], start_local[0], "o", ms=10, color="#22c55e", zorder=5)
    ax.plot(goal_local[1], goal_local[0], "*", ms=16, color="#facc15", zorder=5)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------- GIF 1: A*
fig, ax = base_axes("A* search + path pruning (San Francisco map, 5 m)")
raw_line, = ax.plot([], [], "-", color="#38bdf8", lw=1.5, label=f"A* path ({len(path)} cells)")
wp_line, = ax.plot([], [], "-o", color="#f472b6", lw=2.2, ms=6,
                   label=f"pruned waypoints ({len(pruned)})")
info = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", color="white",
               fontsize=10, bbox=dict(fc=BG, ec="#334155", alpha=0.85))
ax.legend(loc="lower right", facecolor=BG, edgecolor="#334155", labelcolor="white", fontsize=9)

N_DRAW, N_PRUNE, N_HOLD = 45, 15, 20


def anim_astar(i):
    if i < N_DRAW:
        k = max(2, int(len(path) * (i + 1) / N_DRAW))
        raw_line.set_data(path[:k, 1], path[:k, 0])
        wp_line.set_data([], [])
        info.set_text("1) A* (8-connected, diagonal cost sqrt 2)")
    elif i < N_DRAW + N_PRUNE:
        raw_line.set_alpha(0.35)
        k = max(1, int(len(pruned) * (i - N_DRAW + 1) / N_PRUNE))
        wp_line.set_data(pruned[:k, 1], pruned[:k, 0])
        info.set_text("2) Collinearity check removes\n    redundant waypoints")
    else:
        info.set_text(f"cells: {len(path)}  ->  waypoints: {len(pruned)}")
    return raw_line, wp_line, info


ani = FuncAnimation(fig, anim_astar, frames=N_DRAW + N_PRUNE + N_HOLD, blit=False)
ani.save(os.path.join(OUT, "motion_planning_astar.gif"), writer=PillowWriter(fps=12))
plt.close(fig)

# ---------------------------------------------------------------- GIF 2: flight replay
fig, ax = base_axes("Real flight log replay (TLog.txt)")
ax.plot(pruned[:, 1], pruned[:, 0], "--", color="#f472b6", lw=1, alpha=0.6, label="planned")
trail, = ax.plot([], [], "-", color="#38bdf8", lw=2.2, label="flown (telemetry)")
drone, = ax.plot([], [], "o", ms=9, color="white", mec="#38bdf8", mew=2, zorder=6)
ax.legend(loc="lower right", facecolor=BG, edgecolor="#334155", labelcolor="white", fontsize=9)
hud = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", color="white",
              family="monospace", fontsize=10, bbox=dict(fc=BG, ec="#334155", alpha=0.85))

t = lp[:, 0] - lp[0, 0]
N_FRAMES = 110
idx = np.linspace(0, len(lp) - 1, N_FRAMES).astype(int)
idx = np.concatenate([idx, np.full(15, len(lp) - 1)])


def anim_flight(i):
    k = idx[i]
    trail.set_data(lp[:k + 1, 2], lp[:k + 1, 1])
    drone.set_data([lp[k, 2]], [lp[k, 1]])
    hud.set_text(f"t   = {t[k]:6.1f} s\nN   = {lp[k,1]:6.1f} m\nE   = {lp[k,2]:6.1f} m\n"
                 f"alt = {-lp[k,3]:6.1f} m")
    return trail, drone, hud


ani = FuncAnimation(fig, anim_flight, frames=len(idx), blit=False)
ani.save(os.path.join(OUT, "motion_planning_flight.gif"), writer=PillowWriter(fps=12))
plt.close(fig)
print("done")
