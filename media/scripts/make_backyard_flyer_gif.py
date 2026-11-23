"""Generate backyard_flyer.gif.

Re-simulates the event-driven state machine from backyard_flyer/backyard_flyer.py
(same waypoints, same 3 m altitude, same transition thresholds) on a simple
first-order vehicle model, and animates it in 3D.

Usage: python make_backyard_flyer_gif.py <out-dir>
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

OUT = sys.argv[1]
BG = "#0f172a"

# same box as BackyardFlyer.target_position  (north, east, altitude)
WAYPOINTS = np.array([[0, 10, 3], [10, 10, 3], [10, -10, 3], [0, -10, 3], [0, 0, 3]], float)

# ------------------------------------------------------------ simulate
dt, gain, vmax = 0.05, 1.2, 3.0
pos = np.zeros(3)
state, target_i = "ARMING", 0
target = np.array([0, 0, 3.0])
log = []
t = 0.0
hold = 0
while True:
    # ------------- transitions copied from local_position_callback / state_callback
    if state == "ARMING":
        hold += 1
        if hold > 12:
            state = "TAKEOFF"
    elif state == "TAKEOFF" and pos[2] > 3 * 0.98:
        state, target = "WAYPOINT", WAYPOINTS[target_i]
    elif state == "WAYPOINT" and target_i <= 3:
        if np.linalg.norm(WAYPOINTS[target_i, :2] - pos[:2]) < 1:
            target_i += 1
            target = WAYPOINTS[target_i]
    vel = np.clip(gain * (target - pos), -vmax, vmax) if state not in ("ARMING", "DISARMING", "MANUAL") else np.zeros(3)
    if state == "WAYPOINT" and target_i == 4 and np.linalg.norm(vel) < 0.1:
        state, target = "LANDING", np.array([pos[0], pos[1], 0.0])
        vel = np.zeros(3)
    if state == "LANDING" and pos[2] < 0.1:
        state, hold = "DISARMING", 0
    if state == "DISARMING":
        hold += 1
        if hold > 10:
            state = "MANUAL"
    pos = pos + vel * dt
    log.append((t, *pos, state, target_i))
    t += dt
    if state == "MANUAL":
        for _ in range(15):
            log.append((t, *pos, state, target_i))
        break

P = np.array([[r[1], r[2], r[3]] for r in log])
states = [r[4] for r in log]
STEP = 4
frames = list(range(0, len(log), STEP))

# ------------------------------------------------------------ animate
COL = {"ARMING": "#f59e0b", "TAKEOFF": "#22c55e", "WAYPOINT": "#38bdf8",
       "LANDING": "#a78bfa", "DISARMING": "#f87171", "MANUAL": "#94a3b8"}

fig = plt.figure(figsize=(6.0, 4.9), dpi=80)
fig.patch.set_facecolor(BG)
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_pane_color((0.12, 0.16, 0.23, 1))
    a.label.set_color("#cbd5e1")
ax.tick_params(colors="#94a3b8")
ax.set_xlim(-12, 12); ax.set_ylim(-2, 12); ax.set_zlim(0, 4)
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_zlabel("Alt (m)")
ax.view_init(elev=28, azim=-60)
ax.set_title("Backyard Flyer: event-driven box mission", color="white", fontweight="bold")

wp = np.vstack([[0, 0, 3], WAYPOINTS])
ax.plot(wp[:, 1], wp[:, 0], wp[:, 2], "--", color="#64748b", lw=1)
ax.scatter(WAYPOINTS[:, 1], WAYPOINTS[:, 0], WAYPOINTS[:, 2], color="#facc15", s=40, depthshade=False)
trail, = ax.plot([], [], [], color="#38bdf8", lw=2.5)
drone, = ax.plot([], [], [], "o", ms=10, color="white", mec="#38bdf8", mew=2)
shadow, = ax.plot([], [], [], "o", ms=6, color="#000000", alpha=0.4)
label = fig.text(0.03, 0.04, "", color="white", fontsize=13, fontweight="bold", family="monospace",
                 bbox=dict(fc=BG, ec="#334155"))
fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.92)


def update(i):
    k = frames[i]
    trail.set_data(P[:k + 1, 1], P[:k + 1, 0]); trail.set_3d_properties(P[:k + 1, 2])
    drone.set_data([P[k, 1]], [P[k, 0]]); drone.set_3d_properties([P[k, 2]])
    shadow.set_data([P[k, 1]], [P[k, 0]]); shadow.set_3d_properties([0])
    s = states[k]
    label.set_text(f"state: {s}")
    label.set_color(COL[s])
    ax.view_init(elev=28, azim=-60 + 25 * i / len(frames))
    return trail, drone, label


ani = FuncAnimation(fig, update, frames=len(frames))
ani.save(os.path.join(OUT, "backyard_flyer.gif"), writer=PillowWriter(fps=15))
print("frames:", len(frames))
