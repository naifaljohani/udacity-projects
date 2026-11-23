"""Generate quad_controller.gif.

A line-by-line Python port of the cascaded controller in
Control_of_3D_Quadrotor/Controls_Project/QuadControl.cpp, using the tuned gains
from QuadControlParams.txt, flying a 3D figure-eight on a rigid-body quadrotor
model (NED frame, 500 Hz).

Usage: python make_controller_gif.py <out-dir>
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

# ------------------------------------------------ QuadControlParams.txt
mass, L, Ixx, Iyy, Izz, kappa = 0.5, 0.17, 0.0023, 0.0023, 0.0046, 0.016
minMotorThrust, maxMotorThrust = 0.1, 4.5
kpPosXY, kpPosZ, KiPosZ = 3, 10, 70
kpVelXY, kpVelZ = 12, 24
kpBank, kpYaw = 12, 2
kpPQR = np.array([69, 69, 5.0])
maxAscentRate, maxDescentRate, maxSpeedXY, maxAccelXY, maxTiltAngle = 5, 2, 5, 12, 0.7
g = 9.81
I = np.diag([Ixx, Iyy, Izz])


def C(x, lo, hi):
    return min(max(x, lo), hi)


# ------------------------------------------------ controller (port of QuadControl.cpp)
class QuadControl:
    def __init__(self):
        self.integratedAltitudeError = 0.0

    def GenerateMotorCommands(self, collThrustCmd, m):
        f_c = collThrustCmd
        f_p = m[0] * (np.sqrt(2) / L)
        f_q = m[1] * (np.sqrt(2) / L)
        f_r = m[2] / kappa
        return np.array([(f_c + f_p + f_q - f_r) / 4,    # front left
                         (f_c - f_p + f_q + f_r) / 4,    # front right
                         (f_c + f_p - f_q + f_r) / 4,    # rear left
                         (f_c - f_p - f_q - f_r) / 4])   # rear right

    def BodyRateControl(self, pqrCmd, pqr):
        return np.diag(I) * kpPQR * (pqrCmd - pqr)

    def RollPitchControl(self, accelCmd, R, collThrustCmd):
        c_acc = -collThrustCmd / mass
        b_c_x = C(accelCmd[0] / c_acc, -maxTiltAngle, maxTiltAngle)
        b_c_y = C(accelCmd[1] / c_acc, -maxTiltAngle, maxTiltAngle)
        b_c_x_dot = kpBank * (b_c_x - R[0, 2])
        b_c_y_dot = kpBank * (b_c_y - R[1, 2])
        row1 = np.array([R[1, 0], -R[0, 0]]) / R[2, 2]
        row2 = np.array([R[1, 1], -R[0, 1]]) / R[2, 2]
        return np.array([row1 @ [b_c_x_dot, b_c_y_dot], row2 @ [b_c_x_dot, b_c_y_dot], 0.0])

    def AltitudeControl(self, posZCmd, velZCmd, posZ, velZ, R, accelZCmd, dt):
        velZCmd = C(velZCmd, -maxAscentRate, maxDescentRate)
        pos_err = posZCmd - posZ
        self.integratedAltitudeError = C(self.integratedAltitudeError + pos_err * dt, -.27, .27)
        u_1_bar = (kpPosZ * pos_err + KiPosZ * self.integratedAltitudeError
                   + kpVelZ * (velZCmd - velZ) + accelZCmd)
        return ((g - u_1_bar) * mass) / R[2, 2]

    def LateralPositionControl(self, posCmd, velCmd, pos, vel, accelFF):
        velCmd = velCmd.copy(); velCmd[2] = 0
        accelFF = accelFF.copy(); accelFF[2] = 0
        velCmd[:2] += kpPosXY * (posCmd[:2] - pos[:2])
        velCmd[:2] = np.clip(velCmd[:2], -maxSpeedXY, maxSpeedXY)
        acc = accelFF + kpVelXY * (velCmd - vel) + accelFF
        acc[:2] = np.clip(acc[:2], -maxAccelXY, maxAccelXY)
        acc[2] = 0
        return acc

    def YawControl(self, yawCmd, yaw):
        yawRateCmd = 0
        err = yawRateCmd - yaw          # as implemented in the project
        return kpYaw * err

    def RunControl(self, traj, pos, vel, R, omega, dt):
        p, v, a = traj
        T = self.AltitudeControl(p[2], v[2], pos[2], vel[2], R, a[2], dt)
        margin = .1 * (maxMotorThrust - minMotorThrust)
        T = C(T, (minMotorThrust + margin) * 4, (maxMotorThrust - margin) * 4)
        desAcc = self.LateralPositionControl(p, v, pos, vel, a)
        desOmega = self.RollPitchControl(desAcc, R, T)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        desOmega[2] = self.YawControl(0.0, yaw)
        return self.GenerateMotorCommands(T, self.BodyRateControl(desOmega, omega))


# ------------------------------------------------ figure-eight reference
W = 2 * np.pi / 8.0   # 8 s per lap


def trajectory(t):
    p = np.array([2.0 * np.sin(W * t), 1.5 * np.sin(2 * W * t), -1.5 - 0.4 * np.sin(W * t)])
    v = np.array([2.0 * W * np.cos(W * t), 3.0 * W * np.cos(2 * W * t), -0.4 * W * np.cos(W * t)])
    a = np.array([-2.0 * W**2 * np.sin(W * t), -6.0 * W**2 * np.sin(2 * W * t), 0.4 * W**2 * np.sin(W * t)])
    return p, v, a


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


# ------------------------------------------------ simulate
dt, T_END = 0.002, 18.0
ctrl = QuadControl()
pos, vel = np.array([0.0, 0.0, -0.3]), np.zeros(3)
R, omega = np.eye(3), np.zeros(3)
hist = []
for k in range(int(T_END / dt)):
    t = k * dt
    traj = trajectory(t)
    F = np.clip(ctrl.RunControl(traj, pos, vel, R, omega, dt), minMotorThrust, maxMotorThrust)
    Mx = (F[0] - F[1] + F[2] - F[3]) * L / np.sqrt(2)
    My = (F[0] + F[1] - F[2] - F[3]) * L / np.sqrt(2)
    Mz = -kappa * (F[0] - F[1] - F[2] + F[3])
    M = np.array([Mx, My, Mz])
    acc = np.array([0, 0, g]) + R @ np.array([0, 0, -F.sum()]) / mass
    vel = vel + acc * dt
    pos = pos + vel * dt
    omega = omega + np.linalg.solve(I, M - np.cross(omega, I @ omega)) * dt
    R = R @ (np.eye(3) + skew(omega) * dt)
    u, _, vt = np.linalg.svd(R); R = u @ vt   # re-orthonormalise
    if k % 25 == 0:
        hist.append((t, pos.copy(), traj[0].copy(), R.copy(), F.copy()))

ts = np.array([h[0] for h in hist])
P = np.array([h[1] for h in hist])
Pref = np.array([h[2] for h in hist])
err = np.linalg.norm(P - Pref, axis=1)
print("RMS position error after first lap: %.3f m" % np.sqrt(np.mean(err[ts > 8] ** 2)))

# ------------------------------------------------ animate
fig = plt.figure(figsize=(6.4, 5.4), dpi=80)
fig.patch.set_facecolor(BG)
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor(BG)
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.set_pane_color((0.12, 0.16, 0.23, 1))
    a.label.set_color("#cbd5e1")
ax.tick_params(colors="#94a3b8")
ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_zlim(0, 2.5)
ax.set_xlabel("East (m)"); ax.set_ylabel("North (m)"); ax.set_zlabel("Alt (m)")
ax.set_title("Cascaded PID controller: 3D figure-eight", color="white", fontweight="bold")
ref_full = np.array([trajectory(t)[0] for t in np.linspace(0, 8, 300)])
ax.plot(ref_full[:, 1], ref_full[:, 0], -ref_full[:, 2], "--", color="#facc15", lw=1, alpha=0.7)
trail, = ax.plot([], [], [], color="#38bdf8", lw=2)
arm1, = ax.plot([], [], [], color="white", lw=3)
arm2, = ax.plot([], [], [], color="#f472b6", lw=3)
thrust, = ax.plot([], [], [], color="#22c55e", lw=1.5)
hud = fig.text(0.03, 0.04, "", color="white", family="monospace", fontsize=10,
               bbox=dict(fc=BG, ec="#334155"))
fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.93)

ARM = 0.35 * np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]) / np.sqrt(2)
STEP = 2
frames = list(range(0, len(hist), STEP))


def ned_to_plot(v):    # (N, E, D) -> (E, N, Up)
    return v[..., 1], v[..., 0], -v[..., 2]


def update(i):
    k = frames[i]
    t, p, pr, Rk, F = hist[k]
    lo = max(0, k - 120)
    x, y, z = ned_to_plot(P[lo:k + 1]); trail.set_data(x, y); trail.set_3d_properties(z)
    pts = p + ARM @ Rk.T * 1.6
    for line, (a, b) in ((arm1, (0, 1)), (arm2, (2, 3))):   # pink arm = front-left/rear-right diagonal
        x, y, z = ned_to_plot(pts[[a, b]]); line.set_data(x, y); line.set_3d_properties(z)
    tip = p + Rk @ np.array([0, 0, -0.12 * F.sum()])
    x, y, z = ned_to_plot(np.array([p, tip])); thrust.set_data(x, y); thrust.set_3d_properties(z)
    roll = np.degrees(np.arctan2(Rk[2, 1], Rk[2, 2])); pitch = np.degrees(-np.arcsin(Rk[2, 0]))
    hud.set_text(f"t={t:5.1f}s  err={err[k]:.2f} m\nroll={roll:+5.1f}  pitch={pitch:+5.1f} deg")
    ax.view_init(elev=24, azim=-60 + 40 * i / len(frames))
    return trail, arm1, arm2, hud


ani = FuncAnimation(fig, update, frames=len(frames))
ani.save(os.path.join(OUT, "quad_controller.gif"), writer=PillowWriter(fps=14))
print("frames:", len(frames))
