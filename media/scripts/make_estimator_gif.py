"""Generate ekf_estimator.gif.

Python port of the 7-state EKF in Stat_Flight_Estimation/Estimation_solution/
QuadEstimatorEKF.cpp (non-linear complementary attitude filter, IMU-driven
prediction with g'/RbgPrime, GPS and magnetometer updates) using the tuned
values from QuadEstimatorEKF.txt and the measured sensor noise from
06_SensorNoise.txt (GPS XY std 0.7 m, accel XY std 0.5 m/s^2).

Usage: python make_estimator_gif.py <out-dir>
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
rng = np.random.default_rng(7)
g = 9.81

# ------------------------------------------------ QuadEstimatorEKF.txt
QPosXYStd, QPosZStd, QVelXYStd, QVelZStd, QYawStd = 0.011, .05, 0.111, .1, .31
GPSPosXYStd, GPSPosZStd, GPSVelXYStd, GPSVelZStd = 1, 3, .1, .3
MagYawStd = .1
dtIMU, attitudeTau = 0.002, 100
# simulated sensor noise (06_SensorNoise.txt measured values)
ACC_STD, GYRO_STD, GPS_POS_STD, GPS_VEL_STD, MAG_STD = 0.5, 0.05, 0.7, 0.1, 0.1
DT_GPS, DT_MAG = 0.1, 0.04

Q = np.diag(np.array([QPosXYStd, QPosXYStd, QPosZStd, QVelXYStd, QVelXYStd, QVelZStd, QYawStd]) ** 2) * dtIMU
R_GPS = np.diag(np.array([GPSPosXYStd, GPSPosXYStd, GPSPosZStd, GPSVelXYStd, GPSVelXYStd, GPSVelZStd]) ** 2)
R_MAG = np.array([[MagYawStd ** 2]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def rot_bg(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                     [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                     [-sp, sr * cp, cr * cp]])


# ------------------------------------------------ EKF (port of QuadEstimatorEKF.cpp)
class EKF:
    def __init__(self):
        self.x = np.array([0, 0, -1, 0, 0, 0, 0], float)
        self.P = np.diag(np.array([.1, .1, .3, .1, .1, .3, .05]) ** 2)
        self.roll = self.pitch = 0.0

    def UpdateFromIMU(self, accel, gyro):
        phi, theta = self.roll, self.pitch
        rot = np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                        [0, np.cos(phi), -np.sin(phi)],
                        [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])
        ad = rot @ gyro
        pr, pp = phi + dtIMU * ad[0], theta + dtIMU * ad[1]
        self.x[6] = wrap(self.x[6] + dtIMU * ad[2])
        accelRoll = np.arctan2(accel[1], accel[2])
        accelPitch = np.arctan2(-accel[0], 9.81)
        k = attitudeTau / (attitudeTau + dtIMU)
        self.roll = k * pr + (1 - k) * accelRoll
        self.pitch = k * pp + (1 - k) * accelPitch

    def GetRbgPrime(self, roll, pitch, yaw):
        th, ph, ps = pitch, roll, yaw
        Rp = np.zeros((3, 3))
        Rp[0] = [-np.cos(th) * np.sin(ps), -np.sin(ph) * np.sin(th) * np.sin(ps) - np.cos(ph) * np.cos(ps),
                 -np.cos(ph) * np.sin(th) * np.sin(ps) + np.sin(ph) * np.cos(ps)]
        Rp[1] = [np.cos(th) * np.cos(ps), np.sin(ph) * np.sin(th) * np.cos(ps) - np.cos(ph) * np.sin(ps),
                 np.cos(ph) * np.sin(th) * np.cos(ps) + np.sin(ph) * np.sin(ps)]
        return Rp

    def Predict(self, dt, accel):
        # V3F Inertial_frame_acc = attitude.Rotate_BtoI(accel) - V3F(0, 0, CONST_GRAVITY)
        acc_i = rot_bg(self.roll, self.pitch, self.x[6]) @ accel - np.array([0, 0, g])
        new = self.x.copy()
        new[0:3] += self.x[3:6] * dt
        new[3:6] += acc_i * dt
        Rp = self.GetRbgPrime(self.roll, self.pitch, self.x[6])
        G = np.eye(7)
        G[0, 3] = G[1, 4] = G[2, 5] = dt
        G[3, 6] = (Rp[0] @ accel) * dt
        G[4, 6] = (Rp[1] @ accel) * dt
        self.P = G @ self.P @ G.T + Q
        self.x = new

    def Update(self, z, H, R, zFromX):
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - zFromX)
        self.P = (np.eye(7) - K @ H) @ self.P

    def UpdateFromGPS(self, pos, vel):
        H = np.zeros((6, 7)); H[:, :6] = np.eye(6)
        self.Update(np.r_[pos, vel], H, R_GPS, self.x[:6].copy())

    def UpdateFromMag(self, magYaw):
        H = np.zeros((1, 7)); H[0, 6] = 1
        zFromX = self.x[6]
        if magYaw - zFromX > np.pi:
            zFromX += 2 * np.pi
        elif magYaw - zFromX < -np.pi:
            zFromX -= 2 * np.pi
        self.Update(np.array([magYaw]), H, R_MAG, np.array([zFromX]))


# ------------------------------------------------ truth: smooth 3D loop with yaw rotation
W = 2 * np.pi / 10


def truth(t):
    p = np.array([3 * np.sin(W * t), 3 * np.cos(W * t) - 3, -1 - 0.5 * np.sin(2 * W * t)])
    v = np.array([3 * W * np.cos(W * t), -3 * W * np.sin(W * t), -W * np.cos(2 * W * t)])
    a = np.array([-3 * W**2 * np.sin(W * t), -3 * W**2 * np.cos(W * t), 2 * W**2 * np.sin(2 * W * t)])
    yaw = wrap(0.6 * t)
    return p, v, a, yaw


def attitude(a, yaw):
    """Flat-output attitude: body z aligned with required thrust."""
    f = np.array([0, 0, g]) - a               # specific force direction (NED, points 'up' = -z)
    zb = -f / np.linalg.norm(f)
    xc = np.array([np.cos(yaw), np.sin(yaw), 0])
    yb = np.cross(zb, xc); yb /= np.linalg.norm(yb)
    xb = np.cross(yb, zb)
    return np.column_stack([xb, yb, zb])


T_END = 20.0
N = int(T_END / dtIMU)
ekf = EKF()
R_prev = attitude(truth(0)[2], truth(0)[3])
# start the attitude filter at the true initial attitude (as the sim does at hover)
ekf.roll = np.arctan2(R_prev[2, 1], R_prev[2, 2]); ekf.pitch = -np.arcsin(R_prev[2, 0])
gps_every, mag_every = int(DT_GPS / dtIMU), int(DT_MAG / dtIMU)
hist, gps_pts = [], []
for k in range(1, N):
    t = k * dtIMU
    p, v, a, yaw = truth(t)
    Rk = attitude(a, yaw)
    w_skew = R_prev.T @ (Rk - R_prev) / dtIMU
    gyro = np.array([w_skew[2, 1], w_skew[0, 2], w_skew[1, 0]]) + rng.normal(0, GYRO_STD, 3)
    # simulator IMU convention matching the project's PredictState (reads +g at rest)
    accel = Rk.T @ (a + np.array([0, 0, g])) + rng.normal(0, [ACC_STD, ACC_STD, ACC_STD * 2])
    R_prev = Rk
    ekf.UpdateFromIMU(accel, gyro)
    ekf.Predict(dtIMU, accel)
    if k % gps_every == 0:
        gp = p + rng.normal(0, [GPS_POS_STD, GPS_POS_STD, GPS_POS_STD * 2])
        gv = v + rng.normal(0, [GPS_VEL_STD, GPS_VEL_STD, GPS_VEL_STD * 3])
        ekf.UpdateFromGPS(gp, gv)
        gps_pts.append((t, gp.copy()))
    if k % mag_every == 0:
        ekf.UpdateFromMag(wrap(yaw + rng.normal(0, MAG_STD)))
    if k % 20 == 0:
        hist.append((t, p.copy(), ekf.x.copy(), np.sqrt(np.diag(ekf.P)), yaw))

ts = np.array([h[0] for h in hist])
TP = np.array([h[1] for h in hist])
EX = np.array([h[2] for h in hist])
SIG = np.array([h[3] for h in hist])
YAW = np.array([h[4] for h in hist])
GT = np.array([g_[0] for g_ in gps_pts])
GP = np.array([g_[1] for g_ in gps_pts])
gps_err = np.linalg.norm(GP[:, :2] - np.array([truth(t)[0][:2] for t in GT]), axis=1)
ekf_err = np.linalg.norm(EX[:, :2] - TP[:, :2], axis=1)
print("XY RMS: GPS %.2f m  ->  EKF %.2f m" % (np.sqrt(np.mean(gps_err ** 2)), np.sqrt(np.mean(ekf_err[ts > 2] ** 2))))

# ------------------------------------------------ animate
plt.rcParams.update({"font.size": 9})
fig = plt.figure(figsize=(8.4, 4.6), dpi=80)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.45, wspace=0.28,
                      left=0.07, right=0.98, top=0.88, bottom=0.11)
axm, axx, axy = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
for a in (axm, axx, axy):
    a.set_facecolor("#1e293b"); a.tick_params(colors="#94a3b8")
    for s in a.spines.values():
        s.set_color("#334155")
fig.suptitle("Extended Kalman Filter: IMU + GPS + magnetometer fusion", color="white", fontweight="bold")

axm.set_title("top-down position", color="#cbd5e1")
axm.set_xlim(-7.5, 1.5); axm.set_ylim(-4.6, 4.2); axm.set_aspect("equal")
axm.set_xlabel("East (m)", color="#cbd5e1"); axm.set_ylabel("North (m)", color="#cbd5e1")
gps_sc = axm.scatter([], [], s=10, color="#fb923c", alpha=0.6, label="GPS (noisy, 10 Hz)")
true_l, = axm.plot([], [], color="white", lw=1.5, label="ground truth")
est_l, = axm.plot([], [], color="#38bdf8", lw=2, label="EKF estimate")
axm.legend(loc="center", facecolor=BG, edgecolor="#334155", labelcolor="white", fontsize=8)

axx.set_title("North position error (m)", color="#cbd5e1")
axx.set_xlim(0, T_END); axx.set_ylim(-2.2, 2.2)
axy.set_title("yaw error (deg)", color="#cbd5e1")
axy.set_xlim(0, T_END); axy.set_ylim(-15, 15); axy.set_xlabel("time (s)", color="#cbd5e1")
gx_sc = axx.scatter([], [], s=6, color="#fb923c", alpha=0.6)
ex_l, = axx.plot([], [], color="#38bdf8", lw=1.5)
yaw_l, = axy.plot([], [], color="#38bdf8", lw=1.5)
bands = []

STEP = 5
frames = list(range(0, len(hist), STEP)) + [len(hist) - 1] * 12


def update(i):
    k = frames[i]
    t = ts[k]
    m = GT <= t
    gps_sc.set_offsets(np.c_[GP[m, 1], GP[m, 0]] if m.any() else np.empty((0, 2)))
    true_l.set_data(TP[:k + 1, 1], TP[:k + 1, 0])
    est_l.set_data(EX[:k + 1, 1], EX[:k + 1, 0])
    gerr = GP[m, 0] - np.array([truth(tt)[0][0] for tt in GT[m]]) if m.any() else np.array([])
    gx_sc.set_offsets(np.c_[GT[m], gerr] if m.any() else np.empty((0, 2)))
    ex_l.set_data(ts[:k + 1], EX[:k + 1, 0] - TP[:k + 1, 0])
    yerr = np.degrees(wrap(EX[:k + 1, 6] - YAW[:k + 1]))
    yaw_l.set_data(ts[:k + 1], yerr)
    for b in bands:
        b.remove()
    bands.clear()
    bands.append(axx.fill_between(ts[:k + 1], -2 * SIG[:k + 1, 0], 2 * SIG[:k + 1, 0], color="#38bdf8", alpha=0.15, lw=0))
    bands.append(axy.fill_between(ts[:k + 1], -2 * np.degrees(SIG[:k + 1, 6]), 2 * np.degrees(SIG[:k + 1, 6]),
                                  color="#38bdf8", alpha=0.15, lw=0))
    return gps_sc, true_l, est_l


ani = FuncAnimation(fig, update, frames=len(frames))
ani.save(os.path.join(OUT, "ekf_estimator.gif"), writer=PillowWriter(fps=12))
print("frames:", len(frames))
