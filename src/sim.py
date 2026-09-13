"""二维恒速目标 + 多传感器观测仿真,含三种传感器退化模式。

设计原则
--------
* 只用 numpy,无其他依赖,`python src/run_experiment.py` 一条命令跑通;
* 退化模式显式建模,便于研究"退化如何破坏不确定性的可信度";
* 随机种子可复现。

状态定义
--------
x = [px, py, vx, vy]  (m, m, m/s, m/s)
"""
from __future__ import annotations

import numpy as np

DT = 0.1          # 采样周期 (s)
STEPS = 300       # 每条约 30 s
SIGMA_A = 0.5     # 过程加速度噪声 (m/s^2)

# 观测噪声(标称)
SIGMA_GPS = 2.0   # GPS 位置 (m)
SIGMA_VIS = 1.0   # 视觉定位 (m)
SIGMA_IMU = 0.3   # IMU 速度 (m/s)

# 退化参数
SIGMA_VIS_FOG = 3.0     # 雾/雨下视觉噪声放大
OUTLIER_P = 0.10        # 退化时视觉外点比例
OUTLIER_SIGMA = 8.0     # 外点尺度 (m)

MODES = ("nominal", "fog", "gps_outage", "both")


def F() -> np.ndarray:
    """状态转移矩阵。"""
    return np.array(
        [[1.0, 0.0, DT, 0.0],
         [0.0, 1.0, 0.0, DT],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]]
    )


def Q() -> np.ndarray:
    """过程噪声协方差(连续白噪声加速度模型离散化)。"""
    g = np.array([[DT ** 2 / 2, 0.0],
                  [0.0, DT ** 2 / 2],
                  [DT, 0.0],
                  [0.0, DT]])
    return g @ g.T * SIGMA_A ** 2


def is_gps_available(mode: str) -> bool:
    return mode not in ("gps_outage", "both")


def is_vision_degraded(mode: str) -> bool:
    """真实退化状态(滤波器默认不知道,需要检测器;见 NOTES.md)。"""
    return mode in ("fog", "both")


def simulate(seed: int, mode: str):
    """返回 (真实轨迹 [STEPS+1, 4], 观测序列 list[dict])。"""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    rng = np.random.default_rng(seed)
    f, q = F(), Q()

    x = np.array([0.0, 0.0, 8.0, 3.0])
    traj = [x.copy()]
    obs = []

    for _ in range(STEPS):
        x = f @ x + rng.multivariate_normal(np.zeros(4), q)
        traj.append(x.copy())

        z = {"imu": x[2:] + rng.normal(0.0, SIGMA_IMU, 2)}

        if is_gps_available(mode):
            z["gps"] = x[:2] + rng.normal(0.0, SIGMA_GPS, 2)

        sigma = SIGMA_VIS_FOG if is_vision_degraded(mode) else SIGMA_VIS
        z_vis = x[:2] + rng.normal(0.0, sigma, 2)
        if is_vision_degraded(mode) and rng.random() < OUTLIER_P:
            z_vis = z_vis + rng.normal(0.0, OUTLIER_SIGMA, 2)
        z["vis"] = z_vis

        obs.append(z)

    return np.array(traj), obs
