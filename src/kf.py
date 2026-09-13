"""线性卡尔曼滤波器,两种"不确定性口径"。

* StaticKF   : 始终使用标称观测噪声——退化发生时它**不知道自己变差了**,协方差偏乐观;
* AdaptiveKF : 依据退化指示量放大观测噪声,并对测量做马氏距离门控以剔除外点。

两者的**状态估计精度**差别不大,真正的差别在**协方差是否可信**——这正是本项目要量化的对象。
"""
from __future__ import annotations

import numpy as np

from .sim import (
    DT, F, Q,
    SIGMA_GPS, SIGMA_VIS, SIGMA_IMU,
    SIGMA_VIS_FOG, OUTLIER_SIGMA,
)


class _BaseKF:
    name = "base"

    def __init__(self, x0: np.ndarray | None = None):
        self.x = np.zeros(4) if x0 is None else x0.astype(float).copy()
        self.P = np.eye(4) * 1.0
        self.f = F()
        self.q = Q()

    # ---- 内部工具 ----
    @staticmethod
    def _H_pos() -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0]])

    @staticmethod
    def _H_vel() -> np.ndarray:
        return np.array([[0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

    def _predict(self):
        self.x = self.f @ self.x
        self.P = self.f @ self.P @ self.f.T + self.q

    def _update(self, H, z, R, gate_threshold: float | None = None) -> bool:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        if gate_threshold is not None:
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 > gate_threshold:
                return False                      # 判为外点,丢弃
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T   # Joseph 形式,保持对称正定
        return True

    # ---- 对外接口 ----
    def step(self, z: dict, degraded: bool = False):
        self._predict()
        self._update(self._H_vel(), z["imu"], np.eye(2) * SIGMA_IMU ** 2)
        if "gps" in z:
            self._update(self._H_pos(), z["gps"], np.eye(2) * self.r_gps() ** 2)
        if "vis" in z:
            self._update(self._H_pos(), z["vis"], np.eye(2) * self.r_vis(degraded) ** 2,
                         gate_threshold=self.gate(degraded))
        return self.x.copy(), self.P.copy()

    def r_gps(self) -> float:
        return SIGMA_GPS

    def r_vis(self, degraded: bool) -> float:
        return SIGMA_VIS

    def gate(self, degraded: bool):
        return None


class StaticKF(_BaseKF):
    """不知道退化的滤波器:噪声参数写死为标称值。"""
    name = "static"


class AdaptiveKF(_BaseKF):
    """显式建模退化的滤波器:退化时放大 R,并门控外点。"""
    name = "adaptive"

    def r_vis(self, degraded: bool) -> float:
        return SIGMA_VIS_FOG if degraded else SIGMA_VIS

    def gate(self, degraded: bool):
        # 卡方分布 df=2、99.9% 分位 ≈ 13.8;外点尺度(8 m)远大于此
        return 13.8 if degraded else 9.21      # 9.21 = df=2 的 99% 分位


def run_filter(kf: _BaseKF, obs: list[dict], degraded: bool):
    """返回每步的 (x_hat, P)。"""
    est = []
    for z in obs:
        x_hat, P = kf.step(z, degraded=degraded)
        est.append((x_hat, P))
    return est
