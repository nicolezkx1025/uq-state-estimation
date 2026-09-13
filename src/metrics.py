"""评估指标:精度 + **不确定性是否可信**(本项目重点)。"""
from __future__ import annotations

import numpy as np

# df=2 时卡方分布的分位数有解析式:CDF(x) = 1 - exp(-x/2)  =>  x = -2 ln(1 - p)
NOMINAL_LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95, 0.99)


def chi2_df2_quantile(p: float) -> float:
    return -2.0 * np.log(1.0 - p)


def position_error_stats(traj, est, warmup: int = 30):
    """返回每步的位置误差 e (2,) 与其协方差 P_pos,均为 numpy 数组。"""
    errs, covs = [], []
    for k in range(warmup, len(est)):
        x_hat, P = est[k]
        e = traj[k + 1][:2] - x_hat[:2]
        errs.append(e)
        covs.append(P[:2, :2])
    return np.array(errs), np.array(covs)


def rmse_position(errs: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(errs ** 2, axis=1))))


def coverage(errs: np.ndarray, covs: np.ndarray, level: float) -> float:
    """真值落在 level 置信椭圆内的比例(理想值 = level)。"""
    thr = chi2_df2_quantile(level)
    inside = 0
    for e, P in zip(errs, covs):
        d2 = float(e @ np.linalg.solve(P, e))
        inside += int(d2 <= thr)
    return inside / len(errs)


def mean_nees(errs: np.ndarray, covs: np.ndarray) -> float:
    """位置维度的 NEES 均值,理想值 = 2(自由度)。>2 表示过度自信。"""
    vals = [float(e @ np.linalg.solve(P, e)) for e, P in zip(errs, covs)]
    return float(np.mean(vals))


def mean_nll(errs: np.ndarray, covs: np.ndarray) -> float:
    """真值在估计高斯分布下的平均负对数似然(越小越好)。"""
    out = []
    for e, P in zip(errs, covs):
        sign, logdet = np.linalg.slogdet(2 * np.pi * P)
        out.append(0.5 * (logdet + float(e @ np.linalg.solve(P, e))))
    return float(np.mean(out))


def summarize(traj, est, warmup: int = 30) -> dict:
    errs, covs = position_error_stats(traj, est, warmup)
    return {
        "rmse_pos_m": round(rmse_position(errs), 3),
        "coverage_95": round(coverage(errs, covs, 0.95), 3),
        "nees_pos_mean": round(mean_nees(errs, covs), 3),
        "nll_mean": round(mean_nll(errs, covs), 3),
        "levels": {str(l): round(coverage(errs, covs, l), 3) for l in NOMINAL_LEVELS},
    }
