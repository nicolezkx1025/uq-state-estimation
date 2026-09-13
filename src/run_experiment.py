"""实验入口:在四种传感器退化模式下,比较两类滤波器的不确定性可信度。

用法:
    python src/run_experiment.py            # 默认 20 个随机种子
    python src/run_experiment.py --seeds 5  # 快速跑
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim import MODES, STEPS, simulate, is_vision_degraded
from src.kf import StaticKF, AdaptiveKF
from src.metrics import NOMINAL_LEVELS, chi2_df2_quantile, summarize
from src.plot_svg import reliability_svg

FILTERS = {"static": StaticKF, "adaptive": AdaptiveKF}
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def init_filter(cls, first_obs: dict):
    """用首个位置观测初始化,速度先验放宽。"""
    kf = cls()
    z0 = first_obs.get("gps", first_obs["vis"])
    kf.x[:2] = z0
    kf.P = np.diag([4.0, 4.0, 25.0, 25.0])
    return kf


def run(seeds: int = 20, warmup: int = 30):
    rows = []
    # 可靠性曲线累积:{mode: {filter: {level: [empirical]}}}
    acc = {m: {f: {l: [] for l in NOMINAL_LEVELS} for f in FILTERS} for m in MODES}

    for mode in MODES:
        for fname, cls in FILTERS.items():
            per_seed = []
            for seed in range(seeds):
                traj, obs = simulate(seed, mode)
                kf = init_filter(cls, obs[0])
                est = []
                for z in obs:
                    est.append(kf.step(z, degraded=is_vision_degraded(mode)))
                s = summarize(traj, est, warmup=warmup)
                per_seed.append(s)
                for l in NOMINAL_LEVELS:
                    acc[mode][fname][l].append(s["levels"][str(l)])
            agg = {
                "mode": mode,
                "filter": fname,
                "rmse_pos_m": round(float(np.mean([p["rmse_pos_m"] for p in per_seed])), 3),
                "coverage_95": round(float(np.mean([p["coverage_95"] for p in per_seed])), 3),
                "nees_pos_mean": round(float(np.mean([p["nees_pos_mean"] for p in per_seed])), 3),
                "nll_mean": round(float(np.mean([p["nll_mean"] for p in per_seed])), 3),
            }
            rows.append(agg)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) 指标表
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 2) 可靠性曲线(按模式画两张:标称 vs 退化最严重)
    curves_for = ["nominal", "both"]
    for mode in curves_for:
        curves = {}
        for fname in FILTERS:
            curves[fname] = [
                (l, float(np.mean(acc[mode][fname][l]))) for l in NOMINAL_LEVELS
            ]
        reliability_svg(
            curves,
            os.path.join(RESULTS_DIR, f"reliability_{mode}.svg"),
            f"Uncertainty reliability — mode: {mode}",
        )

    # 3) 控制台汇总
    print(f"seeds={seeds}  warmup={warmup}")
    print(f"{'mode':<11}{'filter':<10}{'RMSE(m)':>9}{'cov@95':>9}{'NEES':>8}{'NLL':>9}")
    for r in rows:
        print(f"{r['mode']:<11}{r['filter']:<10}{r['rmse_pos_m']:>9.3f}"
              f"{r['coverage_95']:>9.3f}{r['nees_pos_mean']:>8.2f}{r['nll_mean']:>9.2f}")
    print(f"\nwrote {csv_path}")
    for mode in curves_for:
        print("wrote", os.path.join(RESULTS_DIR, f"reliability_{mode}.svg"))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()
    run(seeds=args.seeds, warmup=args.warmup)
