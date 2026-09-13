# uq-state-estimation

**Sensor degradation breaks the *trustworthiness* of state estimation — not just its accuracy.**

A minimal, reproducible experiment on **uncertainty quantification (UQ) under sensor degradation**:
two Kalman filters see the same data; they differ only in **whether degradation is reflected in the
reported covariance**. Accuracy barely differs in nominal conditions — but the *covariance* does.

> 中文摘要:同一个卡尔曼滤波器,在**传感器退化时**如果把退化如实反映到协方差里,不确定性的可信度就守住了;
> 不反映,则精度或许还行,**但它报告的置信区间已经不可信**——这正是"可信状态估计"的核心问题。

---

## 1. Question

When a sensor degrades (fog on the camera, GPS outage), a filter can still produce a
*plausible-looking* estimate. The failure is silent: the estimate keeps coming, but the reported
uncertainty no longer covers the truth.

**Q: How much does ignoring degradation damage the reliability of the reported uncertainty — separately
from its effect on accuracy?**

## 2. Setup

2D constant-velocity target, `dt = 0.1 s`, 300 steps per run, 20 random seeds.

| Sensor | Measures | Nominal σ | Degraded |
|---|---|---|---|
| GPS | position | 2.0 m | **unavailable** (outage) |
| Vision | position | 1.0 m | **3.0 m + 10% outliers (8 m)** (fog) |
| IMU | velocity | 0.3 m/s | unchanged |

Four modes: `nominal`, `fog`, `gps_outage`, `both`.

Two filters, **identical except for the covariance口径**:

- `static` — always uses nominal σ; it does not know it has degraded
- `adaptive` — inflates vision σ when degraded, and gates measurements by Mahalanobis distance

Both share the same motion model, the same data, and the same initialization.

## 3. Results

Position error and uncertainty reliability (20 seeds, 30-step warmup):

| mode | filter | RMSE (m) | coverage@95% | NEES (ideal 2.0) | NLL |
|---|---|---|---|---|---|
| nominal | static | 0.232 | **0.936** | 2.13 | −0.74 |
| nominal | adaptive | 0.233 | **0.935** | 2.16 | −0.72 |
| fog | static | 0.622 | **0.347** ❌ | 15.37 | 5.88 |
| fog | adaptive | 0.315 | **0.943** ✅ | 1.92 | −0.15 |
| gps_outage | static | 0.246 | 0.953 | 2.09 | −0.64 |
| gps_outage | adaptive | 0.246 | 0.955 | 2.08 | −0.64 |
| both | static | 0.755 | **0.252** ❌ | 19.93 | 8.28 |
| both | adaptive | 1.346 | **0.861** ✅ | 8.26 | 3.90 |

**Three observations:**

1. **In nominal conditions the two are indistinguishable** (0.232 vs 0.233 m; coverage 0.936 vs 0.935).
   Reliability only diverges when the world misbehaves — so this failure is invisible in clean benchmarks.
2. **Under fog, the naive filter is catastrophically overconfident**: it claims 95% coverage and delivers
   **34.7%**; NEES 15.4 vs the ideal 2.0. Its RMSE is also 2× worse.
3. **In the hardest mode (`both`) the trade-off appears**: the adaptive filter has *worse* RMSE
   (1.346 vs 0.755 m) yet much *better* calibration (0.861 vs 0.252) and NEES 8.26 vs 19.93.
   With only a degraded camera left, you cannot be both accurate and honest — **you can only choose
   whether your uncertainty tells the truth**. That choice is what makes a system safe to deploy.

Reliability curves: [`results/reliability_nominal.svg`](results/reliability_nominal.svg) ·
[`results/reliability_both.svg`](results/reliability_both.svg)

## 4. Reproduce

```bash
pip install -r requirements.txt
python src/run_experiment.py            # ~10 s, writes results/
python src/run_experiment.py --seeds 5  # quick run
```

Output: `results/metrics.csv`, `results/reliability_*.svg`

## 5. Limitations (explicit)

- **Synthetic data only.** No real sensor logs; the degradation model (noise inflation + outliers) is
  hand-specified, not learned.
- **The degradation indicator is given (oracle).** `adaptive` receives the true degraded flag.
  In reality it must be *detected* — that detection problem is the next step, and it is itself an
  uncertainty problem.
- **Linear-Gaussian only.** No nonlinear geometry, no correlated noise, no bias states.
- **Gating is a heuristic.** Thresholds are χ² quantiles chosen by hand; no ablation yet.
- **No comparison to learned estimators**, and no metrics beyond RMSE / coverage / NEES / NLL.

## 6. Next steps

- [ ] Replace the oracle flag with a **learned degradation detector**, and report detector-calibration jointly
- [ ] Add **bias states** (fog causes bias, not just variance) — miscalibration that inflates with time
- [ ] Nonlinear case: range-bearing sensors / EKF vs UKF vs particle filter under the same protocol
- [ ] Learned estimator (small NN) with **calibrated** predictive distribution, same evaluation harness
- [ ] Swap the simulator for a public dataset or a simple robot (real degradation, real noise)

See [`NOTES.md`](NOTES.md) for the reasoning behind these choices.

## License

MIT
