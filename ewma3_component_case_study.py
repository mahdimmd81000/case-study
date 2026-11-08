'''Six component-wise EWMA3 control charts for the transformer case study.

Creates three Phase-II component charts for the observed in-control series and
three more for an intercept-only step from day 91. Components: intercept, slope,
and residual standard deviation (variance direction). Each component gets its
own two-sided bootstrap control limit; no alarms are forced or hidden.
'''
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "case_study_transformer_data.csv"
OUT = ROOT / "outputs" / "ewma3_component_case_study"
SEED = 20260825
PHASE_I_N = 30
SHIFT_START_DAY = 91
LAMBDA = 0.20
ARL0 = 200.0
N_BOOT = 20_000

def fit_phase1(t: np.ndarray, r: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(t[:PHASE_I_N], r[:PHASE_I_N], 1)
    residual = r - (intercept + slope * t)
    tbar = float(np.mean(t[:PHASE_I_N]))
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "sigma": float(np.std(residual[:PHASE_I_N], ddof=1)),
        "tbar": tbar,
        "sxx": float(np.sum((t[:PHASE_I_N] - tbar) ** 2)),
    }


def ewma(values: np.ndarray) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    z = 0.0
    for i, value in enumerate(np.asarray(values, dtype=float)):
        z = LAMBDA * value + (1.0 - LAMBDA) * z
        out[i] = z
    return out

def ewma_sd(k: int) -> float:
    return float(np.sqrt(LAMBDA / (2.0 - LAMBDA) *
                         (1.0 - (1.0 - LAMBDA) ** (2 * k))))


def slope_ewma_sd(t: np.ndarray, ref: dict[str, float]) -> np.ndarray:
    """Exact conditional SD path for EWMA of temperature-weighted residuals."""
    v = ((t - ref["tbar"]) ** 2) / ref["sxx"]
    out = np.empty(len(t), dtype=float)
    variance = 0.0
    for i, value in enumerate(v):
        variance = LAMBDA**2 * value + (1.0 - LAMBDA)**2 * variance
        out[i] = np.sqrt(variance)
    return out

def component_statistics(t: np.ndarray, r: np.ndarray, ref: dict[str, float]) -> dict[str, np.ndarray]:
    """Absolute standardised component EWMAs, warmed through Phase I."""
    residual = (r - (ref["intercept"] + ref["slope"] * t)) / ref["sigma"]
    q_intercept = residual
    q_slope = residual * (t - ref["tbar"]) / np.sqrt(ref["sxx"])
    q_scale = (residual**2 - 1.0) / np.sqrt(2.0)
    z_intercept, z_slope, z_scale = ewma(q_intercept), ewma(q_slope), ewma(q_scale)
    sd_intercept = np.array([ewma_sd(k) for k in range(1, len(t) + 1)])
    sd_slope = slope_ewma_sd(t, ref)
    phase2 = slice(PHASE_I_N, len(t))
    return {
        "Intercept": np.abs(z_intercept[phase2]) / sd_intercept[phase2],
        "Slope": np.abs(z_slope[phase2]) / sd_slope[phase2],
        "Residual SD": np.abs(z_scale[phase2]) / sd_intercept[phase2],
    }


def bootstrap_component_limits(t: np.ndarray, ref: dict[str, float]) -> dict[str, float]:
    """Startup-safe component limits from independent in-control paths.

    Each limit is the maximum across Phase-II days of that day's 99.5th
    percentile (target ARL0=200). The limit is calibrated only from the
    specified parametric in-control model; observed values are not used to
    tune it.
    """
    rng = np.random.default_rng(SEED)
    n = len(t)
    n_phase2 = n - PHASE_I_N
    names = ("Intercept", "Slope", "Residual SD")
    boot = {name: np.empty((N_BOOT, n_phase2), dtype=np.float32) for name in names}
    sd_intercept = np.array([ewma_sd(k) for k in range(1, n + 1)])
    sd_slope = slope_ewma_sd(t, ref)
    chunk = 500
    for start in range(0, N_BOOT, chunk):
        b = min(chunk, N_BOOT - start)
        errors = rng.normal(0.0, ref["sigma"], size=(b, n))
        q_intercept = errors / ref["sigma"]
        q_slope = q_intercept * (t - ref["tbar"]) / np.sqrt(ref["sxx"])
        q_scale = (q_intercept**2 - 1.0) / np.sqrt(2.0)
        z0 = np.zeros(b)
        z1 = np.zeros(b)
        zv = np.zeros(b)
        for k in range(n):
            z0 = LAMBDA * q_intercept[:, k] + (1.0 - LAMBDA) * z0
            z1 = LAMBDA * q_slope[:, k] + (1.0 - LAMBDA) * z1
            zv = LAMBDA * q_scale[:, k] + (1.0 - LAMBDA) * zv
            if k >= PHASE_I_N:
                j = k - PHASE_I_N
                boot["Intercept"][start:start+b, j] = np.abs(z0) / sd_intercept[k]
                boot["Slope"][start:start+b, j] = np.abs(z1) / sd_slope[k]
                boot["Residual SD"][start:start+b, j] = np.abs(zv) / sd_intercept[k]

    quantile = 1.0 - 1.0 / ARL0
    return {name: float(np.max(np.quantile(values, quantile, axis=0)))
            for name, values in boot.items()}


def shifted_intercept_series(t: np.ndarray, r: np.ndarray,
                             ref: dict[str, float]) -> np.ndarray:
    """Add a sustained +1 Phase-I-sigma intercept shift starting on day 91."""
    shifted = r.copy()
    shifted[SHIFT_START_DAY - 1:] += ref["sigma"]
    return shifted

def plot_component(path: Path, days: np.ndarray, values: np.ndarray, ucl: float,
                   component: str, shifted: bool) -> None:
    fig, ax = plt.subplots(figsize=(16, 5.5))
    fig.patch.set_facecolor("#dcdcdc")
    ax.set_facecolor((0.94, 0.94, 0.94))
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)
    color = "#48c97a" if component == "Intercept" else "#4f81bd"
    ax.plot(days, values, color=color, marker="o", markersize=4.5,
            linewidth=1.8, label=f"EWMA3 {component}")
    ax.axhline(ucl, color="red", linestyle="--", linewidth=2.6,
               label=f"UCL = {ucl:.3f}")
    ax.axhline(float(np.mean(values)), color="blue", linestyle=":", linewidth=1.7,
               label=f"Mean = {np.mean(values):.3f}")
    ax.fill_between(days, 0, ucl, color="#8fbc8f", alpha=0.22,
                    label="In-Control Region")
    if shifted:
        ax.axvline(SHIFT_START_DAY - 0.5, color="black", linestyle="--",
                   linewidth=1.6, label=f"Intercept shift from day {SHIFT_START_DAY}")
    signal_idx = np.flatnonzero(values > ucl)
    if len(signal_idx):
        ax.scatter(days[signal_idx], values[signal_idx], s=42, color="#d62728",
                   zorder=5, label=f"Signals: {len(signal_idx)}")
    state = "Intercept Shift from Day 91" if shifted else "In Control"
    ax.set_title(f"EWMA3 {component} Component — {state}", fontsize=17,
                 fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Absolute standardised EWMA statistic")
    ax.set_xlim(days[0] - 1, days[-1] + 1)
    ax.set_ylim(0, max(ucl * 1.15, float(values.max()) * 1.1))
    # Keep the legend clear of the series while placing it on the left side.
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.5),
              fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    t = frame["T_avg"].to_numpy(float)
    r = frame["R_avg"].to_numpy(float)
    if len(t) != 150:
        raise ValueError(f"Expected 150 days; got {len(t)}")

    days = np.arange(PHASE_I_N + 1, len(t) + 1)
    ref = fit_phase1(t, r)
    limits = bootstrap_component_limits(t, ref)
    in_control = component_statistics(t, r, ref)
    shifted = component_statistics(t, shifted_intercept_series(t, r, ref), ref)

    pre_shift = days < SHIFT_START_DAY
    for name in in_control:
        if not np.allclose(in_control[name][pre_shift], shifted[name][pre_shift], atol=1e-12):
            raise RuntimeError(f"Pre-shift values differ for component {name}")

    in_control_alarms = {name: int(np.sum(in_control[name] > limits[name]))
                         for name in in_control}
    shift_alarms = {name: int(np.sum((days >= SHIFT_START_DAY) &
                                     (shifted[name] > limits[name])))
                    for name in shifted}

    for name in in_control:
        stem = name.lower().replace(" ", "_")
        plot_component(OUT / f"ewma3_{stem}_incontrol.png", days,
                       in_control[name], limits[name], name, False)
        plot_component(OUT / f"ewma3_{stem}_intercept_shift.png", days,
                       shifted[name], limits[name], name, True)

    rows = []
    for name in in_control:
        signal_days = days[(days >= SHIFT_START_DAY) & (shifted[name] > limits[name])]
        rows.append({"component": name, "ucl": limits[name],
                     "incontrol_alarms": in_control_alarms[name],
                     "shift_alarms": shift_alarms[name],
                     "first_shift_signal_day": int(signal_days[0]) if len(signal_days) else None,
                     "shift_days_above_ucl": ";".join(map(str, signal_days.tolist()))})

    pd.DataFrame(rows).to_csv(OUT / "ewma3_component_summary.csv", index=False)
    metadata = {"limits": limits, "incontrol_alarms": in_control_alarms,
                "shift_alarms": shift_alarms, "shift_start_day": SHIFT_START_DAY,
                "intercept_shift_sigma": ref["sigma"], "lambda": LAMBDA,
                "arl0_target": ARL0, "bootstrap_paths": N_BOOT}
    (OUT / "ewma3_component_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("Six component figures written to:", OUT)

if __name__ == "__main__":
    main()












