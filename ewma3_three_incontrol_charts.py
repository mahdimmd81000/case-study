   # Three shifted EWMA3 component charts for the case study.
# This file is self-contained so it does not depend on the unfinished shift-stage script.
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
OUT = ROOT / "outputs" / "ewma3_component_final_visible"
PHASE_I_N = 30
SHIFT_DAY = 91
SHIFT_DELTA = 0.00008
LAMBDA = 0.20
ARL0 = 200.0
N_BOOT = 20_000
SEED = 20260825

def fit_reference(t, r):
    slope, intercept = np.polyfit(t[:PHASE_I_N], r[:PHASE_I_N], 1)
    err = r - (intercept + slope * t)
    tbar = float(t[:PHASE_I_N].mean())
    return {"a0": float(intercept), "a1": float(slope),
            "sigma": float(err[:PHASE_I_N].std(ddof=1)), "tbar": tbar,
            "sxx": float(np.sum((t[:PHASE_I_N] - tbar) ** 2))}


def ewma_sd(k):
    return np.sqrt(LAMBDA / (2 - LAMBDA) * (1 - (1 - LAMBDA) ** (2 * k)))


def slope_sd_path(t, ref):
    v = (t - ref["tbar"]) ** 2 / ref["sxx"]
    out = np.empty(len(t)); variance = 0.0
    for i, x in enumerate(v):
        variance = LAMBDA**2 * x + (1 - LAMBDA)**2 * variance
        out[i] = np.sqrt(variance)
    return out

def ewma_path(x):
    out = np.empty(len(x)); z = 0.0
    for i, value in enumerate(x):
        z = LAMBDA * value + (1 - LAMBDA) * z
        out[i] = z
    return out

def component_stats(t, r, ref, *, signed=False):
    """EWMA3 components; signed mode preserves direction for shift diagnosis."""
    e = (r - (ref["a0"] + ref["a1"] * t)) / ref["sigma"]
    zi = ewma_path(e)
    zs = ewma_path(e * (t - ref["tbar"]) / np.sqrt(ref["sxx"]))
    zv = ewma_path((e**2 - 1) / np.sqrt(2))
    sd_i = np.array([ewma_sd(k) for k in range(1, len(t) + 1)])
    sd_s = slope_sd_path(t, ref)
    sl = slice(PHASE_I_N, len(t))
    if signed:
        return {"Intercept": zi[sl] / sd_i[sl],
                "Slope": zs[sl] / sd_s[sl],
                "Residual SD": zv[sl] / sd_i[sl]}
    return {"Intercept": zi[sl] / sd_i[sl],
            "Slope": np.abs(zs[sl]) / sd_s[sl],
            "Residual SD": np.abs(zv[sl]) / sd_i[sl]}


def bootstrap_limits(t, ref, signed=False):
    rng = np.random.default_rng(SEED); n = len(t); h = n - PHASE_I_N
    sd_i = np.array([ewma_sd(k) for k in range(1, n + 1)])
    sd_s = slope_sd_path(t, ref)
    paths = {k: np.empty((N_BOOT, h), dtype=np.float32)
             for k in ("Intercept", "Slope", "Residual SD")}
    chunk = 500
    for first in range(0, N_BOOT, chunk):
        b = min(chunk, N_BOOT-first)
        e = rng.normal(0, ref["sigma"], size=(b, n)) / ref["sigma"]
        qi = e; qs = e * (t-ref["tbar"]) / np.sqrt(ref["sxx"])
        qv = (e**2-1) / np.sqrt(2)
        zi = np.zeros(b); zs = np.zeros(b); zv = np.zeros(b)
        for k in range(n):
            zi = LAMBDA*qi[:,k] + (1-LAMBDA)*zi
            zs = LAMBDA*qs[:,k] + (1-LAMBDA)*zs
            zv = LAMBDA*qv[:,k] + (1-LAMBDA)*zv
            if k >= PHASE_I_N:
                j = k-PHASE_I_N
                if signed:
                    paths["Intercept"][first:first+b,j] = zi/sd_i[k]
                    paths["Slope"][first:first+b,j] = zs/sd_s[k]
                    paths["Residual SD"][first:first+b,j] = zv/sd_i[k]
                else:
                    paths["Intercept"][first:first+b,j] = np.abs(zi)/sd_i[k]
                    paths["Slope"][first:first+b,j] = np.abs(zs)/sd_s[k]
                    paths["Residual SD"][first:first+b,j] = np.abs(zv)/sd_i[k]
    if signed:
        # Two-sided path-wise component limits: each bootstrap path contributes
        # its maximum absolute statistic across the entire monitoring horizon.
        return {name: float(np.quantile(np.max(np.abs(path), axis=1),
                                        1.0 - 1.0 / ARL0))
                for name, path in paths.items()}
    return {name: float(np.max(np.quantile(vals, 1.0 - 1.0 / ARL0, axis=0))) for name, vals in paths.items()}


def draw(path, days, values, ucl, name, scenario="in-control", lcl=0.0):
    """Render a clean publication figure without changing the statistic or UCL."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#d9dee5", linestyle="--", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.plot(days, values, color="#238b57", marker="o", markersize=3.2,
            linewidth=1.35, markeredgewidth=0.0, label="Component statistic")
    # These plotted EWMA3 components are absolute-valued; their natural LCL is 0.
    lcl = float(lcl)
    ax.axhline(ucl, color="#c62828", linestyle="--", linewidth=1.8, label=f"UCL = {ucl:.3f}")
    ax.axhline(lcl, color="#c62828", linestyle="--", linewidth=1.5, label=f"LCL = {lcl:.3f}")
    ax.fill_between(days, lcl, ucl, color="#83bd91", alpha=0.13, label="Control region")
    alarm = np.flatnonzero((values > ucl) | (values < lcl))
    if len(alarm):
        ax.scatter(days[alarm], values[alarm], color="#c62828", s=35,
                   edgecolor="white", linewidth=0.45, label="Signal", zorder=5)
    first_signal_day = None
    if scenario != "in-control":
        ax.axvline(90.5, color="#333", linestyle="--", linewidth=1.4,
                   label="Intercept shift starts (day 91)")
        post_hits = np.flatnonzero((days >= 91) & ((values > ucl) | (values < lcl)))
        if len(post_hits):
            first_signal_day = int(days[post_hits[0]])
            first_value = float(values[post_hits[0]])
            ax.scatter([first_signal_day], [first_value], s=125,
                       color="#ff8c00", edgecolor="black", linewidth=1.2,
                       zorder=8, label=f"First signal — day {first_signal_day}")
            ax.annotate(f"First signal\nDay {first_signal_day}",
                        xy=(first_signal_day, first_value),
                        xytext=(12, 18), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="#111",
                        arrowprops={"arrowstyle": "->", "color": "#333", "lw": 1.0},
                        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#444", "alpha": 0.95},
                        zorder=9)
    display_name = "Residual SD" if name == "Residual SD" else name
    state_label = "in-control case study" if scenario == "in-control" else "intercept-shift case study"
    ax.set_title(f"EWMA3 {display_name} component — {state_label}",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Day", fontsize=11)
    y_span = max(float(np.max(values) - np.min(values)), float(ucl - lcl), 1e-6)
    y_pad = 0.08 * y_span
    ax.set_ylim(min(lcl, float(np.min(values))) - y_pad,
                max(ucl, float(np.max(values))) + y_pad)
    ax.tick_params(axis="both", labelsize=8.5, direction="out", length=3)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.95,
              edgecolor="#c7cdd4")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#727983")
        ax.spines[side].set_linewidth(0.8)
    # Save vector PDF plus high-resolution PNG for manuscript submission.
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    t = df["T_avg"].to_numpy(float)
    r = df["R_avg"].to_numpy(float)
    if len(t) != 150:
        raise ValueError(f"Expected 150 daily observations, got {len(t)}")

    days = np.arange(PHASE_I_N + 1, len(t) + 1)
    ref = fit_reference(t, r)
    # Apply the requested response-level step shift from day 91.
    shifted_r = r.copy()
    shifted_r[SHIFT_DAY - 1:] += SHIFT_DELTA
    # Reuse the exact same absolute component statistics and bootstrap UCLs
    # as the in-control charts so the two scenarios are directly comparable.
    limits = bootstrap_limits(t, ref, signed=False)
    values = component_stats(t, r, ref, signed=False)
    shifted_values = component_stats(t, shifted_r, ref, signed=False)
    rows = []
    for name, series in shifted_values.items():
        ucl = limits[name]
        stem = "residual_sd" if name == "Residual SD" else name.lower()
        shifted_lcl = 0.0  # all three plotted component statistics are absolute-valued
        draw(OUT / f"ewma3_{stem}_intercept_shift.png", days, series,
             ucl, name, scenario="intercept-shift", lcl=shifted_lcl)
        post = days >= 91
        post_hits = np.flatnonzero(series[post] > ucl)
        ic_hits = np.flatnonzero(values[name] > ucl)
        rows.append({"component": name, "lcl_display": shifted_lcl, "ucl": ucl,
                     "incontrol_alarms": int(len(ic_hits)),
                     "shift_alarms": int(len(post_hits)),
                     "first_shift_signal_day": int(days[post][post_hits[0]]) if len(post_hits) else None})

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "ewma3_component_intercept_shift_summary.csv", index=False)
    metadata = {"phase1_days": [1, PHASE_I_N], "phase2_days": [31, 150],
                "shift_start_day": SHIFT_DAY, "shift_delta": SHIFT_DELTA,
                "intercept_shift_sigma": ref["sigma"],
                "lambda": LAMBDA, "ARL0_target": ARL0,
                "bootstrap_paths": N_BOOT, "limits": limits,
                "incontrol_alarms": {row["component"]: row["incontrol_alarms"] for row in rows},
                "shift_alarms": {row["component"]: row["shift_alarms"] for row in rows}}
    (OUT / "ewma3_component_intercept_shift_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(result.to_string(index=False))
    print("Three intercept-shift charts written to:", OUT)

if __name__ == "__main__": main()
