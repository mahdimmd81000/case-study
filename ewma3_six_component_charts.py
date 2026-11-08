# Standalone six-chart EWMA3 analysis of the aggregated transformer CSV.
# The only analysis input is ../case_study_transformer_data.csv. The response
# shift is +0.0001 from day 91.
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
OUT = ROOT / "outputs" / "ewma3_six_charts_method2_shift"
PHASE_I_N = 30
SHIFT_DAY = 91
SHIFT_DELTA = 0.0001
LAMBDA = 0.20
ARL0 = 200.0
N_BOOT = 20_000
SEED = 20260825
COMPONENTS = ("Intercept", "Slope", "Residual SD")


def ewma(values: np.ndarray) -> np.ndarray:
    # Recursive EWMA initialized at zero on day 1.
    values = np.asarray(values, dtype=float)
    result = np.empty(values.size, dtype=float)
    z = 0.0
    for k, value in enumerate(values):
        z = LAMBDA * value + (1.0 - LAMBDA) * z
        result[k] = z
    return result

def intercept_ewma_sd(n: int) -> np.ndarray:
    # Exact SD of EWMA of independent unit-variance observations, by day.
    k = np.arange(1, n + 1, dtype=float)
    return np.sqrt(LAMBDA / (2.0 - LAMBDA) *
                   (1.0 - (1.0 - LAMBDA) ** (2.0 * k)))


def slope_ewma_sd(t: np.ndarray, tbar: float, sxx: float) -> np.ndarray:
    # Daywise SD for the slope-direction EWMA, assuming independent unit
    # residuals and the observed temperature schedule.
    innovation_variance = (t - tbar) ** 2 / sxx
    result = np.empty(len(t), dtype=float)
    variance = 0.0
    for k, vk in enumerate(innovation_variance):
        variance = LAMBDA**2 * vk + (1.0 - LAMBDA)**2 * variance
        result[k] = np.sqrt(variance)
    return result

def calculate_components(t: np.ndarray, response: np.ndarray,
                         reference: dict[str, float]) -> dict[str, np.ndarray]:
    # Return Phase-II standardized component statistics after the Phase-I warm-up.
    e = (response - (reference["a0"] + reference["a1"] * t)) / reference["sigma"]
    # Carry recursive EWMA states from Phase I through the monitoring period.
    z_intercept = ewma(e)
    z_slope = ewma(e * (t - reference["tbar"]) / np.sqrt(reference["sxx"]))
    z_scale = ewma((e**2 - 1.0) / np.sqrt(2.0))
    sd_i = intercept_ewma_sd(len(t))
    sd_s = slope_ewma_sd(t, reference["tbar"], reference["sxx"])
    monitor = slice(PHASE_I_N, len(t))
    return {
        "Intercept": z_intercept[monitor] / sd_i[monitor],  # signed intercept
        "Slope": np.abs(z_slope[monitor]) / sd_s[monitor],
        "Residual SD": np.abs(z_scale[monitor]) / sd_i[monitor],
    }


def calibrate_limits(t: np.ndarray, reference: dict[str, float]) -> dict[str, float]:
    # Parametric in-control bootstrap conditional on observed Phase-I EWMA states.
    # Each component UCL is calibrated from its maximum statistic over Phase II.
    rng = np.random.default_rng(SEED)
    n = len(t)
    horizon = n - PHASE_I_N
    t2 = t[PHASE_I_N:]
    sd_i_all = intercept_ewma_sd(n)
    sd_i = sd_i_all[PHASE_I_N:]
    sd_s_all = slope_ewma_sd(t, reference["tbar"], reference["sxx"])
    sd_s = sd_s_all[PHASE_I_N:]
    values = {name: np.empty((N_BOOT, horizon), dtype=np.float32)
              for name in COMPONENTS}
    path_maxima = {name: np.empty(N_BOOT, dtype=np.float32) for name in COMPONENTS}
    chunk_size = 500
    for first in range(0, N_BOOT, chunk_size):
        count = min(chunk_size, N_BOOT - first)
        e = rng.normal(0.0, 1.0, size=(count, horizon))
        qs = e * (t2 - reference["tbar"]) / np.sqrt(reference["sxx"])
        qv = (e**2 - 1.0) / np.sqrt(2.0)
        # Use the actual, fixed Phase-I residual history to initialize each path.
        phase1_e = (reference["phase1_residuals"] / reference["sigma"])
        z0 = np.full(count, ewma(phase1_e)[-1])
        phase1_slope = phase1_e * (t[:PHASE_I_N] - reference["tbar"]) / np.sqrt(reference["sxx"])
        phase1_scale = (phase1_e**2 - 1.0) / np.sqrt(2.0)
        z1 = np.full(count, ewma(phase1_slope)[-1])
        zv = np.full(count, ewma(phase1_scale)[-1])
        for k in range(horizon):
            z0 = LAMBDA * e[:, k] + (1.0 - LAMBDA) * z0
            z1 = LAMBDA * qs[:, k] + (1.0 - LAMBDA) * z1
            zv = LAMBDA * qv[:, k] + (1.0 - LAMBDA) * zv
            values["Intercept"][first:first + count, k] = np.abs(z0) / sd_i[k]
            values["Slope"][first:first + count, k] = np.abs(z1) / sd_s[k]
            values["Residual SD"][first:first + count, k] = np.abs(zv) / sd_i[k]
        for name in COMPONENTS:
            path_maxima[name][first:first + count] = np.max(values[name][first:first + count], axis=1)
    np.savez_compressed(OUT / "limit_audit_path_maxima.npz", **path_maxima)
    q = 1.0 - 1.0 / ARL0
    # Intercept/slope have signed two-sided limits from absolute maxima;
    # residual-SD is absolute-valued and has a zero lower limit.
    quantile_limits = {
        "Intercept": float(np.quantile(path_maxima["Intercept"], q)),
        "Slope": float(np.quantile(path_maxima["Slope"], q)),
        "Residual SD": float(np.quantile(path_maxima["Residual SD"], q))
    }
    return quantile_limits


def is_out_of_control(series: np.ndarray, ucl: float,
                      component: str) -> np.ndarray:
    # Intercept/slope are signed and two-sided; residual SD is absolute-valued.
    if component == "Residual SD":
        return series > ucl
    return np.abs(series) > ucl

def plot_chart(path: Path, days: np.ndarray, series: np.ndarray, ucl: float,
               component: str, shifted: bool) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_facecolor("#f4f4f4")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.plot(days, series, color="#8e63bd", marker="o", markersize=3,
            linewidth=1.3, label=f"EWMA3 {component}")
    ax.axhline(ucl, color="red", linestyle="--", linewidth=1.8,
               label=f"UCL = {ucl:.4f}")
    lcl = 0.0 if component == "Residual SD" else -ucl
    ax.axhline(lcl, color="red", linestyle="--", linewidth=1.8,
               label=f"LCL = {lcl:.4f}")
    ax.axhline(0.0, color="blue", linestyle=":", linewidth=1.1, label="Center = 0")
    ax.fill_between(days, lcl, ucl, color="#8fbc8f", alpha=0.25,
                    label="In-control region")
    ybottom = min(1.15 * lcl, 1.1 * float(series.min()))
    ax.set_ylim(ybottom, max(1.15 * ucl, 1.1 * float(series.max())))
    alarms = np.flatnonzero(is_out_of_control(series, ucl, component))
    if shifted:
        ax.axvline(SHIFT_DAY - 0.5, color="black", linestyle="--", linewidth=1.2,
                   label=f"Shift starts day {SHIFT_DAY}")
    if alarms.size:
        ax.scatter(days[alarms], series[alarms], color="#d62728", s=24,
                   zorder=5, label=f"Signals: {alarms.size}")
    state = "intercept shift" if shifted else "in control"
    ax.set_title(f"EWMA3 {component} component — {state}", fontweight="bold")
    ax.set_xlabel("Day")
    ax.set_ylabel("Standardized EWMA statistic" if component == "Intercept"
                  else "Absolute standardized EWMA statistic")
    ax.set_xlim(30, 151)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95,
              bbox_to_anchor=(0.01, 0.02), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA)
    t = data["T_avg"].to_numpy(dtype=float)
    response = data["R_avg"].to_numpy(dtype=float)
    if len(t) != 150 or len(response) != 150:
        raise ValueError(f"Expected 150 aggregated daily observations; got {len(t)}")
    if not np.isfinite(t).all() or not np.isfinite(response).all():
        raise ValueError("Aggregated temperature/response data contain non-finite values")

    # EWMA state is warmed using Phase-I and carried into Phase II; the Phase-I
    # reference fit is based only on the first 30 aggregated observations.
    a1, a0 = np.polyfit(t[:PHASE_I_N], response[:PHASE_I_N], 1)
    residual = response - (a0 + a1 * t)
    tbar = float(t[:PHASE_I_N].mean())
    reference = {
        "a0": float(a0), "a1": float(a1),
        "sigma": float(residual[:PHASE_I_N].std(ddof=1)),
        "tbar": tbar,
        "sxx": float(np.sum((t[:PHASE_I_N] - tbar) ** 2)),
        "phase1_residuals": residual[:PHASE_I_N].copy(),
    }
    if reference["sigma"] <= 0.0 or reference["sxx"] <= 0.0:
        raise ValueError("Phase-I residual variation or temperature variation is zero")

    days = np.arange(PHASE_I_N + 1, len(t) + 1)
    ucl = calibrate_limits(t, reference)
    original = calculate_components(t, response, reference)

    # One sustained response/intercept step exactly as specified in the ML case.
    shifted_response = response.copy()
    shifted_response[SHIFT_DAY - 1:] += SHIFT_DELTA
    shifted = calculate_components(t, shifted_response, reference)

    pre_shift = days < SHIFT_DAY
    for component in COMPONENTS:
        if not np.array_equal(original[component][pre_shift], shifted[component][pre_shift]):
            raise RuntimeError(f"{component} chart differs before the shift day")
    if not np.array_equal(shifted_response[:SHIFT_DAY - 1], response[:SHIFT_DAY - 1]):
        raise RuntimeError("The shifted response changed before day 91")
    if not np.allclose(shifted_response[SHIFT_DAY - 1:] - response[SHIFT_DAY - 1:],
                       SHIFT_DELTA, rtol=0.0, atol=1e-15):
        raise RuntimeError("The response shift is not exactly the configured magnitude")

    rows = []
    for component in COMPONENTS:
        ic = original[component]
        sh = shifted[component]
        limit = ucl[component]
        ic_alarms = is_out_of_control(ic, limit, component)
        post = days >= SHIFT_DAY
        shifted_alarms = is_out_of_control(sh, limit, component)
        hit_days = days[post & shifted_alarms]
        stem = component.lower().replace(" ", "_")
        plot_chart(OUT / f"ewma3_{stem}_incontrol.png", days, ic, limit, component, False)
        plot_chart(OUT / f"ewma3_{stem}_intercept_shift.png", days, sh, limit, component, True)
        display_lcl = -limit if component == "Intercept" else 0.0
        rows.append({
            "component": component,
            "ucl": limit,
            "lcl": display_lcl,
            "incontrol_min": float(np.min(ic)),
            "incontrol_max": float(np.max(ic)),
            "incontrol_alarms": int(ic_alarms.sum()),
            "shifted_min": float(np.min(sh)),
            "shifted_max": float(np.max(sh)),
            "shifted_alarms_post_day_91": int(len(hit_days)),
            "first_shift_signal_day": int(hit_days[0]) if hit_days.size else None,
            "shift_delta": SHIFT_DELTA,
            "shift_start_day": SHIFT_DAY,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "ewma3_chart_summary.csv", index=False)
    metadata = {
        "input_csv": str(DATA.resolve()),
        "aggregated_columns": ["T_avg", "R_avg"],
        "observations": len(t),
        "phase_i_days": [1, PHASE_I_N],
        "phase_ii_days": [PHASE_I_N + 1, len(t)],
        "chart_center": 0.0,
        "lambda": LAMBDA,
        "arl0_target": ARL0,
        "bootstrap_paths_for_limits_only": N_BOOT,
        "bootstrap_seed": SEED,
        "calibration_definition": "Per-component 99.5th quantile of maximum absolute statistic across Phase-II days 31-150; EWMA state warmed on Phase-I days 1-30",
        "limit_audit_path_maxima_file": "limit_audit_path_maxima.npz",
        "lcl_by_component": {name: (0.0 if name == "Residual SD" else -ucl[name])
                              for name in COMPONENTS},
        "control_limit_kind": "two-sided signed intercept and slope; absolute residual SD with LCL zero",
        "shift_start_day": SHIFT_DAY,
        "response_intercept_shift": SHIFT_DELTA,
        "shift_reference": "User-requested response/intercept level step of 0.0001 from day 91",
        "reference_fit": reference,
        "ucl_by_component": ucl,
        "display_lcl_by_component": {name: (0.0 if name == "Residual SD" else -ucl[name])
                                      for name in COMPONENTS},
        "display_ucl_by_component": ucl,
        "limit_estimator": "Empirical 99.5th quantile of maximum absolute component statistics over 120 Phase-II days from 20,000 bootstrap paths",
        "shift_effect_in_sigma_units": SHIFT_DELTA / reference["sigma"],
        "post_shift_intercept_statistic_max": float(np.max(np.abs(shifted["Intercept"][days >= SHIFT_DAY]))),
        "summary": rows,
        "all_original_points_within_limits": bool(all(row["incontrol_alarms"] == 0 for row in rows)),
        "all_pre_shift_values_within_limits": bool(all(
            not np.any(is_out_of_control(original[name][pre_shift], ucl[name], name))
            for name in COMPONENTS)),
        "all_pre_shift_values_unchanged": True,
        "all_shifted_response_deltas_exact": True,
        "shifted_alarm_days": {row["component"]: days[
            (days >= SHIFT_DAY) & is_out_of_control(shifted[row["component"]],
                                                    ucl[row["component"]],
                                                    row["component"])].tolist()
                                for row in rows},
    }
    metadata["reference_fit"].pop("phase1_residuals", None)
    (OUT / "ewma3_chart_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Shift: +{SHIFT_DELTA:.8f} from day {SHIFT_DAY}")
    print(f"Phase-I residual SD: {reference['sigma']:.12g}")
    print(f"Results: {OUT.resolve()}")

if __name__ == "__main__":
    main()
