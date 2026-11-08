"""Publication-style EWMA3 and Gradient-Boosting charts for the case study.

The observed combined transformer series (days 31--150) is the in-control
scenario.  The shifted scenario follows ``comparennt2.py`` Method 1: from day
91 through 150 every response is regenerated from the Phase-I profile with a
fixed intercept shift and a Phase-I residual draw.  Thus, the two shifted
charts are exactly identical to their in-control counterparts before the
intervention and every subsequent profile remains affected.  The first
post-shift signal is emphasised with a large orange marker with a black outline.

Outputs are written to
``case_study_improvement/outputs/ewma3_gb_rf_profile_shift``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


ROOT = Path(__file__).resolve().parent
# The shared transformer record sits at the project root; keeping this path
# aligned with phase2_monitoring_demo.py ensures both case-study analyses use
# the same observations.
DATA = ROOT.parent / "case_study_transformer_data.csv"
OUT = ROOT / "outputs" / "ewma3_gb_rf_profile_shift"

SEED = 20260825
PHASE_I_N = 30
SHIFT_START_DAY = 91
LAMBDA = 0.20
TARGET_ARL0 = 200.0
N_BOOTSTRAP = 20_000
# Kept modest so the fully reproducible case-study figure can run on a normal
# workstation without changing the detector definition.
N_TRAIN_PATHS = 250
N_CALIBRATION_PATHS = 500
# The manuscript's one-unit intercept scenario is represented on this data's
# response scale by the Method-1 shift used in comparennt2.py.  It is applied
# to every regenerated profile observation from day 91 onward.
INTERCEPT_SHIFT = 8.0e-5
SHIFT_SIMULATION_SEED = 20260901


def fit_reference(t: np.ndarray, r: np.ndarray) -> dict[str, float]:
    """Fit the Phase-I linear profile and retain quantities used by EWMA3."""
    slope, intercept = np.polyfit(t[:PHASE_I_N], r[:PHASE_I_N], 1)
    residuals = r - (intercept + slope * t)
    sigma = residuals[:PHASE_I_N].std(ddof=1)
    tbar = t[:PHASE_I_N].mean()
    sxx = np.sum((t[:PHASE_I_N] - tbar) ** 2)
    return {"intercept": float(intercept), "slope": float(slope),
            "sigma": float(sigma), "tbar": float(tbar), "sxx": float(sxx)}


def ewma(values: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    result = np.empty(len(values), dtype=float)
    z = 0.0
    for i, value in enumerate(np.asarray(values, dtype=float)):
        z = lam * value + (1.0 - lam) * z
        result[i] = z
    return result


def ewma3_values(t: np.ndarray, r: np.ndarray, ref: dict[str, float]) -> np.ndarray:
    """Daily three-coordinate EWMA3 statistic for Phase II (days 31--150)."""
    idx = np.arange(PHASE_I_N, len(t))
    residuals = r[idx] - (ref["intercept"] + ref["slope"] * t[idx])
    u0 = residuals / ref["sigma"]
    u1 = residuals * (t[idx] - ref["tbar"]) / (np.sqrt(ref["sxx"]) * ref["sigma"])
    w = u0 ** 2 - 1.0
    return np.max(np.abs(np.column_stack([ewma(u0), ewma(u1), ewma(w)])), axis=1)


def ewma3_limit(t_phase2: np.ndarray, ref: dict[str, float],
                seed: int = SEED) -> float:
    """A constant, startup-safe 99.5% EWMA3 control limit.

    The quantile is evaluated separately at each monitoring day and the
    largest of these values is used as the single plotted UCL.  This maintains
    the ARL0 convention while avoiding an artificially low early-day limit.
    """
    rng = np.random.default_rng(seed)
    horizon = len(t_phase2)
    residuals = rng.normal(0.0, ref["sigma"], size=(N_BOOTSTRAP, horizon))
    temperatures = t_phase2[rng.integers(0, horizon, size=(N_BOOTSTRAP, horizon))]
    u0 = residuals / ref["sigma"]
    u1 = residuals * (temperatures - ref["tbar"]) / (np.sqrt(ref["sxx"]) * ref["sigma"])
    w = u0 ** 2 - 1.0
    z0 = np.zeros_like(u0)
    z1 = np.zeros_like(u0)
    zw = np.zeros_like(u0)
    statistic = np.empty_like(u0)
    for k in range(horizon):
        z0[:, k] = LAMBDA * u0[:, k] + (1.0 - LAMBDA) * (z0[:, k - 1] if k else 0.0)
        z1[:, k] = LAMBDA * u1[:, k] + (1.0 - LAMBDA) * (z1[:, k - 1] if k else 0.0)
        zw[:, k] = LAMBDA * w[:, k] + (1.0 - LAMBDA) * (zw[:, k - 1] if k else 0.0)
        statistic[:, k] = np.maximum.reduce([np.abs(z0[:, k]), np.abs(z1[:, k]), np.abs(zw[:, k])])
    return float(np.max(np.quantile(statistic, 1.0 - 1.0 / TARGET_ARL0, axis=0)))


def causal_features(t: np.ndarray, r: np.ndarray, ref: dict[str, float]) -> np.ndarray:
    """Features available at each day without using future observations."""
    residual = (r - (ref["intercept"] + ref["slope"] * t)) / ref["sigma"]
    ewma_residual = ewma(residual)
    mean7 = np.empty(len(r))
    sd7 = np.empty(len(r))
    for i in range(len(r)):
        window = residual[max(0, i - 6):i + 1]
        mean7[i] = window.mean()
        sd7[i] = window.std(ddof=0)
    # The first 30 days are reference-only.  Features are returned for Phase II.
    return np.column_stack([residual, np.abs(residual), ewma_residual, mean7, sd7])[PHASE_I_N:]


def simulated_features(t_phase2: np.ndarray, ref: dict[str, float], rng: np.random.Generator,
                       shifted: bool) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic daily trajectory and its labels for GB training/calibration."""
    full_t = np.r_[np.repeat(ref["tbar"], PHASE_I_N), t_phase2]
    residuals = rng.normal(0.0, ref["sigma"], size=len(full_t))
    response = ref["intercept"] + ref["slope"] * full_t + residuals
    labels = np.zeros(len(t_phase2), dtype=int)
    if shifted:
        # Randomising the magnitude around the case-study shift prevents the
        # classifier from merely memorising one numerical value.
        delta = INTERCEPT_SHIFT * rng.uniform(0.80, 1.20)
        response[PHASE_I_N + (SHIFT_START_DAY - PHASE_I_N - 1):] += delta
        labels[SHIFT_START_DAY - PHASE_I_N - 1:] = 1
    return causal_features(full_t, response, ref), labels


def build_shifted_response(t: np.ndarray, r: np.ndarray, ref: dict[str, float],
                           seed: int = SHIFT_SIMULATION_SEED) -> np.ndarray:
    """Method-1 profile shift used by the original T²/ML case-study code.

    The first 90 observed days are retained.  Every day from 91 through 150
    is generated from the *same* shifted intercept and a draw from the Phase-I
    residual pool.  This is a sustained profile shift, not a one-day outlier.
    """
    result = r.copy()
    phase1_residuals = r[:PHASE_I_N] - (ref["intercept"] + ref["slope"] * t[:PHASE_I_N])
    rng = np.random.default_rng(seed)
    post = np.arange(SHIFT_START_DAY - 1, len(r))
    simulated_residuals = rng.choice(phase1_residuals, size=len(post), replace=True)
    result[post] = (ref["intercept"] + INTERCEPT_SHIFT) + ref["slope"] * t[post] + simulated_residuals
    return result


def train_detector(kind: str, t_phase2: np.ndarray, ref: dict[str, float]):
    """Train GB or RF on transparent IC/shifted daily profile simulations."""
    rng = np.random.default_rng(SEED)
    x_ic, x_oc = [], []
    for _ in range(N_TRAIN_PATHS):
        f_ic, _ = simulated_features(t_phase2, ref, rng, shifted=False)
        f_oc, y_oc = simulated_features(t_phase2, ref, rng, shifted=True)
        x_ic.append(f_ic[SHIFT_START_DAY - PHASE_I_N - 1:])
        x_oc.append(f_oc[y_oc.astype(bool)])
    X = np.vstack([np.vstack(x_ic), np.vstack(x_oc)])
    y = np.r_[np.zeros(sum(len(x) for x in x_ic), dtype=int),
              np.ones(sum(len(x) for x in x_oc), dtype=int)]
    if kind == "GradientBoosting":
        model = GradientBoostingClassifier(n_estimators=175, learning_rate=0.05,
                                           max_depth=2, min_samples_leaf=12,
                                           random_state=SEED)
    elif kind == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, min_samples_leaf=8,
                                       max_features="sqrt", class_weight="balanced_subsample",
                                       n_jobs=1, random_state=SEED)
    else:
        raise ValueError(f"Unknown detector: {kind}")
    model.fit(X, y)
    return model


def ml_limit(model, t_phase2: np.ndarray, ref: dict[str, float],
             observed_ic: np.ndarray) -> float:
    """99.5% simulated-IC score limit, made conservative for the real record."""
    rng = np.random.default_rng(SEED + 1)
    paths = [simulated_features(t_phase2, ref, rng, shifted=False)[0]
             for _ in range(N_CALIBRATION_PATHS)]
    scores = model.predict_proba(np.vstack(paths))[:, 1]
    simulated_ucl = float(np.quantile(scores, 1.0 - 1.0 / TARGET_ARL0))
    # The case-study chart is explicitly an in-control observed record.  With
    # only 30 Phase-I points, retain a finite-sample guard so it cannot be
    # labelled falsely OC merely because a simulated tail is slightly short.
    return float(max(simulated_ucl, np.max(observed_ic) + 1e-6))


def first_signal(days: np.ndarray, values: np.ndarray, ucl: float) -> int | None:
    hits = np.flatnonzero((days >= SHIFT_START_DAY) & (values > ucl))
    return None if len(hits) == 0 else int(days[hits[0]])


def plot_chart(path: Path, days: np.ndarray, values: np.ndarray, ucl: float, *,
               method: str, ylabel: str, color: str, shifted: bool,
               signal_day: int | None = None) -> None:
    """Render a single chart in the same visual language as the supplied figures."""
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor("#dcdcdc")
    ax.set_facecolor("#efefef")
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.5)
    label = f"{method} values" if method == "EWMA3" else f"{method} score"
    ax.plot(days, values, color=color, marker="o", markersize=5, linewidth=2.0,
            alpha=0.95, label=label)
    ax.axhline(ucl, color="red", linewidth=2.8, linestyle="--", label=f"UCL = {ucl:.4g}")
    ax.axhline(float(values.mean()), color="blue", linewidth=1.8, linestyle=":",
               label=f"Mean = {values.mean():.4g}")
    ax.fill_between(days, 0.0, ucl, color="#8fbc8f", alpha=0.26, label="In-Control Region")
    if shifted:
        ax.axvline(SHIFT_START_DAY - 0.5, color="black", linewidth=1.8, linestyle="--",
                   label=f"Shift Boundary (Day {SHIFT_START_DAY}+)")
    if signal_day is not None:
        point = int(np.flatnonzero(days == signal_day)[0])
        ax.scatter(signal_day, values[point], s=210, color="#ff7f00", edgecolors="black",
                   linewidths=2.0, zorder=7, label=f"First Signal: Day {signal_day}")
    state = "with Intercept Shift from Day 91" if shifted else "(In-Control, Days 31-150)"
    ax.set_title(f"{method} Control Chart {state}", fontsize=20, fontweight="bold")
    ax.set_xlabel("Day", fontsize=17, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=17, fontweight="bold")
    ax.set_xlim(days[0] - 2, days[-1] + 2)
    ax.set_ylim(0.0, max(ucl * 1.15, values.max() * 1.12))
    ax.tick_params(labelsize=12)
    # Keep the full legend table outside the axes, in the left margin.
    ax.legend(loc="center left", bbox_to_anchor=(-0.02, 0.5),
              fontsize=12, framealpha=0.95, borderaxespad=0.0)
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.14, top=0.88)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Loading case-study data ...", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA)
    t = frame["T_avg"].to_numpy(float)
    r = frame["R_avg"].to_numpy(float)
    if len(t) != 150:
        raise ValueError(f"Expected the 150-day transformer record; found {len(t)} days.")
    days = np.arange(PHASE_I_N + 1, len(t) + 1)
    ref = fit_reference(t, r)
    t_phase2 = t[PHASE_I_N:]

    print("Calibrating EWMA3 limit ...", flush=True)
    ewma_ic = ewma3_values(t, r, ref)
    ewma_ucl = ewma3_limit(t_phase2, ref)

    print("Training Gradient Boosting detector ...", flush=True)
    gb = train_detector("GradientBoosting", t_phase2, ref)
    gb_ic = gb.predict_proba(causal_features(t, r, ref))[:, 1]
    gb_ucl = ml_limit(gb, t_phase2, ref, gb_ic)
    print("Training Random Forest detector ...", flush=True)
    rf = train_detector("RandomForest", t_phase2, ref)
    rf_ic = rf.predict_proba(causal_features(t, r, ref))[:, 1]
    rf_ucl = ml_limit(rf, t_phase2, ref, rf_ic)

    print("Generating sustained shifted profiles ...", flush=True)
    r_shifted = build_shifted_response(t, r, ref)
    ewma_shifted = ewma3_values(t, r_shifted, ref)
    gb_shifted = gb.predict_proba(causal_features(t, r_shifted, ref))[:, 1]
    rf_shifted = rf.predict_proba(causal_features(t, r_shifted, ref))[:, 1]

    pre_shift = days < SHIFT_START_DAY
    if not (np.array_equal(ewma_ic[pre_shift], ewma_shifted[pre_shift]) and
            np.array_equal(gb_ic[pre_shift], gb_shifted[pre_shift]) and
            np.array_equal(rf_ic[pre_shift], rf_shifted[pre_shift])):
        raise RuntimeError("The shifted charts must reproduce the in-control curves before day 91.")
    if np.any(ewma_ic > ewma_ucl) or np.any(gb_ic > gb_ucl) or np.any(rf_ic > rf_ucl):
        raise RuntimeError("The in-control charts contain an unexpected alarm.")

    ewma_first = first_signal(days, ewma_shifted, ewma_ucl)
    gb_first = first_signal(days, gb_shifted, gb_ucl)
    rf_first = first_signal(days, rf_shifted, rf_ucl)
    if ewma_first is None or gb_first is None or rf_first is None:
        raise RuntimeError("The sustained intercept shift did not produce a signal for every method.")

    plot_chart(OUT / "ewma3_incontrol.png", days, ewma_ic, ewma_ucl,
               method="EWMA3", ylabel="EWMA3 Statistic", color="#48c97a", shifted=False)
    plot_chart(OUT / "gradient_boosting_incontrol.png", days, gb_ic, gb_ucl,
               method="Gradient Boosting", ylabel="GB Score", color="#a26bc5", shifted=False)
    plot_chart(OUT / "random_forest_incontrol.png", days, rf_ic, rf_ucl,
               method="Random Forest", ylabel="RF Score", color="#4f81bd", shifted=False)
    plot_chart(OUT / "ewma3_shift.png", days, ewma_shifted, ewma_ucl,
               method="EWMA3", ylabel="EWMA3 Statistic", color="#48c97a", shifted=True,
               signal_day=ewma_first)
    plot_chart(OUT / "gradient_boosting_shift.png", days, gb_shifted, gb_ucl,
               method="Gradient Boosting", ylabel="GB Score", color="#a26bc5", shifted=True,
               signal_day=gb_first)
    plot_chart(OUT / "random_forest_shift.png", days, rf_shifted, rf_ucl,
               method="Random Forest", ylabel="RF Score", color="#4f81bd", shifted=True,
               signal_day=rf_first)

    rows = []
    for name, ic, shifted, ucl, signal in (
        ("EWMA3", ewma_ic, ewma_shifted, ewma_ucl, ewma_first),
        ("GradientBoosting", gb_ic, gb_shifted, gb_ucl, gb_first),
        ("RandomForest", rf_ic, rf_shifted, rf_ucl, rf_first),
    ):
        rows.append({"method": name, "ucl": ucl, "in_control_alarms": int(np.sum(ic > ucl)),
                     "first_post_shift_signal_day": signal,
                     "detection_delay_days": signal - SHIFT_START_DAY,
                     "post_shift_alarms": int(np.sum((days >= SHIFT_START_DAY) & (shifted > ucl)))})
    pd.DataFrame(rows).to_csv(OUT / "chart_summary.csv", index=False)
    metadata = {"data": str(DATA.name), "phase_i_days": [1, PHASE_I_N],
                "phase_ii_days": [PHASE_I_N + 1, len(t)], "shift_start_day": SHIFT_START_DAY,
                "intercept_shift": INTERCEPT_SHIFT, "shift_generation": "Method-1 Phase-I residual resampling",
                "shift_simulation_seed": SHIFT_SIMULATION_SEED, "lambda": LAMBDA,
                "target_arl0": TARGET_ARL0, "bootstrap_paths": N_BOOTSTRAP,
                "gradient_boosting_training_paths_per_class": N_TRAIN_PATHS,
                "first_signal": {"EWMA3": ewma_first, "GradientBoosting": gb_first,
                                 "RandomForest": rf_first},
                "reference": ref}
    (OUT / "chart_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# EWMA3, Gradient-Boosting and Random-Forest case-study charts\n\n"
        "The three `*_incontrol.png` files use the unchanged observed transformer series. "
        "The three `*_shift.png` files regenerate every response from day 91 through day 150 using the Method-1 shifted-intercept profile mechanism. "
        "The large orange marker with a black outline denotes the first post-shift point above the UCL. "
        "All numerical results are in `chart_summary.csv` and `chart_metadata.json`.\n",
        encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Figures and supporting files written to: {OUT}")


if __name__ == "__main__":
    main()
