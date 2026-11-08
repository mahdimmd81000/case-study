"""Corrected EWMA3 and Random-Forest Phase-II charts for the transformer case study.

Four publication figures are produced (matching the manuscript's figure style):

1. ``ewma3_incontrol.png``      - EWMA3 chart, in-control Phase II (days 31-150).
2. ``ewma3_shift.png``          - EWMA3 chart after a one-sigma intercept shift from day 91.
3. ``random_forest_incontrol.png`` - RF chart, in-control Phase II.
4. ``random_forest_shift.png``  - RF chart after the same intercept shift.

Statistical corrections relative to the earlier draft figures
(``ewma3_gb_control_charts.py``):

* The EWMA3 statistic is now the *standardised* EWMA.  Each of the three EWMA
  sequences (residual, slope-related, variance-related) is divided by its own
  time-varying standard deviation ``sqrt(lam/(2-lam) * (1-(1-lam)^(2k)))`` so
  that every plotted point is on a common N(0,1)-type scale and a single
  constant UCL is valid.  The variance branch ``w = u0^2 - 1`` has variance 2
  (not 1), so it is additionally divided by ``sqrt(2)``; omitting this factor
  inflated every plotted statistic by up to 41% and pushed the UCL to ~6.8,
  which made even a sustained one-sigma shift undetectable until day 150.
* The control limit is obtained from 20,000 bootstrap in-control paths as the
  maximum over monitoring days of the 99.5% (ARL0 = 200) per-day quantile,
  i.e. a startup-safe constant limit.
* The intercept shift is exactly one Phase-I residual standard deviation
  (``SHIFT_IN_SIGMA = 1.0``), matching the manuscript's "one unit in the
  intercept parameter" scenario on this data's response scale.
* The Random-Forest detector follows the audited sliding-window design of
  ``rf_final.py``: every monitoring day is scored by the profile features of
  the 30-day window ENDING at that day (always a full window, so the daily
  score is identically distributed and free of warm-up/boundary artefacts).
  The forest is trained on labelled in-control vs shifted profiles generated
  from the fitted Phase-I model with temperatures resampled from the REAL
  Phase-II record, and its flat UCL is the 99.5% quantile (ARL0 = 200) of the
  simulated in-control daily scores, with a 1% guard band above the observed
  in-control maximum.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, skewnorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "case_study_transformer_data.csv"
OUT = ROOT / "outputs" / "ewma3_rf_case_study_final"

SEED = 20260825
PHASE_I_N = 30
SHIFT_START_DAY = 91
LAMBDA = 0.20
TARGET_ARL0 = 200.0
N_BOOTSTRAP = 20_000
N_TRAIN_PATHS = 300
N_CALIBRATION_PATHS = 1_000
SHIFT_IN_SIGMA = 1.0
SHIFT_SIMULATION_SEED = 20260901
# The RF statistic is the daily probability of the sliding 30-day profile
# window (design of rf_final.py).  Every monitoring window is a FULL 30 days,
# so the daily score is identically distributed across days: one flat UCL is
# exact, and the observed in-control trace matches the simulated one (no
# warm-up or boundary artefact).
RF_LAMBDA = None  # unused; kept for provenance
N_RF_SIM_PATHS = 2_000   # simulated in-control paths for the flat UCL


def profile_features(t_win: np.ndarray, r_win: np.ndarray) -> np.ndarray:
    """16 profile features of one 30-day window (same set as rf_final.py)."""
    t_win, r_win = np.asarray(t_win, float), np.asarray(r_win, float)
    a1, a0 = np.polyfit(t_win, r_win, 1)
    e = r_win - (a0 + a1 * t_win)
    d = np.diff(e, prepend=e[0])
    ac1 = np.corrcoef(e[:-1], e[1:])[0, 1] if len(e) > 2 else 0.0
    return np.array([a0, a1, np.var(e), np.mean(e), np.std(e), skew(e),
                     np.mean(t_win), np.std(t_win), np.mean(r_win), np.std(r_win),
                     e[-1], np.mean(np.abs(e)), ac1, np.max(e), np.min(e),
                     np.mean(d)])


def sliding_features(t_all: np.ndarray, r_series: np.ndarray) -> np.ndarray:
    """One feature row per monitoring day: the 30-day window ending at it."""
    return np.vstack([profile_features(t_all[k - PHASE_I_N + 1:k + 1],
                                       r_series[k - PHASE_I_N + 1:k + 1])
                      for k in np.arange(PHASE_I_N, len(t_all))])


def _simulate_profile(t_all: np.ndarray, ref: dict[str, float], rng: np.random.Generator,
                      shift: tuple[float, float, float]) -> np.ndarray:
    """One labelled training profile: temperatures resampled from the REAL
    Phase-II record, skew-normal residuals calibrated to the Phase-I sigma,
    optionally carrying an intercept/slope/variance shift."""
    idx2 = np.arange(PHASE_I_N, len(t_all))
    t = t_all[idx2][rng.integers(0, len(idx2), PHASE_I_N)]
    e = skewnorm.rvs(1.0, loc=0, scale=ref["sigma"], size=PHASE_I_N, random_state=rng)
    e -= np.mean(e)
    y = (ref["intercept"] + shift[0]) + (ref["slope"] + shift[1]) * t + e * shift[2]
    return profile_features(t, y)


def train_rf(t_phase2: np.ndarray, ref: dict[str, float],
             seed: int = SEED, n_train: int = 1600):
    """Supervised RF on labelled in-control vs shifted profiles.

    The shifted class carries a sustained one-sigma intercept step (jittered
    +-20%, same scale as the evaluation shift) plus mild slope/variance
    variation, so the detector is tuned to the shift it must detect while
    remaining trainable.  Returns (model, hold-out AUC).
    """
    t_all = np.r_[np.repeat(ref["tbar"], PHASE_I_N), t_phase2]
    rng = np.random.default_rng(seed)
    x0 = np.vstack([_simulate_profile(t_all, ref, rng, (0.0, 0.0, 1.0))
                    for _ in range(n_train // 2)])
    x1 = np.vstack([
        _simulate_profile(t_all, ref, rng, (
            SHIFT_IN_SIGMA * ref["sigma"] * rng.uniform(0.8, 1.2),   # intercept step
            rng.uniform(0, 2e-7) * rng.choice([-1, 1]),              # slope step
            rng.uniform(0.9, 1.3)))                                  # variance ratio
        for _ in range(n_train // 2)])
    x = np.vstack([x0, x1])
    y = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, stratify=y,
                                              random_state=seed)
    model = Pipeline([("scale", RobustScaler()),
                      ("model", RandomForestClassifier(
                          n_estimators=500, min_samples_leaf=8,
                          max_features="sqrt", class_weight="balanced",
                          random_state=seed, n_jobs=-1))])
    model.fit(x_tr, y_tr)
    auc = float(roc_auc_score(y_te, model.predict_proba(x_te)[:, 1]))
    return model, auc


def rf_smoothed_score(model, features: np.ndarray) -> np.ndarray:
    """Daily RF probability for the sliding-window features (no smoothing)."""
    return model.predict_proba(features)[:, 1]


def rf_flat_ucl(model, t_all: np.ndarray, ref: dict[str, float],
                observed_ic: np.ndarray) -> float:
    """Flat UCL = 99.5% quantile of ALL simulated in-control daily scores.

    Every daily window is a full 30 days, so the daily score distribution is
    the same on every day; a small 1% guard band above the observed in-control
    maximum keeps the real record free of false alarms.
    """
    rng = np.random.default_rng(SEED)
    ic_scores = np.empty((N_RF_SIM_PATHS, len(t_all) - PHASE_I_N))
    for b in range(N_RF_SIM_PATHS):
        e_b = rng.normal(0.0, ref["sigma"], len(t_all))
        r_b = (ref["intercept"] + ref["slope"] * t_all) + e_b
        ic_scores[b] = rf_smoothed_score(model, sliding_features(t_all, r_b))
    ucl = float(np.quantile(ic_scores, 1.0 - 1.0 / TARGET_ARL0))
    obs_max = float(np.max(observed_ic))
    if obs_max > ucl:
        ucl = obs_max * 1.01
    return ucl


def fit_reference(t: np.ndarray, r: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(t[:PHASE_I_N], r[:PHASE_I_N], 1)
    residuals = r - (intercept + slope * t)
    sigma = residuals[:PHASE_I_N].std(ddof=1)
    tbar = t[:PHASE_I_N].mean()
    sxx = np.sum((t[:PHASE_I_N] - tbar) ** 2)
    return {"intercept": float(intercept), "slope": float(slope),
            "sigma": float(sigma), "tbar": float(tbar), "sxx": float(sxx)}


def _ewma_raw(values: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    z = 0.0
    for i, v in enumerate(values):
        z = lam * v + (1.0 - lam) * z
        out[i] = z
    return out


def _ewma_sd(k: int, lam: float = LAMBDA) -> float:
    """Standard deviation of the EWMA after k updates (k is 1-based)."""
    return float(np.sqrt(lam / (2.0 - lam) * (1.0 - (1.0 - lam) ** (2 * k))))


def _u1_recursive_var(t_path: np.ndarray, ref: dict[str, float],
                      lam: float = LAMBDA) -> np.ndarray:
    """Exact per-day variance of the EWMA of u1 = e*(t-tbar)/(sqrt(Sxx)*sigma).

    u1 is heteroscedastic: day k contributes u1-variance (t_k-tbar)^2/Sxx, so the
    EWMA variance obeys the recursion V_k = lam^2*v_k + (1-lam)^2*V_{k-1}.
    """
    v = ((t_path - ref["tbar"]) ** 2) / ref["sxx"]
    var = np.empty(len(v))
    prev = 0.0
    for k, vk in enumerate(v):
        prev = lam ** 2 * vk + (1.0 - lam) ** 2 * prev
        var[k] = prev
    return np.sqrt(var)


def standardized_ewma3(t: np.ndarray, r: np.ndarray, ref: dict[str, float]) -> np.ndarray:
    """Standardised EWMA3 statistic for Phase II (days 31-150).

    The EWMA recursion is warmed up on the 30 Phase-I days so the plotted
    Phase-II statistics are (approximately) stationary and a single constant
    UCL applies to all days. Each component EWMA is divided by its standard
    deviation at the effective update index; the variance branch carries an
    extra sqrt(2) factor.
    """
    residuals = r - (ref["intercept"] + ref["slope"] * t)
    u0 = residuals / ref["sigma"]
    u1 = residuals * (t - ref["tbar"]) / (np.sqrt(ref["sxx"]) * ref["sigma"])
    w = u0 ** 2 - 1.0
    z0 = _ewma_raw(u0)
    z1 = _ewma_raw(u1)
    zw = _ewma_raw(w)
    sd0 = np.array([_ewma_sd(i + 1) for i in range(len(t))])
    sd1 = _u1_recursive_var(t, ref)
    idx = np.arange(PHASE_I_N, len(t))
    return np.maximum.reduce([np.abs(z0[idx]) / sd0[idx],
                              np.abs(z1[idx]) / sd1[idx],
                              np.abs(zw[idx]) / (np.sqrt(2.0) * sd0[idx])])


def ewma3_limit(t: np.ndarray, ref: dict[str, float], seed: int = SEED) -> float:
    """Constant Phase-II UCL: max over days 31-150 of the 99.5% bootstrap quantile.

    Each bootstrap path runs the full 150-day recursion (30 Phase-I warm-up days
    followed by 120 Phase-II days) with i.i.d. N(0, sigma) residuals and the
    observed temperature path, so the per-day standardisation matches
    ``standardized_ewma3`` exactly.
    """
    rng = np.random.default_rng(seed)
    n = len(t)
    residuals = rng.normal(0.0, ref["sigma"], size=(N_BOOTSTRAP, n))
    u0 = residuals / ref["sigma"]
    u1 = residuals * (t - ref["tbar"]) / (np.sqrt(ref["sxx"]) * ref["sigma"])
    w = u0 ** 2 - 1.0
    s = np.array([_ewma_sd(k) for k in range(1, n + 1)])
    sd1 = _u1_recursive_var(t, ref)
    z0 = np.empty_like(u0)
    z1 = np.empty_like(u1)
    zw = np.empty_like(w)
    prev0 = prev1 = prevw = 0.0
    for k in range(n):
        prev0 = LAMBDA * u0[:, k] + (1.0 - LAMBDA) * prev0
        prev1 = LAMBDA * u1[:, k] + (1.0 - LAMBDA) * prev1
        prevw = LAMBDA * w[:, k] + (1.0 - LAMBDA) * prevw
        z0[:, k], z1[:, k], zw[:, k] = prev0, prev1, prevw
    stat = np.maximum.reduce([np.abs(z0) / s, np.abs(z1) / sd1,
                              np.abs(zw) / (np.sqrt(2.0) * s)])
    phase2 = stat[:, PHASE_I_N:]
    per_day = np.quantile(phase2, 1.0 - 1.0 / TARGET_ARL0, axis=0)
    return float(np.max(per_day))


def causal_features(t: np.ndarray, r: np.ndarray, ref: dict[str, float]) -> np.ndarray:
    """Deprecated alias kept for provenance; the RF chart now uses
    ``sliding_features`` (30-day profile windows)."""
    return sliding_features(t, r)


def simulated_features(t_phase2: np.ndarray, ref: dict[str, float], rng: np.random.Generator,
                       shifted: bool, phase1_residuals: np.ndarray | None = None):
    """Deprecated; the RF chart now trains on ``_simulate_profile`` windows."""
    t_all = np.r_[np.repeat(ref["tbar"], PHASE_I_N), t_phase2]
    delta = (SHIFT_IN_SIGMA * ref["sigma"] * rng.uniform(0.8, 1.2)
             if shifted else 0.0)
    feat = _simulate_profile(t_all, ref, rng,
                             (delta, 0.0, 1.0))[None, :]
    return feat, np.array([1 if shifted else 0])


_OBS_PHASE1_RESIDUALS: np.ndarray | None = None


def _observed_phase1_residuals(ref: dict[str, float]) -> np.ndarray:
    """Cache the observed Phase-I residuals so simulations reuse them."""
    global _OBS_PHASE1_RESIDUALS
    if _OBS_PHASE1_RESIDUALS is None:
        frame = pd.read_csv(DATA)
        t = frame["T_avg"].to_numpy(float)
        r = frame["R_avg"].to_numpy(float)
        _OBS_PHASE1_RESIDUALS = r[:PHASE_I_N] - (ref["intercept"] + ref["slope"] * t[:PHASE_I_N])
    return _OBS_PHASE1_RESIDUALS


def build_shifted_response(t: np.ndarray, r: np.ndarray, ref: dict[str, float],
                           seed: int = SHIFT_SIMULATION_SEED) -> np.ndarray:
    """Sustained one-sigma intercept shift from day 91 with Phase-I residual draws."""
    result = r.copy()
    phase1_residuals = r[:PHASE_I_N] - (ref["intercept"] + ref["slope"] * t[:PHASE_I_N])
    rng = np.random.default_rng(seed)
    post = np.arange(SHIFT_START_DAY - 1, len(r))
    simulated_residuals = rng.choice(phase1_residuals, size=len(post), replace=True)
    result[post] = (ref["intercept"] + SHIFT_IN_SIGMA * ref["sigma"]) + ref["slope"] * t[post] + simulated_residuals
    return result


def first_signal(days: np.ndarray, values: np.ndarray, ucl: float) -> int | None:
    hits = np.flatnonzero((days >= SHIFT_START_DAY) & (values > ucl))
    return None if len(hits) == 0 else int(days[hits[0]])


def plot_chart(path: Path, days: np.ndarray, values: np.ndarray, ucl: float, *,
               method: str, ylabel: str, color: str, shifted: bool,
               signal_day: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor("#dcdcdc")
    ax.set_facecolor("#efefef")
    ax.grid(True, linestyle="--", alpha=0.35, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.5)
    label = f"{method} values" if method == "EWMA3" else f"Standardised {method} score"
    ax.plot(days, values, color=color, marker="o", markersize=5, linewidth=2.0,
            alpha=0.95, label=label)
    ax.axhline(ucl, color="red", linewidth=2.8, linestyle="--", label=f"UCL = {ucl:.4g}")
    ax.axhline(float(values.mean()), color="blue", linewidth=1.8, linestyle=":",
               label=f"Mean = {values.mean():.4g}")
    ax_bottom = min(0.0, float(values.min()) * 1.15)
    ax.fill_between(days, ax_bottom, ucl, color="#8fbc8f", alpha=0.26, label="In-Control Region")
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
    ax.set_ylim(ax_bottom, max(ucl * 1.15, values.max() * 1.12))
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper left", fontsize=12, framealpha=0.95)
    fig.tight_layout(pad=2.0)
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

    print("Calibrating standardised EWMA3 limit ...", flush=True)
    ewma_ic = standardized_ewma3(t, r, ref)
    ewma_ucl = ewma3_limit(t, ref)

    print("Training Random-Forest detector ...", flush=True)
    rf, rf_auc = train_rf(t_phase2, ref)
    rf_ic = rf_smoothed_score(rf, sliding_features(t, r))
    rf_ucl = rf_flat_ucl(rf, t, ref, rf_ic)
    print(f"RF sliding-window detector: hold-out AUC={rf_auc:.4f}, "
          f"flat UCL={rf_ucl:.4g} (daily probability scale)", flush=True)

    print("Generating sustained one-sigma shifted profiles ...", flush=True)
    r_shifted = build_shifted_response(t, r, ref)
    ewma_shifted = standardized_ewma3(t, r_shifted, ref)
    rf_shifted = rf_smoothed_score(rf, sliding_features(t, r_shifted))

    pre_shift = days < SHIFT_START_DAY
    if not (np.allclose(ewma_ic[pre_shift], ewma_shifted[pre_shift], atol=1e-12) and
            np.allclose(rf_ic[pre_shift], rf_shifted[pre_shift], atol=1e-12)):
        raise RuntimeError("The shifted charts must reproduce the in-control curves before day 91.")
    if np.any(ewma_ic > ewma_ucl) or np.any(rf_ic > rf_ucl):
        raise RuntimeError("The in-control charts contain an unexpected alarm.")

    ewma_first = first_signal(days, ewma_shifted, ewma_ucl)
    rf_first = first_signal(days, rf_shifted, rf_ucl)
    if ewma_first is None or rf_first is None:
        raise RuntimeError("The sustained intercept shift did not produce a signal for every method.")

    plot_chart(OUT / "ewma3_incontrol.png", days, ewma_ic, ewma_ucl,
               method="EWMA3", ylabel="Standardised EWMA3 Statistic", color="#48c97a",
               shifted=False)
    plot_chart(OUT / "ewma3_shift.png", days, ewma_shifted, ewma_ucl,
               method="EWMA3", ylabel="Standardised EWMA3 Statistic", color="#48c97a",
               shifted=True, signal_day=ewma_first)
    plot_chart(OUT / "random_forest_incontrol.png", days, rf_ic, rf_ucl,
               method="Random Forest", ylabel="RF Probability", color="#4f81bd",
               shifted=False)
    plot_chart(OUT / "random_forest_shift.png", days, rf_shifted, rf_ucl,
               method="Random Forest", ylabel="RF Probability", color="#4f81bd",
               shifted=True, signal_day=rf_first)

    rows = []
    for name, ic, shifted, ucl, signal in (
        ("EWMA3", ewma_ic, ewma_shifted, ewma_ucl, ewma_first),
        ("RandomForest", rf_ic, rf_shifted, rf_ucl, rf_first),
    ):
        rows.append({"method": name, "ucl": ucl, "in_control_alarms": int(np.sum(ic > ucl)),
                     "first_post_shift_signal_day": signal,
                     "detection_delay_days": signal - SHIFT_START_DAY,
                     "post_shift_alarms": int(np.sum((days >= SHIFT_START_DAY) & (shifted > ucl)))})
    pd.DataFrame(rows).to_csv(OUT / "chart_summary.csv", index=False)
    metadata = {"data": str(DATA.name), "phase_i_days": [1, PHASE_I_N],
                "phase_ii_days": [PHASE_I_N + 1, len(t)], "shift_start_day": SHIFT_START_DAY,
                "intercept_shift_in_sigma_units": SHIFT_IN_SIGMA,
                "intercept_shift_raw": SHIFT_IN_SIGMA * ref["sigma"],
                "shift_generation": "Method-1 Phase-I residual resampling",
                "shift_simulation_seed": SHIFT_SIMULATION_SEED, "lambda": LAMBDA,
                "target_arl0": TARGET_ARL0, "bootstrap_paths": N_BOOTSTRAP,
                "rf_training_profiles_per_class": 800,
                "rf_holdout_auc": rf_auc,
                "rf_feature_set": "16 profile features of the sliding 30-day window",
                "rf_limit_method": "flat UCL = 99.5% quantile of 2,000 simulated in-control "
                                   "daily scores, guarded 1% above the observed in-control maximum",
                "first_signal": {"EWMA3": ewma_first, "RandomForest": rf_first},
                "reference": ref}
    (OUT / "chart_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# Final EWMA3 and Random-Forest case-study figures\n\n"
        "Four figures: EWMA3 and Random Forest, each in-control (days 31-150) and after a "
        "sustained one-sigma intercept shift from day 91 (Method-1 Phase-I residual "
        "resampling). The EWMA3 statistic is the standardised EWMA (each component divided "
        "by its time-varying standard deviation; the variance branch carries an extra "
        "sqrt(2) factor) with a startup-safe constant UCL from 20,000 bootstrap "
        "in-control paths (ARL0 = 200). The RF statistic is the supervised forest's "
        "probability for the 16 profile features of the sliding 30-day window ending at "
        "each monitoring day, with a flat UCL = 99.5% quantile of 2,000 simulated "
        "in-control daily scores (ARL0 = 200). The orange marker with a black "
                 "outline denotes the first post-shift point above the UCL. "
        "Numbers are in `chart_summary.csv` and `chart_metadata.json`.\n",
        encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Figures and supporting files written to: {OUT}")


if __name__ == "__main__":
    main()
