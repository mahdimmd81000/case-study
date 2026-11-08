'''Reproducible EDA for the transformer case study.

The source file contains four longitudinal temperature-response profiles (TR1--TR4),
with one response resistance observed at each day/temperature. The reviewer asks
for profile plots, residual distributions, and a summary of fitted profile
parameters. This script produces all requested items without changing the
simulation/reviewer-benchmark files.
'''
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import normaltest, shapiro, skew
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "case_study_transformer_data.csv"
OUT = Path(__file__).resolve().parent / "outputs"


def fit_profiles(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    summary_rows = []
    for i in range(1, 5):
        name = f"TR{i}"
        x = data[f"T{i}"].to_numpy(float)
        y = data[f"R{i}"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        resid = y - fitted
        for day, temp, response, fit, error in zip(
            data["day"], x, y, fitted, resid
        ):
            long_rows.append({
                "transformer": name, "day": int(day),
                "temperature": temp, "response": response,
                "fitted_response": fit, "residual": error,
            })
        summary_rows.append({
            "transformer": name, "n_observations": len(x),
            "temperature_min": x.min(), "temperature_max": x.max(),
            "temperature_mean": x.mean(),
            "response_min": y.min(), "response_max": y.max(),
            "response_mean": y.mean(), "intercept": intercept,
            "slope": slope,
            "error_variance": np.var(resid, ddof=1),
            "error_sd": np.std(resid, ddof=1),
            "residual_mean": resid.mean(),
            "residual_skewness": skew(resid, bias=False),
            "residual_q25": np.quantile(resid, .25),
            "residual_median": np.median(resid),
            "residual_q75": np.quantile(resid, .75),
            "residual_max_abs": np.max(np.abs(resid)),
        })
    return pd.DataFrame(long_rows), pd.DataFrame(summary_rows)


def plot_scatter_profiles(data: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    for i, color in zip(range(1, 5), colors):
        x, y = data[f"T{i}"], data[f"R{i}"]
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.scatter(x, y, s=24, alpha=.72, color=color, label=f"TR{i}")
        ax.plot(grid, intercept + slope * grid, color=color, lw=1.5)
    ax.set_title("Temperature–response scatterplots for all available profiles")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Response (resistance)")
    ax.grid(alpha=.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out / "temperature_response_all_profiles.png")
    plt.close(fig)


def plot_profile_panels(data: pd.DataFrame, out: Path, requested: int = 10) -> None:
    # The source has four profiles; do not fabricate six additional profiles.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180, sharex=False, sharey=False)
    for i, ax in enumerate(axes.flat, 1):
        x, y = data[f"T{i}"], data[f"R{i}"]
        slope, intercept = np.polyfit(x, y, 1)
        grid = np.linspace(x.min(), x.max(), 100)
        ax.scatter(x, y, s=20, alpha=.75, color="#1769aa")
        ax.plot(grid, intercept + slope * grid, color="#d1495b", lw=1.5)
        ax.set_title(f"TR{i}: fitted line")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Response")
        ax.grid(alpha=.2)
    fig.suptitle(
        f"Individual profile fits ({min(4, requested)} of 4 available profiles)",
        y=.995,
    )
    fig.tight_layout()
    fig.savefig(out / "individual_profile_panels.png")
    plt.close(fig)


def plot_residual_distributions(long: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    groups = [long.loc[long.transformer == f"TR{i}", "residual"] for i in range(1, 5)]
    axes[0].hist(groups, bins=14, alpha=.52, label=[f"TR{i}" for i in range(1, 5)])
    axes[0].set_title("Residual histograms by profile")
    axes[0].set_xlabel("Residual")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
    axes[0].grid(alpha=.2)
    axes[1].boxplot(groups, labels=[f"TR{i}" for i in range(1, 5)], showmeans=True)
    axes[1].set_title("Residual boxplots by profile")
    axes[1].set_xlabel("Profile")
    axes[1].set_ylabel("Residual")
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / "residual_histograms_boxplots.png")
    plt.close(fig)


def plot_parameter_summary(summary: pd.DataFrame, out: Path) -> None:
    params = ["intercept", "slope", "error_variance"]
    labels = ["Intercept", "Slope", "Error variance"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), dpi=180)
    for ax, col, label in zip(axes, params, labels):
        ax.bar(summary.transformer, summary[col], color="#4c78a8")
        ax.set_title(label)
        ax.set_xlabel("Profile")
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("Estimated profile parameters across transformers", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "profile_parameter_summary.png")
    plt.close(fig)


def make_report(data: pd.DataFrame, long: pd.DataFrame, summary: pd.DataFrame,
                out: Path, requested: int) -> None:
    all_resid = long.residual.to_numpy()
    report = {
        "source": str(DEFAULT_DATA),
        "n_profiles_available": 4,
        "requested_profile_plots": requested,
        "note": "The source dataset contains four transformer profiles; no synthetic profiles were added.",
        "n_observations_per_profile": summary.n_observations.tolist(),
        "missing_values": int(data.isna().sum().sum()),
        "overall_temperature_range": [float(data[[f"T{i}" for i in range(1, 5)]].min().min()),
                                      float(data[[f"T{i}" for i in range(1, 5)]].max().max())],
        "overall_response_range": [float(data[[f"R{i}" for i in range(1, 5)]].min().min()),
                                  float(data[[f"R{i}" for i in range(1, 5)]].max().max())],
        "overall_residual_mean": float(all_resid.mean()),
        "overall_residual_sd": float(all_resid.std(ddof=1)),
        "overall_residual_skewness": float(skew(all_resid, bias=False)),
        "shapiro_pvalue_overall_residuals": float(shapiro(all_resid).pvalue),
        "normaltest_pvalue_overall_residuals": float(normaltest(all_resid).pvalue),
    }
    (out / "eda_quality_and_diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Case-study exploratory data analysis",
        "",
        f"- Source: `{DEFAULT_DATA.name}`",
        f"- Profiles available: **4** (TR1–TR4); requested plotting capacity: {requested}.",
        "- No synthetic profiles were created. The four available profiles are all plotted.",
        f"- Observations per profile: {int(summary.n_observations.iloc[0])}",
        f"- Missing values: {int(data.isna().sum().sum())}",
        "",
        "## Generated figures",
        "",
        "1. `temperature_response_all_profiles.png`: overlaid temperature–response scatterplots and fitted lines.",
        "2. `individual_profile_panels.png`: one panel for each available transformer profile.",
        "3. `residual_histograms_boxplots.png`: residual histograms and boxplots.",
        "4. `profile_parameter_summary.png`: intercept, slope, and error-variance summaries.",
        "",
        "## Estimated profile parameters",
        "",
        "| Profile | Intercept | Slope | Error variance | Error SD | Residual skewness |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.transformer} | {r.intercept:.8g} | {r.slope:.8g} | "
            f"{r.error_variance:.8g} | {r.error_sd:.8g} | {r.residual_skewness:.4f} |"
        )
    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Each transformer is treated as one simple linear temperature–response profile.",
        "- The fitted line is estimated separately for each transformer using all available observations.",
        "- Error variance is the unbiased sample variance of residuals (`ddof=1`).",
        "- Residual histograms and boxplots are descriptive diagnostics; they are not used to remove observations.",
        "- The residual tests are reported as diagnostics, not as a substitute for the plots.",
    ]
    (out / "EDA_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--requested-profiles", type=int, default=10)
    args = parser.parse_args()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data)
    long, summary = fit_profiles(data)
    long.to_csv(out / "profile_observations_and_residuals.csv", index=False)
    summary.to_csv(out / "profile_parameter_summary.csv", index=False)
    plot_scatter_profiles(data, out)
    plot_profile_panels(data, out, args.requested_profiles)
    plot_residual_distributions(long, out)
    plot_parameter_summary(summary, out)
    make_report(data, long, summary, out, args.requested_profiles)
    print(f"EDA complete: {len(summary)} available profiles, outputs -> {out}")

if __name__ == "__main__":
    main()
