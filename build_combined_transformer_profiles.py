'''Build a case-study profile table after combining the four transformers.

The manuscript treats the four units as one representative transformer. Therefore
this script uses the supplied T_avg/R_avg columns as the single case-study profile.
It creates a transparent profile-level diagnostic sample: 20 in-control and 10
out-of-control profiles. Every row contains beta-hat-0, beta-hat-1, residual
variance, and the status/shift metadata.

The out-of-control defaults follow the shifts used in the existing reviewer
analysis: intercept +0.00008, slope +0.00045, and residual SD multiplied by 2.
Use --oc-shift-type to isolate one shift if desired.
'''
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "case_study_transformer_data.csv"
OUT = Path(__file__).resolve().parent / "outputs"
SEED = 20260825
N_IC, N_OC = 20, 10
PROFILE_LENGTH = 30

def fit_profile(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    beta1, beta0 = np.polyfit(x, y, 1)
    residual = y - (beta0 + beta1 * x)
    return float(beta0), float(beta1), float(np.var(residual, ddof=1))


def make_profiles(data: pd.DataFrame, n_ic: int, n_oc: int,
                  profile_length: int, shift_type: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = data["T_avg"].to_numpy(float)
    y = data["R_avg"].to_numpy(float)
    beta0, beta1, var0 = fit_profile(x, y)
    fitted = beta0 + beta1 * x
    residuals = y - fitted
    rows = []

    def one_profile(status: str, index: int) -> dict:
        # Bootstrap observed temperatures and residuals separately, preserving
        # the observed temperature range while retaining the fitted relationship.
        idx_x = rng.integers(0, len(x), profile_length)
        idx_e = rng.integers(0, len(residuals), profile_length)
        xp = x[idx_x]
        ep = residuals[idx_e]
        int_shift = slope_shift = .0
        sd_ratio = 1.0
        if status == "OC":
            if shift_type in ("intercept", "all"):
                int_shift = 0.00008
            if shift_type in ("slope", "all"):
                slope_shift = 0.00045
            if shift_type in ("variance", "all"):
                sd_ratio = 2.0
        yp = (beta0 + int_shift) + (beta1 + slope_shift) * xp + sd_ratio * ep
        b0, b1, v = fit_profile(xp, yp)
        return {
            "profile_id": f"{status}_{index:02d}",
            "status": status,
            "n_observations": profile_length,
            "beta_hat_0": b0,
            "beta_hat_1": b1,
            "error_variance_hat": v,
            "error_sd_hat": np.sqrt(v),
            "temperature_mean": float(xp.mean()),
            "response_mean": float(yp.mean()),
            "shift_type": shift_type if status == "OC" else "none",
            "intercept_shift": int_shift,
            "slope_shift": slope_shift,
            "sd_ratio": sd_ratio,
        }

    for i in range(1, n_ic + 1):
        rows.append(one_profile("IC", i))
    for i in range(1, n_oc + 1):
        rows.append(one_profile("OC", i))
    result = pd.DataFrame(rows)
    result.attrs["baseline_beta_hat_0"] = beta0
    result.attrs["baseline_beta_hat_1"] = beta1
    result.attrs["baseline_error_variance"] = var0
    return result

def write_report(result: pd.DataFrame, data: pd.DataFrame, shift_type: str,
                 profile_length: int, out: Path) -> None:
    ic = result[result.status == "IC"]
    oc = result[result.status == "OC"]
    base_b0, base_b1, base_v = fit_profile(
        data["T_avg"].to_numpy(float), data["R_avg"].to_numpy(float)
    )
    report = f'''# Combined-transformer case-study profile diagnostics
## Analysis unit
The four transformer series were combined into the single representative
case-study profile already used in the manuscript, using `T_avg` and `R_avg`.
No TR1--TR4 labels are used in the profile-level analysis.

- Source: `{DATA.name}`
- Observed combined series length: {len(data)}
- Profile length used for the diagnostic sample: {profile_length}
- In-control profiles: {len(ic)}
- Out-of-control profiles: {len(oc)}
- Out-of-control shift scenario: `{shift_type}`
- Baseline beta-hat-0: `{base_b0:.10g}`
- Baseline beta-hat-1: `{base_b1:.10g}`
- Baseline error variance: `{base_v:.10g}`

## Required profile estimates
| Profile | Status | beta-hat-0 | beta-hat-1 | Error variance-hat | Error SD-hat |
|:--|:--:|--:|--:|--:|--:|
'''
    for _, r in result.iterrows():
        report += (
            f"| {r.profile_id} | {r.status} | {r.beta_hat_0:.10g} | "
            f"{r.beta_hat_1:.10g} | {r.error_variance_hat:.10g} | "
            f"{r.error_sd_hat:.10g} |\n"
        )
    report += '''
## Interpretation
The table is a profile-level diagnostic sample derived from the combined
case-study series. The IC/OC label is known because the OC profiles are generated
with the stated shifts; it is not inferred from the fitted parameters. The three
reported estimates are the quantities subsequently used by the monitoring
procedures: intercept, slope, and residual error variance.
'''

    (out / "combined_transformer_profile_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--n-ic", type=int, default=N_IC)
    parser.add_argument("--n-oc", type=int, default=N_OC)
    parser.add_argument("--profile-length", type=int, default=PROFILE_LENGTH)
    parser.add_argument("--oc-shift-type", choices=["intercept", "slope", "variance", "all"], default="all")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data)
    result = make_profiles(data, args.n_ic, args.n_oc, args.profile_length,
                           args.oc_shift_type, args.seed)
    suffix = f"_{args.oc_shift_type}"
    result.to_csv(out / f"combined_transformer_profiles{suffix}.csv", index=False)
    write_report(result, data, args.oc_shift_type, args.profile_length, out)
    print(f"Wrote {len(result)} combined-transformer profiles to {out}")

if __name__ == "__main__":
    main()
