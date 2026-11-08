"Independent direct-path bootstrap audit of the three EWMA3 limits."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ewma3_six_component_charts as analysis

def main():
    d = pd.read_csv(analysis.DATA)
    t = d["T_avg"].to_numpy(float)
    r = d["R_avg"].to_numpy(float)
    slope, intercept = np.polyfit(t[:30], r[:30], 1)
    resid = r - (intercept + slope * t)
    tbar = float(t[:30].mean())
    ref = {"a0": float(intercept), "a1": float(slope),
           "sigma": float(resid[:30].std(ddof=1)), "tbar": tbar,
           "sxx": float(np.sum((t[:30] - tbar) ** 2)),
           "phase1_residuals": resid[:30].copy()}
    calibrated = analysis.calibrate_limits(t, ref)

    # Independently reproduce bootstrap maxima with scalar recursion initialized
    # from the actual Phase-I EWMA states.
    rng = np.random.default_rng(analysis.SEED)
    t2 = t[analysis.PHASE_I_N:]
    h = len(t2)
    sd_i = analysis.intercept_ewma_sd(len(t))[analysis.PHASE_I_N:]
    sd_s = analysis.slope_ewma_sd(t, ref["tbar"], ref["sxx"])[analysis.PHASE_I_N:]
    ep1 = ref["phase1_residuals"] / ref["sigma"]
    z0_start = analysis.ewma(ep1)[-1]
    z1_start = analysis.ewma(ep1 * (t[:30] - ref["tbar"]) / np.sqrt(ref["sxx"]))[-1]
    zv_start = analysis.ewma((ep1**2 - 1) / np.sqrt(2))[-1]
    maxima = {name: np.empty(analysis.N_BOOT) for name in analysis.COMPONENTS}
    for b in range(analysis.N_BOOT):
        errors = rng.normal(0.0, 1.0, h)
        z0, z1, zv = z0_start, z1_start, zv_start
        largest = {name: 0.0 for name in analysis.COMPONENTS}
        for k, error in enumerate(errors):
            q1 = error * (t2[k] - ref["tbar"]) / np.sqrt(ref["sxx"])
            qv = (error**2 - 1.0) / np.sqrt(2.0)
            z0 = analysis.LAMBDA * error + (1.0 - analysis.LAMBDA) * z0
            z1 = analysis.LAMBDA * q1 + (1.0 - analysis.LAMBDA) * z1
            zv = analysis.LAMBDA * qv + (1.0 - analysis.LAMBDA) * zv
            largest["Intercept"] = max(largest["Intercept"], abs(z0 / sd_i[k]))
            largest["Slope"] = max(largest["Slope"], abs(z1 / sd_s[k]))
            largest["Residual SD"] = max(largest["Residual SD"], abs(zv / sd_i[k]))
        for name in analysis.COMPONENTS:
            maxima[name][b] = largest[name]
    q = 1.0 - 1.0 / analysis.ARL0
    audited = {name: float(np.quantile(values, q)) for name, values in maxima.items()}

    print("Main calibration:", calibrated)
    print("Independent path-loop calibration:", audited)
    print("Absolute differences:", {k: abs(calibrated[k] - audited[k]) for k in calibrated})
    if any(abs(calibrated[k] - audited[k]) > 0.02 for k in calibrated):
        raise AssertionError("Independent pathwise limit audit mismatch exceeds Monte Carlo tolerance")
    observed = analysis.calculate_components(t, r, ref)
    shifted_response = r.copy()
    shifted_response[analysis.SHIFT_DAY - 1:] += analysis.SHIFT_DELTA
    shifted = analysis.calculate_components(t, shifted_response, ref)
    days = np.arange(31, 151)
    for name in analysis.COMPONENTS:
        ucl = audited[name]
        # Intercept and slope are signed/two-sided; residual-SD is absolute.
        lcl = 0.0 if name == "Residual SD" else -ucl
        out_ic = (observed[name] < lcl) | (observed[name] > ucl)
        out_sh = (shifted[name] < lcl) | (shifted[name] > ucl)
        sh_days = days[(days >= analysis.SHIFT_DAY) & out_sh]
        print(f"{name}: LCL={lcl:.9f}, UCL={ucl:.9f}, original range="
              f"[{observed[name].min():.9f}, {observed[name].max():.9f}], "
              f"shifted range=[{shifted[name].min():.9f}, {shifted[name].max():.9f}], "
              f"original alarms={out_ic.sum()}, shifted alarms={sh_days.tolist()}")
        if out_ic.any():
            raise AssertionError(f"Observed in-control data crosses {name} displayed limits")
    print("Limit calculations independently reproduced; observed and shifted values checked "
          "against each component's displayed LCL/UCL.")

if __name__ == "__main__":
    main()
