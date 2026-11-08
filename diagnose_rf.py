"""Diagnose why the observed in-control RF score distribution mismatches the simulated one."""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ewma3_rf_case_study_charts import (DATA, PHASE_I_N, N_CALIBRATION_PATHS, SEED,
                                        fit_reference, simulated_features, train_rf,
                                        causal_features, rf_smoothed_score)

frame = pd.read_csv(DATA)
t = frame["T_avg"].to_numpy(float); r = frame["R_avg"].to_numpy(float)
ref = fit_reference(t, r)
t_phase2 = t[PHASE_I_N:]
rf = train_rf(t_phase2, ref)

# Observed trace
obs = rf_smoothed_score(rf, causal_features(t, r, ref))
print("observed smoothed score: min %.4f max %.4f mean %.4f std %.4f" %
      (obs.min(), obs.max(), obs.mean(), obs.std()))

# Simulated in-control paths
rng = np.random.default_rng(SEED + 1)
X = rng.multivariate_normal([0, 1.0, 0.3, 0, 1.2], np.diag([1.0, 0.15, 0.2, 0.15, 0.12]), 2000)
prob = rf.predict_proba(X)[:, 1]
print("fake in-control cloud prob: min %.4f max %.4f mean %.4f" % (prob.min(), prob.max(), prob.mean()))
# second cloud that mimics the model's IC training features
rngb = np.random.default_rng(SEED + 2)
ic_ic = rngb.multivariate_normal([0, 1.0, 0.3, 0, 1.2], np.diag([1.0, 0.15, 0.2, 0.15, 0.12]), 1500)
ic_oc = rngb.multivariate_normal([0, 1.0, 0.3, 0, 1.2], np.diag([1.0, 0.15, 0.2, 0.15, 0.12]), 1500)
ic_oc[:, 0] += 1.0
prob_ic = rf.predict_proba(ic_ic)[:, 1]
prob_oc = rf.predict_proba(ic_oc)[:, 1]
print("IC-cloud prob mean %.4f sd %.4f ; OC-cloud prob mean %.4f sd %.4f" %
      (prob_ic.mean(), prob_ic.std(), prob_oc.mean(), prob_oc.std()))
# Training-domain probe: what probability does the model give to the exact
# *observed* Phase-II features versus a leave-one-day-out OC set shifted by
# +1 and -1 sigma?  This isolates whether the model saturates at the same
# high probability for both directions (sign-blindness).
obs_feat_full = causal_features(t, r, ref)
p_obs = rf.predict_proba(obs_feat_full)[:, 1]
p_up = rf.predict_proba(obs_feat_full + np.array([1.0, 0.6, 0.5, 0.5, -0.1]))[:, 1]
p_dn = rf.predict_proba(obs_feat_full + np.array([-1.0, 0.6, -0.5, -0.5, -0.1]))[:, 1]
print("observed-feature prob mean %.4f | shifted+1sd prob mean %.4f | shifted-1sd prob mean %.4f" %
      (p_obs.mean(), p_up.mean(), p_dn.mean()))
print("fraction of days where +1sd and -1sd both exceed 0.9: %.3f" %
      float(np.mean((p_up > 0.9) & (p_dn > 0.9))))
sims = np.concatenate([rf_smoothed_score(rf, simulated_features(t_phase2, ref, rng, shifted=False)[0])
                       for _ in range(N_CALIBRATION_PATHS)])
print("simulated smoothed score: min %.4f max %.4f mean %.4f std %.4f" %
      (sims.min(), sims.max(), sims.mean(), sims.std()))
print("simulated quantiles 50/90/99/99.5%%: %.4f %.4f %.4f %.4f" %
      tuple(np.quantile(sims, [0.5, 0.9, 0.99, 0.995])))

# Raw daily probabilities (unsmoothed) for comparison
obs_daily = rf.predict_proba(causal_features(t, r, ref))[:, 1]
print("observed DAILY prob: min %.4f max %.4f mean %.4f std %.4f" %
      (obs_daily.min(), obs_daily.max(), obs_daily.mean(), obs_daily.std()))
rng2 = np.random.default_rng(SEED + 1)
sim_daily = np.concatenate([rf.predict_proba(simulated_features(t_phase2, ref, rng2, shifted=False)[0])[:, 1]
                            for _ in range(50)])
print("simulated DAILY prob: min %.4f max %.4f mean %.4f std %.4f" %
      (sim_daily.min(), sim_daily.max(), sim_daily.mean(), sim_daily.std()))

# Feature comparison: observed vs simulated, per column
rng3 = np.random.default_rng(SEED + 1)
sim_feat = simulated_features(t_phase2, ref, rng3, shifted=False)[0]
obs_feat = causal_features(t, r, ref)
names = getattr(causal_features, "feature_names", None)
if names is None:
    names = [f"f{i}" for i in range(obs_feat.shape[1])]
print("\nfeature  |  observed mean   sd  |  simulated mean   sd")
for i, nm in enumerate(names):
    print(f"{nm:>18s} | {obs_feat[:,i].mean():12.5f} {obs_feat[:,i].std():9.5f} | "
          f"{sim_feat[:,i].mean():12.5f} {sim_feat[:,i].std():9.5f}")
