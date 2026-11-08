'''Phase-II demonstration for the combined transformer case study.

The four transformers are represented by T_avg and R_avg. The first 30 observed
combined measurements form the Phase-I reference profile. The remaining observed
measurements are partitioned into five subsequent 30-point profiles and monitored
with (i) a Hotelling T2 chart on (beta0, beta1, log residual variance), and (ii)
a Gradient-Boosting ML detector trained on transparent IC/OC profile-parameter
simulations calibrated from the Phase-I reference distribution.

This is an observed-data demonstration: no observed Phase-II profile is labelled
OC in advance. A separate optional synthetic OC sensitivity check is reported only
to verify that the two detectors respond to the shifts used in the manuscript.
'''
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'case_study_transformer_data.csv'
OUT = Path(__file__).resolve().parent / 'outputs'
SEED = 20260825
PROFILE_LENGTH = 30

def fit_profile(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    return float(beta[0]), float(beta[1]), float(np.var(e, ddof=1))


def profile_vector(x, y):
    b0, b1, var = fit_profile(x, y)
    return np.array([b0, b1, np.log(max(var, 1e-30))], dtype=float)


def observed_profiles(data, length=PROFILE_LENGTH):
    rows = []
    x = data.T_avg.to_numpy(float)
    y = data.R_avg.to_numpy(float)
    for start in range(0, len(data) - length + 1, length):
        end = start + length
        vec = profile_vector(x[start:end], y[start:end])
        rows.append({'profile_id': f'OBS_{start // length + 1:02d}',
                     'day_start': start + 1, 'day_end': end,
                     'beta_hat_0': vec[0], 'beta_hat_1': vec[1],
                     'log_error_variance_hat': vec[2],
                     'error_variance_hat': float(np.exp(vec[2])),
                     'observed_phase': 'Phase I' if start == 0 else 'Phase II'})
    return pd.DataFrame(rows)


def reference_from_table(table):
    ic = table.loc[table.status.eq('IC')].copy()
    z = ic[['beta_hat_0', 'beta_hat_1', 'error_variance_hat']].to_numpy(float)
    z[:, 2] = np.log(np.maximum(z[:, 2], 1e-30))
    center = z.mean(axis=0)
    cov = np.cov(z, rowvar=False) + 1e-10 * np.eye(3)
    return z, center, cov

def t2_values(z, center, cov):
    inv = np.linalg.inv(cov)
    d = z - center
    return np.einsum('ij,jk,ik->i', d, inv, d)


def train_ml_reference(z, seed=SEED):
    rng = np.random.default_rng(seed)
    n = max(1200, len(z) * 60)
    ic = z[rng.integers(0, len(z), n)] + rng.multivariate_normal(np.zeros(3),
        np.diag(np.maximum(np.var(z, axis=0, ddof=1), 1e-12)) * 0.25, n)
    oc = ic.copy()
    choices = rng.choice(3, n)
    oc[np.arange(n), choices] += rng.choice([-1.0, 1.0], n) * np.array([0.00008, 0.00045, 0.7])[choices]
    # Add a subset of simultaneous changes, matching the case-study shift logic.
    idx = rng.random(n) < .35
    oc[idx, 0] += .00008
    oc[idx, 1] += .00045
    oc[idx, 2] += np.log(2.0)
    X = np.vstack([ic, oc]); labels = np.r_[np.zeros(n), np.ones(n)]
    model = make_pipeline(StandardScaler(), GradientBoostingClassifier(
        n_estimators=150, max_depth=2, random_state=seed))
    model.fit(X, labels)
    return model, ic

def limits(z_ic, t2_ic, ml_ic):
    # A transparent Phase-II limit: 99% empirical IC quantile, with a chi-square
    # reference shown for context. The empirical limit is used for both charts.
    t2_limit = float(np.quantile(t2_ic, .99))
    ml_limit = float(np.quantile(ml_ic, .99))
    return {'T2_limit_empirical_99pct': t2_limit,
            'T2_chi2_99pct_df3': float(chi2.ppf(.99, 3)),
            'ML_limit_empirical_99pct': ml_limit}


def synthetic_check(center, cov, model, limits_, seed=SEED):
    rng = np.random.default_rng(seed)
    base = rng.multivariate_normal(center, cov, 1000)
    scenarios = []
    for name, shift in [('intercept', np.array([.00008, 0., 0.])),
                        ('slope', np.array([0., .00045, 0.])),
                        ('variance', np.array([0., 0., np.log(2.)]))]:
        z = base[:100] + shift
        t2 = t2_values(z, center, cov)
        ml = model.predict_proba(z)[:, 1]
        scenarios.append({'scenario': name,
                          'T2_mean': float(t2.mean()),
                          'ML_mean_probability': float(ml.mean()),
                          'T2_detection_rate': float(np.mean(t2 > limits_['T2_limit_empirical_99pct'])),
                          'ML_detection_rate': float(np.mean(ml > limits_['ML_limit_empirical_99pct']))})
    return scenarios

def plot_results(results, limits_, out):
    phase2 = results[results.observed_phase.eq('Phase II')]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=180, sharex=True)
    axes[0].plot(phase2.profile_id, phase2.T2_statistic, 'o-', label='T2')
    axes[0].axhline(limits_['T2_limit_empirical_99pct'], color='r', ls='--', label='99% IC limit')
    axes[0].set_ylabel('T2 statistic'); axes[0].set_title('Phase-II Hotelling T2 monitoring'); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(phase2.profile_id, phase2.ML_probability, 'o-', color='#2a9d8f', label='ML probability')
    axes[1].axhline(limits_['ML_limit_empirical_99pct'], color='r', ls='--', label='99% IC limit')
    axes[1].set_ylabel('ML score'); axes[1].set_xlabel('Observed subsequent profile'); axes[1].set_title('Phase-II ML monitoring'); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.tight_layout(); fig.savefig(out / 'phase2_T2_ML_monitoring.png'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=DATA)
    ap.add_argument('--profile-table', type=Path, default=OUT / 'combined_transformer_profiles_all.csv')
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data); table = pd.read_csv(args.profile_table)
    obs = observed_profiles(data)
    z_ic, center, cov = reference_from_table(table)
    t2_ic = t2_values(z_ic, center, cov)
    model, ml_ic_sample = train_ml_reference(z_ic, args.seed)
    ml_ic = model.predict_proba(ml_ic_sample)[:, 1]
    lim = limits(t2_ic, t2_ic, ml_ic)
    z_obs = obs[['beta_hat_0', 'beta_hat_1', 'log_error_variance_hat']].to_numpy(float)
    obs['T2_statistic'] = t2_values(z_obs, center, cov)
    obs['ML_probability'] = model.predict_proba(z_obs)[:, 1]
    obs['T2_signal'] = obs.T2_statistic > lim['T2_limit_empirical_99pct']
    obs['ML_signal'] = obs.ML_probability > lim['ML_limit_empirical_99pct']
    obs['interpretation'] = np.where(obs.T2_signal & obs.ML_signal, 'both charts signal', np.where(obs.T2_signal, 'T2 only signals', np.where(obs.ML_signal, 'ML only signals', 'no signal')))
    obs.to_csv(OUT / 'phase2_observed_monitoring.csv', index=False)
    synthetic = synthetic_check(center, cov, model, lim, args.seed)
    pd.DataFrame(synthetic).to_csv(OUT / 'phase2_synthetic_sensitivity_check.csv', index=False)
    plot_results(obs, lim, OUT)
    signals = obs[obs.observed_phase.eq('Phase II') & (obs.T2_signal | obs.ML_signal)]
    metadata = {'analysis_unit': 'combined T_avg/R_avg transformer profile', 'phase1_profile': 'first 30 observed measurements plus IC bootstrap reference', 'phase2_profiles': int(sum(obs.observed_phase.eq('Phase II'))), 'limits': lim, 'n_phase2_signals': int(len(signals)), 'signal_profile_ids': signals.profile_id.tolist(), 'synthetic_sensitivity_check': synthetic}
    (OUT / 'phase2_monitoring_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    lines = ['# Phase II monitoring demonstration', '', '## Design', '', 'The four transformer channels were represented by the combined `T_avg` and `R_avg` series. The first 30 observed measurements were treated as the initial Phase-I reference profile; the remaining 120 observations were divided into four subsequent 30-point Phase-II profiles. Each profile was reduced to beta-hat-0, beta-hat-1, and log residual variance.', '', f"The empirical 99% Phase-I limits were T2 = {lim['T2_limit_empirical_99pct']:.6g} and ML score = {lim['ML_limit_empirical_99pct']:.6g}. The chi-square reference value for T2 with 3 degrees of freedom is {lim['T2_chi2_99pct_df3']:.4f}; the empirical limit was used because the profile reference is finite and estimated.", '', '## Observed Phase-II profiles', '', '| Profile | Days | T2 | T2 signal | ML probability | ML signal | Interpretation |', '|:--|:--|--:|:--:|--:|:--:|:--|']
    for _, r in obs[obs.observed_phase.eq('Phase II')].iterrows():
        lines.append(f"| {r.profile_id} | {int(r.day_start)}–{int(r.day_end)} | {r.T2_statistic:.4f} | {bool(r.T2_signal)} | {r.ML_probability:.4f} | {bool(r.ML_signal)} | {r.interpretation} |")
    lines += ['', '## Interpretation', '']
    if len(signals) == 0:
        lines.append('Neither chart signalled on the observed subsequent profiles. This indicates that the combined transformer response remained within the Phase-I operating envelope during the observed Phase-II period; it is not evidence that future faults are impossible.')
    else:
        lines.append(f"{len(signals)} subsequent profile(s) generated at least one signal: {', '.join(signals.profile_id)}. These profiles should be inspected for changes in fitted slope, intercept, or residual variability before maintenance conclusions are drawn.")
    lines += ['', 'A synthetic sensitivity check is reported separately in `phase2_synthetic_sensitivity_check.csv`. It verifies that the implementation responds to the manuscript shifts; it is not presented as an observed field alarm.']
    (OUT / 'phase2_monitoring_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Phase-II demonstration complete ->', OUT)

if __name__ == '__main__': main()
