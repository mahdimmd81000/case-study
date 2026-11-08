import io
import json
import sys
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import gamma, kstest, normaltest, shapiro, skew, skewnorm, skewtest


# Safe UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass


N_POINTS = 150
STEP_HOURS = 12
ALPHA_CU = 0.00393
T_REF_C = 20.0
BASE_SEED = 20260220
CSV_DATA_PATH = Path(__file__).resolve().parent / "case_study_transformer_data.csv"


@dataclass(frozen=True)
class TransformerSpec:
    name: str
    r20: float
    ambient_offset: float
    rise_min: float
    rise_max: float
    temp_noise_std: float
    err_std: float
    skew_bounds: tuple
    seed_shift: int


SPECS = [
    TransformerSpec("TR1", 0.0878, -0.5, 16.5, 28.5, 0.55, 0.00020, (1.00, 1.08), 11),
    TransformerSpec("TR2", 0.0899, 0.2, 17.0, 30.0, 0.60, 0.00022, (1.06, 1.14), 23),
    TransformerSpec("TR3", 0.0859, -0.3, 16.0, 27.8, 0.52, 0.00019, (1.10, 1.18), 37),
    TransformerSpec("TR4", 0.0911, 0.4, 17.5, 31.2, 0.62, 0.00023, (1.12, 1.20), 47),
]


def _ar1_process(rng, n, phi, innovation_std):
    x = np.zeros(n, dtype=float)
    eps = rng.normal(0.0, innovation_std, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _simulate_temperature(spec, common_ambient, n, rng):
    t = np.arange(n, dtype=float)

    # Shared environment + transformer-specific loading dynamics.
    local_ambient = common_ambient + spec.ambient_offset + rng.normal(0.0, 0.35, size=n)
    load_state = _ar1_process(rng, n, phi=0.82, innovation_std=0.55)
    load_index = 1.0 / (1.0 + np.exp(-load_state))
    weekly_cycle = 0.08 * np.sin(2.0 * np.pi * t / 14.0 + 0.17 * spec.seed_shift)

    thermal_rise = spec.rise_min + (spec.rise_max - spec.rise_min) * np.clip(load_index + weekly_cycle, 0.0, 1.0)
    temp = local_ambient + thermal_rise + rng.normal(0.0, spec.temp_noise_std, size=n)

    # Typical LV winding/hot-spot operating range for distribution transformers.
    temp = np.clip(temp, 34.0, 88.0)
    return np.round(temp, 3)


def _simulate_skewed_error(rng, n, err_std, skew_bounds, max_trials=5000):
    low, high = skew_bounds

    for _ in range(max_trials):
        sigma_ln = rng.uniform(0.325, 0.375)
        base = rng.lognormal(mean=-7.95, sigma=sigma_ln, size=n)
        err = base - np.mean(base)
        err = err * (err_std / (np.std(err, ddof=1) + 1e-12))
        s = float(skew(err, bias=False))
        if low <= s <= high:
            return err, s

    raise RuntimeError(f"Could not generate residuals in skewness range {skew_bounds}")


def generate_realistic_dataset(n_points=N_POINTS, seed=BASE_SEED):
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=float)

    ambient_state = _ar1_process(rng, n_points, phi=0.72, innovation_std=0.28)
    common_ambient = (
        27.6
        + 0.9 * np.sin(2.0 * np.pi * t / 14.0 + 0.2)
        + 1.5 * np.sin(np.pi * t + 0.7)
        + ambient_state
        + rng.normal(0.0, 0.18, size=n_points)
    )

    out = {"meta": {"n_points": n_points, "step_hours": STEP_HOURS, "alpha_cu": ALPHA_CU, "t_ref_c": T_REF_C}}

    for i, spec in enumerate(SPECS, start=1):
        local_rng = np.random.default_rng(seed + spec.seed_shift)
        temp = _simulate_temperature(spec, common_ambient, n_points, local_rng)
        err, sample_skew = _simulate_skewed_error(local_rng, n_points, spec.err_std, spec.skew_bounds)

        resistance = spec.r20 * (1.0 + ALPHA_CU * (temp - T_REF_C)) + err
        resistance = np.round(resistance, 5)

        out[f"T_{i}"] = temp.astype(float)
        out[f"R_{i}"] = resistance.astype(float)
        out[f"R20_{i}"] = float(spec.r20)
        out[f"residual_skew_{i}"] = float(sample_skew)

    return out


def compute_physics_residuals(temp, resistance, r20):
    expected = r20 * (1.0 + ALPHA_CU * (temp - T_REF_C))
    return resistance - expected


def _fit_stats(residuals):
    residuals = np.asarray(residuals, dtype=float)
    n = len(residuals)

    s = float(skew(residuals, bias=False))
    sk_stat, sk_p = skewtest(residuals)
    norm_stat, norm_p = normaltest(residuals)
    shp_stat, shp_p = shapiro(residuals)

    sn_params = skewnorm.fit(residuals)
    sn_ll = float(np.sum(skewnorm.logpdf(residuals, *sn_params)))
    sn_aic = float(2 * 3 - 2 * sn_ll)
    sn_ks_stat, sn_ks_p = kstest(residuals, "skewnorm", args=sn_params)

    gm_ok = True
    try:
        gm_params = gamma.fit(residuals)
        gm_ll = float(np.sum(gamma.logpdf(residuals, *gm_params)))
        gm_aic = float(2 * 3 - 2 * gm_ll)
        gm_ks_stat, gm_ks_p = kstest(residuals, "gamma", args=gm_params)
    except Exception:
        gm_ok = False
        gm_aic = np.nan
        gm_ks_stat = np.nan
        gm_ks_p = np.nan

    return {
        "n": int(n),
        "skewness": s,
        "skewtest_p": float(sk_p),
        "normaltest_p": float(norm_p),
        "shapiro_p": float(shp_p),
        "sn_aic": sn_aic,
        "sn_ks_p": float(sn_ks_p),
        "sn_ks_stat": float(sn_ks_stat),
        "gamma_aic": float(gm_aic),
        "gamma_ks_p": float(gm_ks_p),
        "gamma_ks_stat": float(gm_ks_stat),
        "gamma_fit_ok": bool(gm_ok),
    }


def run_goodness_report(dataset):
    report = {}
    for i in range(1, 5):
        t = dataset[f"T_{i}"]
        r = dataset[f"R_{i}"]
        r20 = dataset[f"R20_{i}"]
        resid = compute_physics_residuals(t, r, r20)
        stats = _fit_stats(resid)
        stats["temp_mean"] = float(np.mean(t))
        stats["temp_min"] = float(np.min(t))
        stats["temp_max"] = float(np.max(t))
        stats["res_mean"] = float(np.mean(r))
        stats["res_min"] = float(np.min(r))
        stats["res_max"] = float(np.max(r))
        report[f"TR{i}"] = stats
    return report


CSV_REQUIRED_COLUMNS = (
    "day",
    "T1", "R1",
    "T2", "R2",
    "T3", "R3",
    "T4", "R4",
)


def _float_from_row(row, key, row_idx):
    try:
        return float(row[key])
    except Exception as exc:
        raise ValueError(f"Invalid numeric value in CSV at row {row_idx}, column '{key}'") from exc


def load_case_study_dataset_from_csv(csv_path=CSV_DATA_PATH):
    """
    Load the fixed case-study dataset from CSV and return the project dataset dict.
    This is the single source of truth used by all scripts.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Case-study CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [c for c in CSV_REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")
        rows = list(reader)
        has_avg_cols = ("T_avg" in reader.fieldnames) and ("R_avg" in reader.fieldnames)

    if len(rows) != N_POINTS:
        raise ValueError(f"CSV must contain exactly {N_POINTS} rows, found {len(rows)}")

    t_values = {f"T_{i}": [] for i in range(1, 5)}
    r_values = {f"R_{i}": [] for i in range(1, 5)}

    for idx, row in enumerate(rows, start=1):
        day_raw = _float_from_row(row, "day", idx)
        day = int(round(day_raw))
        if day != idx:
            raise ValueError(f"CSV day sequence mismatch at row {idx}: expected {idx}, got {row['day']}")

        t1 = _float_from_row(row, "T1", idx)
        r1 = _float_from_row(row, "R1", idx)
        t2 = _float_from_row(row, "T2", idx)
        r2 = _float_from_row(row, "R2", idx)
        t3 = _float_from_row(row, "T3", idx)
        r3 = _float_from_row(row, "R3", idx)
        t4 = _float_from_row(row, "T4", idx)
        r4 = _float_from_row(row, "R4", idx)

        if has_avg_cols:
            csv_t_avg = _float_from_row(row, "T_avg", idx)
            csv_r_avg = _float_from_row(row, "R_avg", idx)
            calc_t_avg = 0.25 * (t1 + t2 + t3 + t4)
            calc_r_avg = 0.25 * (r1 + r2 + r3 + r4)
            if not np.isclose(csv_t_avg, calc_t_avg, atol=1e-3):
                raise ValueError(
                    f"CSV integrity check failed at row {idx} for T_avg: "
                    f"csv={csv_t_avg}, calc={calc_t_avg}"
                )
            if not np.isclose(csv_r_avg, calc_r_avg, atol=1e-5):
                raise ValueError(
                    f"CSV integrity check failed at row {idx} for R_avg: "
                    f"csv={csv_r_avg}, calc={calc_r_avg}"
                )

        t_values["T_1"].append(t1)
        r_values["R_1"].append(r1)
        t_values["T_2"].append(t2)
        r_values["R_2"].append(r2)
        t_values["T_3"].append(t3)
        r_values["R_3"].append(r3)
        t_values["T_4"].append(t4)
        r_values["R_4"].append(r4)

    out = {
        "meta": {
            "n_points": len(rows),
            "step_hours": STEP_HOURS,
            "alpha_cu": ALPHA_CU,
            "t_ref_c": T_REF_C,
            "data_source": "csv_locked",
            "csv_file": str(csv_path.name),
        }
    }

    for i, spec in enumerate(SPECS, start=1):
        t_arr = np.asarray(t_values[f"T_{i}"], dtype=float)
        r_arr = np.asarray(r_values[f"R_{i}"], dtype=float)
        out[f"T_{i}"] = t_arr
        out[f"R_{i}"] = r_arr
        out[f"R20_{i}"] = float(spec.r20)
        resid = compute_physics_residuals(t_arr, r_arr, out[f"R20_{i}"])
        out[f"residual_skew_{i}"] = float(skew(resid, bias=False))

    return out


# --------------------------------------------------------------------------
# Fixed manual dataset (realistic 20kV/400V case study), 150 points each.
# These are the exact arrays used in baducl.py and ucl30day.py.
# --------------------------------------------------------------------------
T_1 = np.array([52.453,51.131,52.742,51.163,49.351,49.064,51.908,49.179,51.741,48.59 ,49.802,47.025,47.767,47.502,45.718,44.999,46.669,46.424,46.462,44.042,46.268,44.53 ,44.501,44.214,47.218,44.839,49.566,46.359,47.786,46.548,48.974,46.669,47.498,47.784,49.131,47.916,49.142,48.698,49.919,49.75 ,53.144,53.638,56.216,51.565,51.315,47.05 ,48.167,46.151,47.694,46.859,50.092,50.187,51.994,51.171,53.252,48.973,49.217,52.133,52.461,48.992,48.778,47.304,48.374,47.339,47.962,43.673,48.014,44.433,47.168,45.473,49.394,48.102,47.879,44.929,48.539,46.772,46.785,44.738,46.616,45.071,44.064,44.42 ,47.863,43.41 ,48.207,50.388,49.316,46.64 ,47.743,44.651,45.263,44.915,47.436,45.551,47.15 ,48.117,51.096,50.206,52.866,51.045,54.787,50.31 ,55.857,52.221,52.288,50.197,54.483,51.83 ,52.709,51.442,52.78 ,52.049,53.994,52.849,53.618,51.244,50.047,47.903,48.65 ,45.035,50.466,47.339,49.088,44.193,47.12 ,45.124,49.266,45.039,46.536,46.036,49.394,47.172,52.589,50.524,51.456,47.722,49.915,49.062,48.445,47.222,48.53 ,46.224,47.264,48.29 ,48.669,48.2  ,48.152,44.521,47.643,47.85 ])
R_1 = np.array([0.0989 ,0.09853,0.09895,0.09858,0.09788,0.09778,0.0988 ,0.09807,0.09879,0.0978 ,0.09813,0.09697,0.09732,0.09702,0.09655,0.09636,0.09732,0.09677,0.09699,0.09607,0.09667,0.09633,0.09626,0.09618,0.09718,0.09619,0.09823,0.09699,0.09772,0.0967 ,0.098  ,0.09686,0.09757,0.09726,0.09758,0.09734,0.09768,0.09749,0.09817,0.09806,0.09896,0.0993 ,0.10011,0.09849,0.09855,0.097  ,0.09743,0.09684,0.09736,0.09714,0.0982 ,0.09811,0.09947,0.09863,0.099  ,0.09851,0.09814,0.0988 ,0.09865,0.09802,0.09791,0.09717,0.09767,0.09707,0.09732,0.09586,0.09771,0.09688,0.09736,0.09665,0.09791,0.09727,0.09762,0.09622,0.09764,0.09694,0.09715,0.09654,0.09715,0.09651,0.09595,0.09657,0.09721,0.09583,0.0977 ,0.09825,0.09768,0.09677,0.09729,0.09653,0.09617,0.0963 ,0.0975 ,0.09693,0.09719,0.09737,0.09832,0.09821,0.09913,0.09836,0.09998,0.09818,0.09996,0.09913,0.09876,0.09826,0.09955,0.09878,0.09891,0.09865,0.09919,0.09887,0.09958,0.09951,0.09928,0.09873,0.09796,0.09759,0.09816,0.09628,0.0981 ,0.0978 ,0.09807,0.09603,0.09695,0.09654,0.09828,0.0967 ,0.0967 ,0.09674,0.09776,0.09724,0.09915,0.09844,0.09874,0.09727,0.09788,0.09809,0.09747,0.09736,0.09779,0.09669,0.09716,0.09734,0.09764,0.09742,0.09737,0.0962 ,0.09707,0.09725])
T_2 = np.array([51.988,48.093,53.31 ,51.571,54.805,53.049,56.366,55.025,56.971,52.899,55.429,51.689,53.56 ,51.614,54.276,51.705,54.047,49.858,50.797,49.286,51.977,52.09 ,54.278,49.117,51.502,45.31 ,50.158,46.849,45.913,44.432,46.862,44.853,49.393,50.459,53.707,54.155,50.093,51.612,52.517,50.271,49.954,48.598,48.724,45.426,48.794,44.98 ,49.406,51.696,54.114,50.087,51.281,48.897,49.185,48.716,46.628,44.233,47.398,51.245,56.178,52.827,56.025,55.216,57.429,56.278,56.634,54.115,54.8  ,52.31 ,54.556,54.867,56.081,53.853,53.133,53.379,56.674,51.693,54.296,50.43 ,54.961,54.429,55.49 ,55.301,58.084,54.417,57.079,54.72 ,57.194,56.512,57.673,55.953,58.477,57.664,58.528,53.   ,52.735,49.436,53.305,46.252,49.58 ,48.704,49.52 ,47.648,48.738,47.075,49.682,48.614,48.176,46.844,50.855,49.125,53.222,50.753,56.172,51.726,51.942,51.925,50.984,50.787,54.678,51.605,53.358,51.907,57.485,55.246,58.548,55.169,57.366,55.27 ,57.445,54.99 ,56.352,55.119,58.695,55.08 ,56.451,54.804,56.752,54.338,55.498,49.226,49.426,48.074,49.208,50.155,51.401,52.744,54.546,55.562,54.163,52.73 ])
R_2 = np.array([0.10105,0.10036,0.10143,0.10089,0.10217,0.10157,0.10287,0.10217,0.10279,0.10173,0.10232,0.10136,0.10212,0.10125,0.10212,0.10114,0.10187,0.10033,0.10063,0.10008,0.10097,0.10109,0.1017 ,0.10062,0.10115,0.09894,0.10049,0.09916,0.09938,0.09831,0.0992 ,0.09885,0.10043,0.10073,0.1017 ,0.10197,0.10043,0.10073,0.10185,0.10065,0.10042,0.09987,0.10007,0.09923,0.0999 ,0.09874,0.1002 ,0.10112,0.10214,0.10068,0.101  ,0.10046,0.10007,0.09974,0.09923,0.09837,0.09935,0.10123,0.10293,0.10152,0.10292,0.10284,0.10304,0.10306,0.10262,0.10195,0.10197,0.10127,0.10208,0.10238,0.10256,0.10167,0.10194,0.10137,0.10273,0.10112,0.10196,0.10063,0.10219,0.10196,0.10215,0.10218,0.1037 ,0.10209,0.10291,0.10221,0.10309,0.10272,0.10361,0.10286,0.10364,0.10323,0.10326,0.10159,0.10136,0.10094,0.10163,0.09893,0.10021,0.09992,0.10005,0.09964,0.09993,0.09928,0.10058,0.09994,0.0996 ,0.09988,0.10092,0.0999 ,0.10137,0.10057,0.10256,0.10133,0.1013 ,0.10102,0.10077,0.1011 ,0.10203,0.10095,0.10152,0.101  ,0.1031 ,0.10229,0.10325,0.10244,0.10298,0.10239,0.10305,0.10215,0.10275,0.10244,0.10336,0.10239,0.10256,0.10197,0.10282,0.10176,0.10334,0.10028,0.10042,0.09957,0.10003,0.1005 ,0.1009 ,0.10128,0.1025 ,0.10236,0.10229,0.10179])
T_3 = np.array([50.397,48.259,50.939,45.59 ,46.255,46.716,52.352,47.472,49.745,48.499,49.397,48.913,51.543,46.547,48.809,46.568,49.164,48.037,50.204,46.853,46.362,46.932,46.56 ,43.961,46.212,41.982,45.45 ,44.432,46.018,44.591,48.086,44.631,49.362,47.145,47.694,47.156,47.658,44.129,47.855,47.038,48.243,48.273,48.355,44.239,47.825,46.063,49.4  ,48.964,50.72 ,48.833,47.122,48.055,50.718,47.96 ,49.16 ,49.13 ,53.131,49.441,50.987,47.38 ,51.609,52.551,54.886,53.093,53.889,51.827,54.127,50.642,54.122,51.879,55.573,54.41 ,56.239,55.339,56.476,54.693,52.246,49.941,49.663,48.966,49.132,46.873,53.371,50.353,55.243,51.862,53.913,51.58 ,52.01 ,49.857,51.591,48.62 ,50.363,46.607,48.547,49.499,50.325,47.507,52.702,49.681,50.869,50.196,54.145,50.226,51.958,50.324,50.902,45.728,50.29 ,49.201,53.026,51.208,53.082,51.098,53.348,49.693,52.23 ,51.689,49.676,48.583,51.508,48.118,50.613,50.676,52.365,51.246,53.16 ,52.563,54.818,53.414,54.26 ,50.847,49.312,47.993,47.931,44.804,44.809,44.123,45.355,42.331,44.088,46.664,49.452,48.062,50.579,47.57 ,50.369,47.829,48.502,48.114])
R_3 = np.array([0.09601,0.0955 ,0.09632,0.09486,0.0948 ,0.09486,0.09681,0.09496,0.09582,0.09568,0.09628,0.09551,0.09655,0.09489,0.09551,0.09474,0.09554,0.09601,0.09624,0.09485,0.09476,0.09511,0.09474,0.09417,0.09479,0.0932 ,0.0944 ,0.09433,0.09456,0.09414,0.09549,0.09426,0.09582,0.09525,0.09515,0.09504,0.09553,0.09396,0.09525,0.09483,0.09521,0.09532,0.09565,0.09401,0.09521,0.09467,0.09628,0.09546,0.09633,0.09567,0.09533,0.09535,0.09622,0.09513,0.09582,0.0956 ,0.09706,0.09603,0.09611,0.09495,0.09657,0.09668,0.09773,0.09685,0.09715,0.09651,0.0973 ,0.09616,0.09748,0.09661,0.09782,0.09742,0.09865,0.09761,0.09811,0.0974 ,0.09654,0.0959 ,0.09605,0.09543,0.0956 ,0.09484,0.09702,0.09622,0.09797,0.09665,0.09709,0.09708,0.09668,0.09599,0.09667,0.09538,0.09606,0.09468,0.09558,0.09591,0.09588,0.0952 ,0.09724,0.09604,0.0964 ,0.09599,0.09748,0.0959 ,0.09703,0.09602,0.09683,0.09447,0.09605,0.09558,0.09691,0.09632,0.09692,0.09679,0.09733,0.09595,0.09687,0.09649,0.0961 ,0.09551,0.09678,0.09535,0.09617,0.09618,0.09728,0.09699,0.09703,0.09735,0.09792,0.09726,0.09748,0.09612,0.09601,0.09531,0.09531,0.09426,0.09439,0.09403,0.09419,0.09342,0.09421,0.09474,0.09571,0.09532,0.09627,0.09516,0.09593,0.09516,0.09547,0.09519])
T_4 = np.array([53.939,54.339,56.308,51.563,50.993,49.422,52.328,48.292,50.378,49.53 ,52.869,54.052,57.662,55.53 ,55.825,52.281,55.029,52.818,53.105,48.102,49.369,47.446,48.468,46.13 ,50.995,46.62 ,52.064,49.593,52.468,49.293,52.22 ,49.134,49.841,46.885,47.226,45.658,46.567,46.542,50.13 ,50.736,50.766,53.121,52.366,50.952,52.414,51.381,55.569,51.011,53.704,49.506,46.512,45.038,47.007,46.116,49.124,47.924,49.753,50.277,51.7  ,47.47 ,48.298,47.16 ,47.977,48.036,50.4  ,45.832,50.381,48.675,49.567,53.521,54.825,54.071,56.02 ,53.647,56.514,51.341,51.162,51.381,52.414,48.539,53.07 ,51.544,51.381,53.325,56.418,53.928,55.18 ,51.159,49.494,49.827,48.292,45.673,53.162,55.314,55.755,52.909,54.68 ,55.181,58.526,55.426,57.458,55.012,60.2  ,56.357,58.69 ,56.188,57.023,52.72 ,56.208,55.528,56.98 ,53.957,53.175,54.559,52.619,51.393,52.354,52.565,56.945,52.233,56.144,54.189,53.611,51.619,55.831,52.555,51.866,50.838,56.515,52.26 ,54.177,49.816,50.382,49.07 ,53.129,53.661,55.462,53.974,53.502,53.921,57.755,57.263,59.88 ,57.481,60.932,54.466,51.571,46.343,51.656,51.511])
R_4 = np.array([0.10309,0.10321,0.10383,0.10307,0.10208,0.10149,0.10272,0.10099,0.10228,0.10148,0.1029 ,0.10312,0.10449,0.10381,0.10385,0.10271,0.10365,0.10264,0.10269,0.10112,0.1016 ,0.10096,0.10116,0.10026,0.10208,0.10064,0.10262,0.10196,0.1037 ,0.10219,0.10274,0.10147,0.10176,0.10074,0.10098,0.10008,0.10055,0.10035,0.10166,0.10192,0.10201,0.10256,0.10266,0.10188,0.10269,0.1022 ,0.10369,0.10257,0.10314,0.10178,0.10055,0.09981,0.10069,0.10079,0.10122,0.10093,0.10169,0.10188,0.10232,0.10086,0.10117,0.10056,0.10108,0.10087,0.10173,0.10015,0.102  ,0.10146,0.10159,0.10331,0.10365,0.10355,0.10403,0.10336,0.10412,0.10269,0.10265,0.10273,0.1027 ,0.10108,0.10265,0.10244,0.10239,0.10284,0.10427,0.10333,0.10395,0.10208,0.10193,0.10167,0.10121,0.1001 ,0.10273,0.10373,0.10404,0.10263,0.10361,0.10349,0.10481,0.10418,0.10452,0.10359,0.1053 ,0.10385,0.1049 ,0.10429,0.1044 ,0.10268,0.1042 ,0.10381,0.10413,0.10318,0.10286,0.10331,0.1026 ,0.10212,0.1031 ,0.10255,0.10475,0.10301,0.10428,0.10361,0.10274,0.10265,0.1041 ,0.10345,0.1026 ,0.10209,0.10435,0.10245,0.10321,0.10179,0.10228,0.1018 ,0.10292,0.10302,0.10366,0.10365,0.10335,0.10306,0.10491,0.10449,0.1053 ,0.10411,0.10554,0.10341,0.1026 ,0.10056,0.1027 ,0.10221])

R20_1 = 0.0878
R20_2 = 0.0899
R20_3 = 0.0859
R20_4 = 0.0911

MANUAL_DATASET = {
    "meta": {"n_points": N_POINTS, "step_hours": STEP_HOURS, "alpha_cu": ALPHA_CU, "t_ref_c": T_REF_C},
    "T_1": T_1, "R_1": R_1, "R20_1": R20_1,
    "T_2": T_2, "R_2": R_2, "R20_2": R20_2,
    "T_3": T_3, "R_3": R_3, "R20_3": R20_3,
    "T_4": T_4, "R_4": R_4, "R20_4": R20_4,
}

def generate_stationary_case_dataset(n_points=N_POINTS, seed=20260612):
    """
    A more stationary realistic dataset:
    - still physics-based with R = R20 * (1 + alpha*(T-20)) + e
    - same skewness targets per transformer (about 1.0-1.2)
    - includes realistic short-term load jitter and occasional thermal pulses
      so T2 does not collapse into long flat clusters
    - default seed tuned for a more article-friendly T2 shape
      while keeping both charts in-control with fixed UCLs.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points, dtype=float)

    # Common ambient with daily + biweekly components and realistic weather jitter.
    common_ambient = (
        27.6
        + 1.10 * np.sin(np.pi * t + 0.4)
        + 0.70 * np.sin(2.0 * np.pi * t / 14.0 + 0.2)
        + 0.30 * np.sin(2.0 * np.pi * t / 30.0 + 0.9)
        + rng.normal(0.0, 0.18, size=n_points)
    )

    out = {
        "meta": {
            "n_points": n_points,
            "step_hours": STEP_HOURS,
            "alpha_cu": ALPHA_CU,
            "t_ref_c": T_REF_C,
            "dataset_type": "stationary_realistic",
            "seed": int(seed),
        }
    }

    # Slightly higher measurement/noise scale to avoid unrealistically tiny sigma_ref.
    err_scale = [1.12, 1.10, 1.12, 1.10]

    for i, spec in enumerate(SPECS, start=1):
        local_rng = np.random.default_rng(seed + 101 * i)
        load = _ar1_process(local_rng, n_points, phi=0.18, innovation_std=0.34)
        fast_jitter = local_rng.normal(0.0, 0.35, size=n_points)
        pulse_flag = (local_rng.random(n_points) < 0.08).astype(float)
        pulse_mag = local_rng.uniform(0.7, 1.6, size=n_points) * pulse_flag

        rise = (
            8.0
            + 14.0 * (load - load.min()) / (load.max() - load.min() + 1e-9)
            + 1.5  * np.sin(2.0 * np.pi * t / 7.0 + 0.45 * i)
            + 0.55 * fast_jitter
            + pulse_mag
        )

        temp = common_ambient + 0.30 * spec.ambient_offset + rise + local_rng.normal(0.0, 0.23, size=n_points)
        temp = np.clip(temp, 30.0, 88.0)

        e_std = spec.err_std * err_scale[i - 1]
        err, sample_skew = _simulate_skewed_error(local_rng, n_points, e_std, spec.skew_bounds)
        resistance = spec.r20 * (1.0 + ALPHA_CU * (temp - T_REF_C)) + err

        out[f"T_{i}"] = np.round(temp, 3).astype(float)
        out[f"R_{i}"] = np.round(resistance, 5).astype(float)
        out[f"R20_{i}"] = float(spec.r20)
        out[f"residual_skew_{i}"] = float(sample_skew)

    return out


# Active dataset used by all scripts (baducl/ucl30day/comparennt2/goodnes report).
# Single source of truth is the CSV file in project root.
DATASET = load_case_study_dataset_from_csv()

T_1 = DATASET["T_1"]
R_1 = DATASET["R_1"]
T_2 = DATASET["T_2"]
R_2 = DATASET["R_2"]
T_3 = DATASET["T_3"]
R_3 = DATASET["R_3"]
T_4 = DATASET["T_4"]
R_4 = DATASET["R_4"]

R20_1 = DATASET["R20_1"]
R20_2 = DATASET["R20_2"]
R20_3 = DATASET["R20_3"]
R20_4 = DATASET["R20_4"]

T_avg = np.mean([T_1, T_2, T_3, T_4], axis=0)
R_avg = np.mean([R_1, R_2, R_3, R_4], axis=0)


def save_report(report, path="goodnes_report.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _print_report(report):
    print("=" * 92)
    print(f"Realistic 20kV/400V transformer-output dataset (CSV-locked source: {CSV_DATA_PATH.name})")
    print("Model: R = R20 * (1 + alpha * (T - 20)) + e, alpha=0.00393")
    print("Sampling: every 12h, total points per transformer: 150")
    print("=" * 92)
    for k, v in report.items():
        print(f"\n{k}")
        print(f"  Temp C   : mean={v['temp_mean']:.2f}, min={v['temp_min']:.2f}, max={v['temp_max']:.2f}")
        print(f"  Resist o : mean={v['res_mean']:.5f}, min={v['res_min']:.5f}, max={v['res_max']:.5f}")
        print(f"  Residual skewness = {v['skewness']:.4f}")
        print(
            "  P-values (tests): "
            f"skewtest={v['skewtest_p']:.4g}, normaltest={v['normaltest_p']:.4g}, shapiro={v['shapiro_p']:.4g}"
        )
        print(
            "  GOF (lower AIC better): "
            f"AIC_skewnorm={v['sn_aic']:.2f}, AIC_gamma={v['gamma_aic']:.2f}, "
            f"KS_p_skewnorm={v['sn_ks_p']:.4g}, KS_p_gamma={v['gamma_ks_p']:.4g}"
        )

    skews = [report[f"TR{i}"]["skewness"] for i in range(1, 5)]
    print("\nResidual skewness check (required 1.0 to 1.2):")
    for i, s in enumerate(skews, start=1):
        ok = 1.0 <= s <= 1.2
        print(f"  TR{i}: {s:.4f} -> {'OK' if ok else 'OUT'}")

    avg_skew = float(np.mean(skews))
    print(f"  Average skewness across 4 transformers: {avg_skew:.4f}")
    print("=" * 92)


if __name__ == "__main__":
    report = run_goodness_report(DATASET)
    _print_report(report)
    save_report(report, path="goodnes_report.json")
    print("Saved: goodnes_report.json")
