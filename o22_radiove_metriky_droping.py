import pandas as pd
import numpy as np
import csv
import joblib

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# =========================
# 1) nastavenia
# =========================
CSV_PATH = "kompletcsv_final_filled.csv"
TIMESTAMP_COL = "timestamp"
TARGET = "statistics_input.p1203_output_report.O22"

RADIO_COLS = [
    "connection_detail_trace.rsrp",
    "connection_detail_trace.rsrq",
    "connection_detail_trace.rssnr",
]

VIDEO_COL = "client_report.video.resolution"

SEGMENT_LENGTH = 3
STRIDE = 1
HORIZON = 1

TEST_SIZE = 0.2
RANDOM_STATE = 45

USE_FORWARD_FILL = False
FFILL_LIMIT = 5

RADIO_BOUNDS = {
    "connection_detail_trace.rsrp": (-160, -30),
    "connection_detail_trace.rsrq": (-40, 20),
    "connection_detail_trace.rssnr": (-50, 50),
}

# =========================
# 2) pomocne funkcie
# =========================
def to_num(x):
    return pd.to_numeric(str(x).replace(",", ".").strip(), errors="coerce")

def series_stats(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()

    if len(s) == 0:
        return None

    return {
        "mean": float(s.mean()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }

def drop_all_nan_train_columns(Xtr: pd.DataFrame, Xte: pd.DataFrame):
    # vyhod stlpce ktore su v train uplne prazdne
    cols_all_nan = [c for c in Xtr.columns if Xtr[c].isna().all()]
    if cols_all_nan:
        print("Dropping columns with no observed values in TRAIN:", cols_all_nan)
        Xtr = Xtr.drop(columns=cols_all_nan)
        Xte = Xte.drop(columns=cols_all_nan, errors="ignore")
    return Xtr, Xte

# =========================
# 3) nacitanie csv
# =========================
with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f, delimiter=";")
    rows = list(reader)

max_cols = max(len(r) for r in rows)
print(f"Max stlpcov v CSV: {max_cols}")

# zarovname riadky aby mali rovnaky pocet stlpcov
clean_rows = []
for r in rows:
    if len(r) < max_cols:
        r.extend([None] * (max_cols - len(r)))
    elif len(r) > max_cols:
        r = r[:max_cols - 1] + [",".join(r[max_cols - 1:])]
    clean_rows.append(r)

df = pd.DataFrame(clean_rows[1:], columns=clean_rows[0])

# =========================
# 4) odstran riadky bez video resolution
# =========================
if VIDEO_COL not in df.columns:
    raise KeyError(f"Chyba stlpec: {VIDEO_COL}")

before = len(df)

df[VIDEO_COL] = df[VIDEO_COL].astype(str).str.strip()

df = df[
    df[VIDEO_COL].notna() &
    (df[VIDEO_COL] != "") &
    (df[VIDEO_COL].str.lower() != "none") &
    (df[VIDEO_COL].str.lower() != "nan")
].copy()

print(f"Dropped rows without video resolution: {before - len(df)}")

# =========================
# 5) konverzia targetu a radio hodnot
# =========================
if TARGET not in df.columns:
    raise KeyError(f"Chyba stlpec TARGET: {TARGET}")

df[TARGET] = df[TARGET].apply(to_num)

for c in RADIO_COLS:
    if c not in df.columns:
        raise KeyError(f"Chyba radiovy stlpec: {c}")
    df[c] = df[c].apply(to_num)

# odstran nezmyselne hodnoty (sentinel a mimo rozsah)
for c in RADIO_COLS:
    df[c] = df[c].replace([2147483647, -2147483648], np.nan)
    low, high = RADIO_BOUNDS[c]
    df.loc[(df[c] < low) | (df[c] > high), c] = np.nan

before = len(df)
df = df.dropna(subset=[TARGET])
print(f"Dropped rows with NaN TARGET: {before - len(df)}")

print("\nRadio diagnostics after cleaning")
for c in RADIO_COLS:
    s = df[c].dropna()
    print(f"{c}: count={len(s)}, min={s.min() if len(s) else np.nan}, max={s.max() if len(s) else np.nan}")

# =========================
# 6) cas, zoradenie, pripadne forward fill
# =========================
df["_orig_order"] = np.arange(len(df))

df[TIMESTAMP_COL] = pd.to_datetime(
    df[TIMESTAMP_COL]
      .astype(str)
      .str.replace(r"\s+", " ", regex=True)
      .str.strip(),
    dayfirst=True,
    errors="coerce"
)

n_nat = df[TIMESTAMP_COL].isna().sum()
print(f"NaT timestamps after parsing: {n_nat}")

# zoradime podla casu
df = (
    df.dropna(subset=[TIMESTAMP_COL])
      .sort_values([TIMESTAMP_COL, "_orig_order"])
      .reset_index(drop=True)
      .drop(columns=["_orig_order"])
)

# doplnenie hodnot dopredu (ak zapnute)
if USE_FORWARD_FILL:
    df[RADIO_COLS] = df[RADIO_COLS].ffill(limit=FFILL_LIMIT)

if len(df) < SEGMENT_LENGTH + HORIZON:
    raise ValueError(
        f"Prilis malo riadkov po cisteni: {len(df)}. "
        f"Potrebne aspon {SEGMENT_LENGTH + HORIZON}."
    )

# =========================
# 7) segmentacia (okna)
# =========================
segments = []
segment_rows = []

for start in range(0, len(df) - SEGMENT_LENGTH - HORIZON + 1, STRIDE):
    end = start + SEGMENT_LENGTH - 1
    seg = df.iloc[start:end + 1]

    if HORIZON == 0:
        t = pd.to_numeric(seg[TARGET], errors="coerce").dropna()
        if len(t) == 0:
            continue

        target_val = float(t.iloc[-1])
        target_row = end

    else:
        target_row = end + HORIZON
        if target_row >= len(df):
            break

        tv = pd.to_numeric(df.iloc[target_row][TARGET], errors="coerce")
        if pd.isna(tv):
            continue

        target_val = float(tv)

    s_rsrp = series_stats(seg["connection_detail_trace.rsrp"])
    s_rsrq = series_stats(seg["connection_detail_trace.rsrq"])
    s_rssnr = series_stats(seg["connection_detail_trace.rssnr"])

    # ak v okne nie je ziadna radio hodnota, zahodime segment
    if s_rsrp is None and s_rsrq is None and s_rssnr is None:
        continue

    feat = {}

    def add_stats(prefix, st):
        keys = ["mean", "p25", "p75", "min", "max"]
        if st is None:
            for k in keys:
                feat[f"{prefix}_{k}"] = np.nan
        else:
            for k in keys:
                feat[f"{prefix}_{k}"] = st[k]

    add_stats("rsrp", s_rsrp)
    add_stats("rsrq", s_rsrq)
    add_stats("rssnr", s_rssnr)

    feat["target"] = target_val
    segments.append(feat)

    segment_rows.append({
        "start_row": start,
        "end_row": end,
        "target_row": target_row
    })

df_segments = pd.DataFrame(segments)
df_seg_rows = pd.DataFrame(segment_rows)

print(f"Total segments: {len(df_segments)}")

if len(df_segments) == 0:
    raise ValueError("Po segmentacii nevznikol ziaden segment.")

# vyhod stlpce ktore su uplne prazdne
all_nan_cols = [c for c in df_segments.columns if c != "target" and df_segments[c].isna().all()]
if all_nan_cols:
    print("Dropping all-NaN feature columns:", all_nan_cols)
    df_segments = df_segments.drop(columns=all_nan_cols)

# =========================
# 8) split s casovym odstupom (embargo)
# =========================
GAP = (SEGMENT_LENGTH - 1) + max(0, HORIZON)
split_row = int(len(df) * (1 - TEST_SIZE))

train_mask = df_seg_rows["end_row"] < (split_row - (SEGMENT_LENGTH - 1))
test_mask = df_seg_rows["start_row"] >= (split_row + GAP)

X = df_segments.drop(columns=["target"])
y = df_segments["target"]

expected_cols = [
    "rsrp_mean", "rsrp_p25", "rsrp_p75", "rsrp_min", "rsrp_max",
    "rsrq_mean", "rsrq_p25", "rsrq_p75", "rsrq_min", "rsrq_max",
    "rssnr_mean", "rssnr_p25", "rssnr_p75", "rssnr_min", "rssnr_max",
]

present_cols = [c for c in expected_cols if c in X.columns]
missing_cols = [c for c in expected_cols if c not in X.columns]

print("Feature cols present in X:", present_cols)
if missing_cols:
    print("Feature cols missing from X:", missing_cols)

X_train = X[train_mask].reset_index(drop=True)
y_train = y[train_mask].reset_index(drop=True)
X_test = X[test_mask].reset_index(drop=True)
y_test = y[test_mask].reset_index(drop=True)

if len(X_train) == 0:
    raise ValueError("Train set je prazdny po aplikovani embarga.")
if len(X_test) == 0:
    raise ValueError("Test set je prazdny po aplikovani embarga.")

print(f"Train segments: {len(X_train)}, Test segments: {len(X_test)} (embargo GAP={GAP})")

# =========================
# 9) bez imputacie - zahod riadky s NaN
# =========================
X_train, X_test = drop_all_nan_train_columns(X_train, X_test)
FEATURES = list(X_train.columns)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

print("\nFEATURE diagnostics before dropping NaN rows")
print("-------------------------------------------")
print("Any NaN in X_train?", X_train.isna().any().any())
print("Any NaN in X_test?", X_test.isna().any().any())
print("Any inf in X_train?", np.isinf(X_train.values).any())
print("Any inf in X_test?", np.isinf(X_test.values).any())

# nechame len riadky bez NaN
train_keep_mask = ~X_train.isna().any(axis=1)
test_keep_mask = ~X_test.isna().any(axis=1)

X_train_clean = X_train.loc[train_keep_mask].reset_index(drop=True)
y_train_clean = y_train.loc[train_keep_mask].reset_index(drop=True)

X_test_clean = X_test.loc[test_keep_mask].reset_index(drop=True)
y_test_clean = y_test.loc[test_keep_mask].reset_index(drop=True)

print("\nComplete-case filtering")
print("-----------------------")
print(f"Train rows kept: {len(X_train_clean)} / {len(X_train)}")
print(f"Test rows kept:  {len(X_test_clean)} / {len(X_test)}")

if len(X_train_clean) == 0:
    raise ValueError("Po odstraneni riadkov s NaN je train prazdny.")
if len(X_test_clean) == 0:
    raise ValueError("Po odstraneni riadkov s NaN je test prazdny.")

# standardizacia
scaler = StandardScaler()
X_train_s = pd.DataFrame(
    scaler.fit_transform(X_train_clean),
    columns=FEATURES,
    index=X_train_clean.index
)
X_test_s = pd.DataFrame(
    scaler.transform(X_test_clean),
    columns=FEATURES,
    index=X_test_clean.index
)

# =========================
# 10) jednoduchy baseline (priemer)
# =========================
y_pred_baseline = np.full(len(y_test_clean), fill_value=float(y_train_clean.mean()))
baseline_rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_baseline))
baseline_r2 = r2_score(y_test_clean, y_pred_baseline)

print("\nBASELINE (train mean)")
print("---------------------")
print(f"RMSE: {baseline_rmse:.3f}, R2: {baseline_r2:.3f}")

# =========================
# 11) modely
# =========================