import pandas as pd
import numpy as np
import csv
import joblib

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# =========================
# 1) KONFIGURÁCIA
# =========================
CSV_PATH = "kompletcsv_final.csv"
TIMESTAMP_COL = "timestamp"
TARGET = "statistics_input.p1203_output_report.O22"

RADIO_COLS = [
    "connection_detail_trace.rsrp",
    "connection_detail_trace.rsrq",
    "connection_detail_trace.rssnr",
]

SEGMENT_LENGTH = 12
STRIDE = 1
HORIZON = 1

TEST_SIZE = 0.2
RANDOM_STATE = 45

USE_FORWARD_FILL = False
FFILL_LIMIT = 5

# =========================
# 2) NAČÍTANIE CSV
# =========================
with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f, delimiter=";")
    rows = list(reader)

max_cols = max(len(r) for r in rows)
print(f"Max stĺpcov v CSV: {max_cols}")

clean_rows = []
for r in rows:
    if len(r) < max_cols:
        r.extend([None] * (max_cols - len(r)))
    elif len(r) > max_cols:
        r = r[:max_cols - 1] + [",".join(r[max_cols - 1:])]
    clean_rows.append(r)

df = pd.DataFrame(clean_rows[1:], columns=clean_rows[0])

# =========================
# 2.1) DROP RIADKOV BEZ VIDEO RESOLUTION
# =========================
VIDEO_COL = "client_report.video.resolution"

if VIDEO_COL not in df.columns:
    raise KeyError(f"Chýba stĺpec: {VIDEO_COL}")

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
# 3) KONVERZIE: len TARGET + RADIO
# =========================
def to_num(x):
    return pd.to_numeric(str(x).replace(",", ".").strip(), errors="coerce")

if TARGET not in df.columns:
    raise KeyError(f"Chýba stĺpec TARGET: {TARGET}")

df[TARGET] = df[TARGET].apply(to_num)

for c in RADIO_COLS:
    if c not in df.columns:
        raise KeyError(f"Chýba rádiový stĺpec: {c}")
    df[c] = df[c].apply(to_num)

RADIO_BOUNDS = {
    "connection_detail_trace.rsrp": (-160, -30),
    "connection_detail_trace.rsrq": (-40, 20),
    "connection_detail_trace.rssnr": (-50, 50),
}

for c in RADIO_COLS:
    df[c] = df[c].replace([2147483647, -2147483648], np.nan)
    low, high = RADIO_BOUNDS[c]
    df.loc[(df[c] < low) | (df[c] > high), c] = np.nan

before = len(df)
df = df.dropna(subset=[TARGET])
print(f"Dropped rows with NaN TARGET: {before - len(df)}")

# =========================
# 4) ČAS + SORT + FORWARD-FILL
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

df = (
    df.dropna(subset=[TIMESTAMP_COL])
      .sort_values([TIMESTAMP_COL, "_orig_order"])
      .reset_index(drop=True)
      .drop(columns=["_orig_order"])
)

if USE_FORWARD_FILL:
    df[RADIO_COLS] = df[RADIO_COLS].ffill(limit=FFILL_LIMIT)

if len(df) < SEGMENT_LENGTH + HORIZON:
    raise ValueError(
        f"Príliš málo riadkov po čistení: {len(df)}. "
        f"Potrebné aspoň {SEGMENT_LENGTH + HORIZON}."
    )

# =========================
# 5) SEGMENTÁCIA
# =========================
def series_stats(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()
    n = len(s)

    if n == 0:
        return None

    out = {
        "mean": np.nan,
        "p25": np.nan,
        "p75": np.nan,
        "min": np.nan,
        "max": np.nan,
    }

    if n == 1:
        out["mean"] = float(s.mean())
        return out

    if n == 2:
        out["mean"] = float(s.mean())
        out["min"] = float(s.min())
        out["max"] = float(s.max())
        return out

    out["mean"] = float(s.mean())
    out["p25"] = float(s.quantile(0.25))
    out["p75"] = float(s.quantile(0.75))
    out["min"] = float(s.min())
    out["max"] = float(s.max())
    return out

def below_prop(s: pd.Series, thresh):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float((s < thresh).mean()) if len(s) else np.nan

segments = []
segment_rows = []

for start in range(0, len(df) - SEGMENT_LENGTH - HORIZON + 1, STRIDE):
    end = start + SEGMENT_LENGTH - 1
    seg = df.iloc[start:end + 1]

    if HORIZON == 0:
        t = pd.to_numeric(seg[TARGET], errors="coerce").dropna()
        if len(t) == 0:
            continue
        target_val = float(t.mean())
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
    s_snr  = series_stats(seg["connection_detail_trace.rssnr"])

    if s_rsrp is None and s_rsrq is None and s_snr is None:
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
    add_stats("rssnr", s_snr)

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
    raise ValueError("Po segmentácii nevznikol žiadny segment.")

all_nan_cols = [c for c in df_segments.columns if c != "target" and df_segments[c].isna().all()]
if all_nan_cols:
    print("Dropping all-NaN feature columns:", all_nan_cols)
    df_segments = df_segments.drop(columns=all_nan_cols)

# =========================
# 6) SPLIT S EMBARGOM
# =========================
GAP = (SEGMENT_LENGTH - 1) + max(0, HORIZON)
split_row = int(len(df) * (1 - TEST_SIZE))

train_mask = df_seg_rows["end_row"] < (split_row - (SEGMENT_LENGTH - 1))
test_mask  = df_seg_rows["start_row"] >= (split_row + GAP)

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
X_test  = X[test_mask].reset_index(drop=True)
y_test  = y[test_mask].reset_index(drop=True)

if len(X_train) == 0:
    raise ValueError("Train set je prázdny po aplikovaní embarga.")
if len(X_test) == 0:
    raise ValueError("Test set je prázdny po aplikovaní embarga.")

print(f"Train segments: {len(X_train)}, Test segments: {len(X_test)} (embargo GAP={GAP})")

# =========================
# 7) IMPUTÁCIA + ŠKÁLOVANIE
# =========================
def drop_all_nan_train_columns(Xtr: pd.DataFrame, Xte: pd.DataFrame):
    cols_all_nan = [c for c in Xtr.columns if Xtr[c].isna().all()]
    if cols_all_nan:
        print("Dropping columns with no observed values in TRAIN:", cols_all_nan)
        Xtr = Xtr.drop(columns=cols_all_nan)
        Xte = Xte.drop(columns=cols_all_nan, errors="ignore")
    return Xtr, Xte

X_train, X_test = drop_all_nan_train_columns(X_train, X_test)
FEATURES = list(X_train.columns)

imputer = SimpleImputer(strategy="median")
X_train_imp_arr = imputer.fit_transform(X_train)
X_test_imp_arr  = imputer.transform(X_test)

X_train_imp = pd.DataFrame(X_train_imp_arr, columns=FEATURES, index=X_train.index)
X_test_imp  = pd.DataFrame(X_test_imp_arr, columns=FEATURES, index=X_test.index)

if np.isnan(X_train_imp.values).any() or np.isnan(X_test_imp.values).any():
    nan_cols_train = list(X_train_imp.columns[X_train_imp.isna().any()])
    nan_cols_test  = list(X_test_imp.columns[X_test_imp.isna().any()])
    print("WARNING: NaN after imputation (train/test):", nan_cols_train, nan_cols_test)
    X_train_imp = X_train_imp.fillna(0.0)
    X_test_imp  = X_test_imp.fillna(0.0)

X_train_imp = X_train_imp.replace([np.inf, -np.inf], 0.0)
X_test_imp  = X_test_imp.replace([np.inf, -np.inf], 0.0)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=FEATURES, index=X_train_imp.index)
X_test_s  = pd.DataFrame(scaler.transform(X_test_imp), columns=FEATURES, index=X_test_imp.index)

if np.isnan(X_train_s.values).any() or np.isnan(X_test_s.values).any():
    print("WARNING: NaN after scaling — filling with 0.0")
    X_train_s = X_train_s.fillna(0.0)
    X_test_s  = X_test_s.fillna(0.0)

# =========================
# 8) BASELINE
# =========================
y_pred_baseline = np.full(len(y_test), fill_value=float(y_train.mean()))
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
baseline_r2 = r2_score(y_test, y_pred_baseline)

print("\nBASELINE (train mean)")
print("---------------------")
print(f"RMSE: {baseline_rmse:.3f}, R2: {baseline_r2:.3f}")

# =========================
# 9) MODELY
# =========================
models_tree = {
    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=700,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        random_state=RANDOM_STATE
    ),
}

models_lin = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.005),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=5e-4,
        learning_rate="adaptive",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        max_iter=1200,
        random_state=RANDOM_STATE
    ),
}

results = []

for name, model in models_tree.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_imp, y_train)
    y_pred = model.predict(X_test_imp)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append((name, rmse, r2))
    print(f"{name} -> RMSE: {rmse:.3f}, R2: {r2:.3f}")

    joblib.dump(
        {"model": model, "imputer": imputer, "features": FEATURES},
        f"{name.lower()}_o22_radio_only.pkl"
    )

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\nFeature importance (top 15):\n", imp.head(15))

for name, model in models_lin.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append((name, rmse, r2))
    print(f"{name} -> RMSE: {rmse:.3f}, R2: {r2:.3f}")

    joblib.dump(
        {"model": model, "imputer": imputer, "scaler": scaler, "features": FEATURES},
        f"{name.lower()}_o22_radio_only.pkl"
    )

print("\nSUMMARY (radio-only)")
print("--------------------")
for name, rmse, r2 in results:
    print(f"{name:15s} -> RMSE: {rmse:.3f}, R2: {r2:.3f}")