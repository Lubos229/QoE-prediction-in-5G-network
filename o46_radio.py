import pandas as pd
import numpy as np
import joblib

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.inspection import permutation_importance


# 1) nastavenia

CSV_PATH = "iba_sedmerovec_data.csv"
SEP = ";"

TIMESTAMP_COL = "timestamp"
START_COL = "study.started_at"
END_COL   = "study.finished_at"
TARGET = "statistics_input.p1203_output_report.O46"

RADIO_COLS = [
    "connection_detail_trace.rsrp",
    "connection_detail_trace.rsrq",
    "connection_detail_trace.rssnr",
]

VIDEO_COL = "client_report.video.resolution"

TEST_SIZE = 0.2
RANDOM_STATE = 42

USE_FORWARD_FILL = False
FFILL_LIMIT = 5


# 2) pomocne funkcie
def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def parse_time(col):
    return pd.to_datetime(
        col.astype(str)
           .str.replace(r"\s+", " ", regex=True)
           .str.strip(),
        errors="coerce"
    )

def series_stats(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").dropna()

    return {
        "mean": float(s.mean()),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "min": float(s.min()),
        "max": float(s.max()),
    }

def below_prop(s: pd.Series, thresh: float):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float((s < thresh).mean()) if len(s) else np.nan

def drop_all_nan_train_columns(Xtr: pd.DataFrame, Xte: pd.DataFrame):
    # vyhod stlpce, kde v train casti nie je ani jedna hodnota
    cols_all_nan = [c for c in Xtr.columns if Xtr[c].isna().all()]
    if cols_all_nan:
        print("Dropping columns with no observed values in TRAIN:", cols_all_nan)
        Xtr = Xtr.drop(columns=cols_all_nan)
        Xte = Xte.drop(columns=cols_all_nan, errors="ignore")
    return Xtr, Xte


# 3) nacitanie csv

df = pd.read_csv(CSV_PATH, sep=SEP, low_memory=False)
df.columns = df.columns.str.strip()

required_cols = [TIMESTAMP_COL, START_COL, END_COL, TARGET, VIDEO_COL] + RADIO_COLS
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise KeyError(f"Missing required columns: {missing_required}")

df[TIMESTAMP_COL] = parse_time(df[TIMESTAMP_COL])
df[START_COL] = parse_time(df[START_COL])
df[END_COL] = parse_time(df[END_COL])

df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)


# 3.1) vyhod riadky bez video resolution
before = len(df)
df[VIDEO_COL] = df[VIDEO_COL].astype(str).str.strip()

df = df[
    df[VIDEO_COL].notna() &
    (df[VIDEO_COL] != "") &
    (df[VIDEO_COL].str.lower() != "none") &
    (df[VIDEO_COL].str.lower() != "nan")
].copy()

print(f"Dropped rows without video resolution: {before - len(df)}")


# 3.2) pretypovanie na cisla + cistenie radio hodnot

def to_num(x):
    return pd.to_numeric(str(x).replace(",", ".").strip(), errors="coerce")

df[TARGET] = df[TARGET].apply(to_num)

for c in RADIO_COLS:
    df[c] = df[c].apply(to_num)

RADIO_BOUNDS = {
    "connection_detail_trace.rsrp": (-160, -30),
    "connection_detail_trace.rsrq": (-40, 20),
    "connection_detail_trace.rssnr": (-50, 50),
}

# odstran sentinel hodnoty a veci mimo realny rozsah
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

if USE_FORWARD_FILL:
    df[RADIO_COLS] = df[RADIO_COLS].ffill(limit=FFILL_LIMIT)


# 4) rozdelenie na segmenty

df = df.dropna(subset=[START_COL, END_COL]).copy()
df["seg_id"] = df[START_COL].astype(str) + "_" + df[END_COL].astype(str)

print("Unique segments (runs):", df["seg_id"].nunique())

pairs = df[[START_COL, END_COL, "seg_id"]].drop_duplicates()
print("\n--- SEGMENT ID CHECK ---")
print("Unique (start,end) pairs:", len(pairs))

dur = (df.groupby("seg_id")[END_COL].max() - df.groupby("seg_id")[START_COL].min()).dt.total_seconds()
dur_valid = dur.dropna()
print(
    "Duration (s) min/median/p95/max:",
    float(dur_valid.min()),
    float(dur_valid.median()),
    float(np.nanpercentile(dur_valid, 95)),
    float(dur_valid.max())
)

rows_per_seg = df.groupby("seg_id").size()
print(
    "Rows per segment -> min/median/p95/max:",
    int(rows_per_seg.min()),
    int(rows_per_seg.median()),
    int(rows_per_seg.quantile(0.95)),
    int(rows_per_seg.max())
)
print("--------------------------------------------\n")


# 5) vytiahnutie featureov zo segmentov

segments = []

for seg_id, seg in df.groupby("seg_id", sort=False):
    tgt = seg[TARGET].dropna()

    if len(tgt) == 0:
        continue

    target_val = float(tgt.iloc[0])

    rsrp = series_stats(seg[RADIO_COLS[0]])
    rsrq = series_stats(seg[RADIO_COLS[1]])
    snr  = series_stats(seg[RADIO_COLS[2]])

    feats = {}

    def add_stats(prefix, st):
        for k in ["mean", "p25", "p75", "min", "max"]:
            feats[f"{prefix}_{k}"] = st[k]

    add_stats("rsrp", rsrp)
    add_stats("rsrq", rsrq)
    add_stats("rssnr", snr)

    # tieto featurey som zatial nechal vypnute
    # feats["rsrp_below_-110"] = below_prop(seg[RADIO_COLS[0]], -110)
    # feats["rsrp_below_-100"] = below_prop(seg[RADIO_COLS[0]], -100)
    # feats["rsrq_below_-14"]  = below_prop(seg[RADIO_COLS[1]], -14)
    # feats["rssnr_below_0"]   = below_prop(seg[RADIO_COLS[2]], 0)

    feats["seg_id"] = seg_id
    feats["target"] = target_val
    segments.append(feats)

df_segments = pd.DataFrame(segments).reset_index(drop=True)
print("Total O46 segments (samples):", len(df_segments))

if len(df_segments) == 0:
    raise ValueError("No O46 segments created after feature extraction.")

all_nan_cols = [
    c for c in df_segments.columns
    if c not in ["target", "seg_id"] and df_segments[c].isna().all()
]
if all_nan_cols:
    print("Dropping all-NaN columns:", all_nan_cols)
    df_segments = df_segments.drop(columns=all_nan_cols)

expected_cols = [
    "rsrp_mean", "rsrp_p25", "rsrp_p75", "rsrp_min", "rsrp_max",
    "rsrq_mean", "rsrq_p25", "rsrq_p75", "rsrq_min", "rsrq_max",
    "rssnr_mean", "rssnr_p25", "rssnr_p75", "rssnr_min", "rssnr_max",
]
present_cols = [c for c in expected_cols if c in df_segments.columns]
missing_cols = [c for c in expected_cols if c not in df_segments.columns]
print("Feature cols present:", present_cols)
if missing_cols:
    print("Feature cols missing:", missing_cols)


# 6) train/test split

FEATURES = [c for c in df_segments.columns if c not in ["target", "seg_id"]]

X = df_segments[FEATURES]
y = df_segments["target"]

X_train, X_test, y_train, y_test, segid_train, segid_test = train_test_split(
    X, y, df_segments["seg_id"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
segid_train = segid_train.reset_index(drop=True)
segid_test = segid_test.reset_index(drop=True)

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Train or test split is empty.")

print(f"Train segments: {len(X_train)}  |  Test segments: {len(X_test)}\n")

print("--- TARGET DISTRIBUTION ---")
print("FULL std:", float(y.std()))
print("TRAIN std:", float(y_train.std()))
print("TEST std:", float(y_test.std()))
print("TRAIN min/max:", float(y_train.min()), float(y_train.max()))
print("TEST min/max:", float(y_test.min()), float(y_test.max()))
print("\nTRAIN describe:\n", y_train.describe())
print("\nTEST describe:\n", y_test.describe())
print("---------------------------\n")


# 7) vyhod NaN a sprav scaling

X_train, X_test = drop_all_nan_train_columns(X_train, X_test)
FEATURES = list(X_train.columns)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test  = X_test.replace([np.inf, -np.inf], np.nan)

print("\nFEATURE diagnostics before dropping NaN rows")
print("-------------------------------------------")
print("Any NaN in X_train?", X_train.isna().any().any())
print("Any NaN in X_test?", X_test.isna().any().any())

train_keep_mask = ~X_train.isna().any(axis=1)
test_keep_mask  = ~X_test.isna().any(axis=1)

X_train_clean = X_train.loc[train_keep_mask].reset_index(drop=True)
y_train_clean = y_train.loc[train_keep_mask].reset_index(drop=True)

X_test_clean = X_test.loc[test_keep_mask].reset_index(drop=True)
y_test_clean = y_test.loc[test_keep_mask].reset_index(drop=True)

print("\nComplete-case filtering")
print("-----------------------")
print(f"Train rows kept: {len(X_train_clean)} / {len(X_train)}")
print(f"Test rows kept:  {len(X_test_clean)} / {len(X_test)}")

if len(X_train_clean) == 0:
    raise ValueError("Po odstraneni riadkov s NaN je train set prazdny.")
if len(X_test_clean) == 0:
    raise ValueError("Po odstraneni riadkov s NaN je test set prazdny.")

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


# 8) baseline

y_pred_baseline = np.full(len(y_test_clean), fill_value=float(y_train_clean.mean()))
baseline_rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred_baseline))
baseline_r2 = r2_score(y_test_clean, y_pred_baseline)

print(f"RMSE: {baseline_rmse:.3f}, R2: {baseline_r2:.3f}")


# 9) modely

models_tree = {
    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=400, max_depth=10, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=800, learning_rate=0.03, max_depth=3,
        random_state=RANDOM_STATE, loss="huber"
    ),
}

models_lin = {
    "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "Lasso": Lasso(alpha=0.005, random_state=RANDOM_STATE),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        early_stopping=True,
        validation_fraction=0.2,
        max_iter=1500,
        random_state=RANDOM_STATE
    ),
}

def evaluate_and_save(models, Xtr, Xte, ytr, yte, scaled=False):
    results = []

    for name, model in models.items():
        print(f"\nTraining {name} ({'scaled' if scaled else 'raw'})...")
        model.fit(Xtr, ytr)
        y_pred = model.predict(Xte)

        rmse = np.sqrt(mean_squared_error(yte, y_pred))
        r2 = r2_score(yte, y_pred)

        results.append((name, rmse, r2))
        print(f"{name:15s} -> RMSE: {rmse:.3f}, R2: {r2:.3f}")

        payload = {"model": model, "features": FEATURES}
        if scaled:
            payload["scaler"] = scaler
        joblib.dump(payload, f"{name.lower()}_o46_segment.pkl")

        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
            print("\nFeature importance (top 15):\n", imp.head(15))

    return results

results_tree = evaluate_and_save(models_tree, X_train_clean, X_test_clean, y_train_clean, y_test_clean, scaled=False)
results_lin  = evaluate_and_save(models_lin,  X_train_s,     X_test_s,     y_train_clean, y_test_clean, scaled=True)

print("\nSUMMARY (O46 radio-only)")
print("------------------------")
for name, rmse, r2 in results_tree + results_lin:
    print(f"{name:15s} -> rmse: {rmse:.3f}, R2: {r2:.3f}")


# 10) 5-fold k-fold validacia

print("\n--- 5-FOLD K-FOLD (RF) ---")
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rmse_list, r2_list = [], []

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

for tr_idx, te_idx in kf.split(X_train_clean):
    Xtr_fold = X_train_clean.iloc[tr_idx]
    Xte_fold = X_train_clean.iloc[te_idx]
    ytr_fold = y_train_clean.iloc[tr_idx]
    yte_fold = y_train_clean.iloc[te_idx]

    rf.fit(Xtr_fold, ytr_fold)
    pred = rf.predict(Xte_fold)

    rmse_list.append(np.sqrt(mean_squared_error(yte_fold, pred)))
    r2_list.append(r2_score(yte_fold, pred))

print(
    f"KFold RF -> "
    f"RMSE: {np.mean(rmse_list):.3f} +/- {np.std(rmse_list):.3f}, "
    f"R2: {np.mean(r2_list):.3f} +/- {np.std(r2_list):.3f}"
)


# 11) permutation importance

print("\n--- PERMUTATION IMPORTANCE (RF on TEST) ---")
rf_fit = RandomForestRegressor(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
).fit(X_train_clean, y_train_clean)

r = permutation_importance(
    rf_fit,
    X_test_clean,
    y_test_clean,
    n_repeats=15,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pi = pd.Series(r.importances_mean, index=X_test_clean.columns).sort_values(ascending=False)
print(pi.head(15))