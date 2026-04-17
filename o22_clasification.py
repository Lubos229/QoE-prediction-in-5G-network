import pandas as pd
import numpy as np
import csv
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier


# 1) nastavenia

CSV_PATH = "kompletcsv_final_filled.csv"
TIMESTAMP_COL = "timestamp"
TARGET = "statistics_input.p1203_output_report.O22"

RADIO_COLS = [
    "connection_detail_trace.rsrp",
    "connection_detail_trace.rsrq",
    "connection_detail_trace.rssnr",
]

VIDEO_COL = "client_report.video.resolution"

SEGMENT_LENGTH = 7
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

# mapovanie O22 na triedy rozlisenia
# 5.0 beriem ako chybnu hodnotu, ta sa vyhodi
MANUAL_CLASS_MAP = {
    3.852918928505388: "480p",
    4.482576544226886: "720p",
    4.90038555371506: "1080p",
    4.913700216003174: "2160p",
}

INVALID_O22_VALUES = {5.0, 4.8760506471620015}


# 2) pomocne funkcie

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
    # vyhod stlpce, kde v traine nie je ani jedna pouzitelna hodnota
    cols_all_nan = [c for c in Xtr.columns if Xtr[c].isna().all()]
    if cols_all_nan:
        print("Dropping columns with no observed values in TRAIN:", cols_all_nan)
        Xtr = Xtr.drop(columns=cols_all_nan)
        Xte = Xte.drop(columns=cols_all_nan, errors="ignore")
    return Xtr, Xte

def normalize_o22_value(x):
    if pd.isna(x):
        return np.nan
    return float(x)

def map_o22_to_label(x, manual_map, tol=1e-6):
    if pd.isna(x):
        return None

    x = float(x)

    for ref_value, label in manual_map.items():
        if abs(x - float(ref_value)) <= tol:
            return label

    return None

def build_class_mapping(values, manual_map, tol=1e-6):
    uniq = sorted(pd.Series(values).dropna().astype(float).unique())

    value_to_label = {}

    for v in uniq:
        label = map_o22_to_label(v, manual_map, tol=tol)
        if label is None:
            raise ValueError(f"MANUAL_CLASS_MAP neobsahuje O22 hodnotu blizku: {v}")
        value_to_label[float(v)] = label

    label_names = list(dict.fromkeys(value_to_label.values()))
    label_to_id = {label: i for i, label in enumerate(label_names)}

    value_to_class_id = {v: label_to_id[label] for v, label in value_to_label.items()}
    class_id_to_label = {i: label for label, i in label_to_id.items()}

    return value_to_label, value_to_class_id, class_id_to_label

def plot_conf_matrix(y_true, y_pred, labels, class_id_to_label, title="", normalize=False, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    label_names = [class_id_to_label[i] for i in labels]

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=True
    )
    plt.xlabel("Predikovana trieda")
    plt.ylabel("Skutocna trieda")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix plot to: {save_path}")

    plt.show()


# 3) nacitanie csv

with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
    reader = csv.reader(f, delimiter=";")
    rows = list(reader)

max_cols = max(len(r) for r in rows)
print(f"Max stlpcov v CSV: {max_cols}")

# uprava riadkov, aby sedel pocet stlpcov
clean_rows = []
for r in rows:
    if len(r) < max_cols:
        r.extend([None] * (max_cols - len(r)))
    elif len(r) > max_cols:
        r = r[:max_cols - 1] + [",".join(r[max_cols - 1:])]
    clean_rows.append(r)

df = pd.DataFrame(clean_rows[1:], columns=clean_rows[0])


# 4) vyhod riadky bez video resolution

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


# 5) pretypovanie targetu a radio stlpcov

if TARGET not in df.columns:
    raise KeyError(f"Chyba stlpec TARGET: {TARGET}")

df[TARGET] = df[TARGET].apply(to_num)

for c in RADIO_COLS:
    if c not in df.columns:
        raise KeyError(f"Chyba radiovy stlpec: {c}")
    df[c] = df[c].apply(to_num)

# vyhod chybne alebo mimo rozsah radio hodnoty
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

print("\nUnique raw TARGET values:")
print(sorted(df[TARGET].dropna().astype(float).unique()))


# 6) cas, zoradenie a pripadne forward fill

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

# doplnenie predoslej hodnoty, ak je to zapnute
if USE_FORWARD_FILL:
    df[RADIO_COLS] = df[RADIO_COLS].ffill(limit=FFILL_LIMIT)

if len(df) < SEGMENT_LENGTH + HORIZON:
    raise ValueError(
        f"Prilis malo riadkov po cisteni: {len(df)}. "
        f"Potrebne aspon {SEGMENT_LENGTH + HORIZON}."
    )


# 7) segmentacia do okien

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

    # ak v celom okne nie je ani jedna radio hodnota, segment preskoc
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

    feat["target_raw"] = float(target_val)
    segments.append(feat)

    segment_rows.append({
        "start_row": start,
        "end_row": end,
        "target_row": target_row
    })

df_segments = pd.DataFrame(segments)
df_seg_rows = pd.DataFrame(segment_rows)

print(f"Total segments before O22 filtering: {len(df_segments)}")

if len(df_segments) == 0:
    raise ValueError("Po segmentacii nevznikol ziaden segment.")

# vyhod feature stlpce, kde nie je nic
all_nan_cols = [c for c in df_segments.columns if c != "target_raw" and df_segments[c].isna().all()]
if all_nan_cols:
    print("Dropping all-NaN feature columns:", all_nan_cols)
    df_segments = df_segments.drop(columns=all_nan_cols)


# 8) odfiltrovanie zlych alebo nechcenych O22 hodnot

before_invalid = len(df_segments)

def is_invalid_o22(x, invalid_values, tol=1e-6):
    if pd.isna(x):
        return False
    x = float(x)
    for bad in invalid_values:
        if abs(x - float(bad)) <= tol:
            return True
    return False

valid_mask = ~df_segments["target_raw"].apply(
    lambda x: is_invalid_o22(x, INVALID_O22_VALUES, tol=1e-6)
)

df_segments = df_segments.loc[valid_mask].reset_index(drop=True)
df_seg_rows = df_seg_rows.loc[valid_mask].reset_index(drop=True)

print(f"Dropped segments with invalid O22 values: {before_invalid - len(df_segments)}")
print("Unique TARGET values after invalid filtering:")
print(sorted(df_segments["target_raw"].dropna().unique()))

if len(df_segments) == 0:
    raise ValueError("Po odfiltrovani neplatnych O22 hodnot nezostal ziaden segment.")


# 9) preved target na triedy

value_to_label, value_to_class_id, class_id_to_label = build_class_mapping(
    df_segments["target_raw"],
    manual_map=MANUAL_CLASS_MAP,
    tol=1e-6
)

df_segments["target_label"] = df_segments["target_raw"].map(value_to_label)
df_segments["target_class"] = df_segments["target_raw"].map(value_to_class_id)

print("\nTarget class mapping:")
for raw_val in sorted(value_to_label.keys()):
    print(f"O22={raw_val} -> label='{value_to_label[raw_val]}' -> class_id={value_to_class_id[raw_val]}")

print("\nClass distribution before split:")
print(df_segments["target_label"].value_counts(dropna=False).sort_index())


# 10) split s embargom

GAP = (SEGMENT_LENGTH - 1) + max(0, HORIZON)
split_row = int(len(df) * (1 - TEST_SIZE))

train_mask = df_seg_rows["end_row"] < (split_row - (SEGMENT_LENGTH - 1))
test_mask = df_seg_rows["start_row"] >= (split_row + GAP)

X = df_segments.drop(columns=["target_raw", "target_label", "target_class"])
y = df_segments["target_class"]

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

print("\nClass distribution in TRAIN:")
print(y_train.map(class_id_to_label).value_counts().sort_index())
print("\nClass distribution in TEST:")
print(y_test.map(class_id_to_label).value_counts().sort_index())


# 11) bez imputacie - vyhod riadky s NaN

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
print("Max abs X_train:", np.nanmax(np.abs(X_train.values)))
print("Max abs X_test:", np.nanmax(np.abs(X_test.values)))

# nechame len kompletne riadky bez NaN
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
    raise ValueError("Po odstraneni riadkov s NaN je train set prazdny.")
if len(X_test_clean) == 0:
    raise ValueError("Po odstraneni riadkov s NaN je test set prazdny.")

# scaling pre linearne modely
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

print("\nFEATURE diagnostics after scaling")
print("--------------------------------")
print("Any NaN in X_train_s?", X_train_s.isna().any().any())
print("Any NaN in X_test_s?", X_test_s.isna().any().any())
print("Any inf in X_train_s?", np.isinf(X_train_s.values).any())
print("Any inf in X_test_s?", np.isinf(X_test_s.values).any())
print("Max abs X_train_s:", np.abs(X_train_s.values).max())
print("Max abs X_test_s:", np.abs(X_test_s.values).max())


# 12) jednoduchy baseline

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train_clean, y_train_clean)
y_pred_baseline = baseline.predict(X_test_clean)

labels_sorted = sorted(class_id_to_label.keys())
target_names_sorted = [class_id_to_label[i] for i in labels_sorted]

print("\nBASELINE (most frequent class)")
print("-----------------------------")
print(f"Accuracy:          {accuracy_score(y_test_clean, y_pred_baseline):.3f}")
print(f"Balanced accuracy: {balanced_accuracy_score(y_test_clean, y_pred_baseline):.3f}")
print("Confusion matrix:")
print(confusion_matrix(y_test_clean, y_pred_baseline, labels=labels_sorted))
print("Classification report:")
print(classification_report(
    y_test_clean,
    y_pred_baseline,
    labels=labels_sorted,
    target_names=target_names_sorted,
    zero_division=0
))

plot_conf_matrix(
    y_test_clean,
    y_pred_baseline,
    labels=labels_sorted,
    class_id_to_label=class_id_to_label,
    title="Confusion matrix - Baseline",
    normalize=False,
    save_path="confusion_matrix_baseline.png"
)

plot_conf_matrix(
    y_test_clean,
    y_pred_baseline,
    labels=labels_sorted,
    class_id_to_label=class_id_to_label,
    title="Normalized confusion matrix - Baseline",
    normalize=True,
    save_path="confusion_matrix_baseline_normalized.png"
)

# 13) modely

num_classes = len(class_id_to_label)

models_tree = {
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=700,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=600,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        random_state=RANDOM_STATE
    ),
}

models_lin = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        multi_class="auto",
        random_state=RANDOM_STATE
    ),
    "RidgeClassifier": RidgeClassifier(alpha=1.0),
    "MLP": MLPClassifier(
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

# stromove modely
for name, model in models_tree.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_clean, y_train_clean)
    y_pred = model.predict(X_test_clean)

    acc = accuracy_score(y_test_clean, y_pred)
    bacc = balanced_accuracy_score(y_test_clean, y_pred)
    results.append((name, acc, bacc))

    print(f"{name} -> Accuracy: {acc:.3f}, Balanced accuracy: {bacc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test_clean, y_pred, labels=labels_sorted))
    print("Classification report:")
    print(classification_report(
        y_test_clean,
        y_pred,
        labels=labels_sorted,
        target_names=target_names_sorted,
        zero_division=0
    ))

    if name == "RandomForest":
        plot_conf_matrix(
            y_test_clean,
            y_pred,
            labels=labels_sorted,
            class_id_to_label=class_id_to_label,
            title="Confusion matrix - Random Forest",
            normalize=False,
            save_path="confusion_matrix_random_forest.png"
        )

        plot_conf_matrix(
            y_test_clean,
            y_pred,
            labels=labels_sorted,
            class_id_to_label=class_id_to_label,
            title="Normalized confusion matrix - Random Forest",
            normalize=True,
            save_path="confusion_matrix_random_forest_normalized.png"
        )

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\nFeature importance (top 15):")
        print(imp.head(15))

    save_obj = {
        "model": model,
        "features": FEATURES,
        "class_id_to_label": class_id_to_label,
        "value_to_label": value_to_label,
    }
    joblib.dump(save_obj, f"{name.lower()}_o22_classification_radio_only.pkl")

# linearne modely a MLP
for name, model in models_lin.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_s, y_train_clean)
    y_pred = model.predict(X_test_s)

    acc = accuracy_score(y_test_clean, y_pred)
    bacc = balanced_accuracy_score(y_test_clean, y_pred)
    results.append((name, acc, bacc))

    print(f"{name} -> Accuracy: {acc:.3f}, Balanced accuracy: {bacc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test_clean, y_pred, labels=labels_sorted))
    print("Classification report:")
    print(classification_report(
        y_test_clean,
        y_pred,
        labels=labels_sorted,
        target_names=target_names_sorted,
        zero_division=0
    ))

    if name == "LogisticRegression":
        plot_conf_matrix(
            y_test_clean,
            y_pred,
            labels=labels_sorted,
            class_id_to_label=class_id_to_label,
            title="Confusion matrix - Logistic Regression",
            normalize=False,
            save_path="confusion_matrix_logistic_regression.png"
        )

        plot_conf_matrix(
            y_test_clean,
            y_pred,
            labels=labels_sorted,
            class_id_to_label=class_id_to_label,
            title="Normalizovana confusion matrix - Logistic Regression",
            normalize=True,
            save_path="confusion_matrix_logistic_regression_normalized.png"
        )

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": FEATURES,
            "class_id_to_label": class_id_to_label,
            "value_to_label": value_to_label,
        },
        f"{name.lower()}_o22_classification_radio_only.pkl"
    )

# =========================
# 14) zhrnutie
# =========================
print("\nSUMMARY (radio-only classification)")
print("-----------------------------------")
for name, acc, bacc in results:
    print(f"{name:20s} -> Accuracy: {acc:.3f}, Balanced accuracy: {bacc:.3f}")