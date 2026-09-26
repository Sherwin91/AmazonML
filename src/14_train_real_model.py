#!/usr/bin/env python3

import csv
import os
import time
import joblib

import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    fbeta_score,
)
from sklearn.model_selection import GroupShuffleSplit


BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE,
    "output",
    "real_raw_training_features.tsv"
)

MODEL_FILE = os.path.join(
    BASE,
    "output",
    "real_raw_match_model.joblib"
)

METADATA_FILE = os.path.join(
    BASE,
    "output",
    "real_raw_model_metadata.txt"
)


# ---------------------------------------------------------------------
# SAFE NUMERIC CONVERSION
# ---------------------------------------------------------------------

def to_float(value):
    """
    Convert TSV feature values safely.

    Handles:
      0 / 1
      0.5 / 1.0
      True / False
      empty values
    """

    if value is None:
        return 0.0

    value = str(value).strip()

    if value == "":
        return 0.0

    lower = value.lower()

    if lower == "true":
        return 1.0

    if lower == "false":
        return 0.0

    return float(value)


# ---------------------------------------------------------------------
# START
# ---------------------------------------------------------------------

print("=" * 80)
print("STEP 14 — TRAIN REAL RAW-FEATURE MODEL")
print("=" * 80)

start = time.time()


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

print("\nLoading features...")

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    fieldnames = reader.fieldnames

    if fieldnames is None:
        raise RuntimeError(
            "Could not read feature header."
        )

    rows = list(reader)


id_columns = {
    "s1_entity_id",
    "matched_entity_id",
    "matched_source",
    "label",
}

feature_names = [
    c
    for c in fieldnames
    if c not in id_columns
]


print(
    f"Rows: {len(rows):,}"
)

print(
    f"Features: {len(feature_names)}"
)


# ---------------------------------------------------------------------
# BUILD MATRICES
# ---------------------------------------------------------------------

X = np.asarray(
    [
        [
            to_float(row[c])
            for c in feature_names
        ]
        for row in rows
    ],
    dtype=np.float32
)

y = np.asarray(
    [
        int(row["label"])
        for row in rows
    ],
    dtype=np.int8
)

groups = np.asarray(
    [
        row["s1_entity_id"]
        for row in rows
    ]
)


print(
    f"X shape: {X.shape}"
)

print(
    f"Positive: {int(y.sum()):,}"
)

print(
    f"Negative: {int((y == 0).sum()):,}"
)

print(
    f"Unique S1: {len(np.unique(groups)):,}"
)


# ---------------------------------------------------------------------
# SANITY CHECK
# ---------------------------------------------------------------------

if not np.isfinite(X).all():
    raise RuntimeError(
        "Feature matrix contains NaN or infinite values."
    )

print(
    "Feature sanity check: PASSED"
)


# ---------------------------------------------------------------------
# GROUP SPLIT
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("GROUPED TRAIN / VALIDATION SPLIT")
print("=" * 80)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, val_idx = next(
    splitter.split(
        X,
        y,
        groups
    )
)

X_train = X[train_idx]
X_val = X[val_idx]

y_train = y[train_idx]
y_val = y[val_idx]

groups_train = groups[train_idx]
groups_val = groups[val_idx]


overlap = len(
    set(groups_train)
    &
    set(groups_val)
)


print(
    f"Training rows:   {len(train_idx):,}"
)

print(
    f"Validation rows: {len(val_idx):,}"
)

print(
    f"Training S1:     {len(np.unique(groups_train)):,}"
)

print(
    f"Validation S1:   {len(np.unique(groups_val)):,}"
)

print(
    f"S1 overlap:      {overlap}"
)


if overlap != 0:
    raise RuntimeError(
        "GROUP LEAKAGE DETECTED"
    )

print(
    "GROUP LEAKAGE CHECK: PASSED"
)


# ---------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

model = HistGradientBoostingClassifier(
    max_iter=400,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=25,
    l2_regularization=1.5,
    random_state=42,
)

train_start = time.time()

model.fit(
    X_train,
    y_train
)

train_time = time.time() - train_start

print(
    f"Training time: {train_time:.2f}s"
)


# ---------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)

prob = model.predict_proba(
    X_val
)[:, 1]


auc = roc_auc_score(
    y_val,
    prob
)

print(
    f"Grouped ROC-AUC: {auc:.9f}"
)


# ---------------------------------------------------------------------
# THRESHOLD SEARCH
# ---------------------------------------------------------------------

best_threshold = None
best_f05 = -1.0
best_precision = 0.0
best_recall = 0.0


print("\nThreshold analysis:")


for threshold in np.arange(
    0.50,
    0.991,
    0.005
):

    pred = (
        prob >= threshold
    ).astype(np.int8)

    precision = precision_score(
        y_val,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        pred,
        zero_division=0
    )

    f05 = fbeta_score(
        y_val,
        pred,
        beta=0.5,
        zero_division=0
    )

    if f05 > best_f05:

        best_f05 = f05

        best_threshold = float(
            threshold
        )

        best_precision = precision
        best_recall = recall


print(
    f"\nBest threshold: {best_threshold:.3f}"
)

print(
    f"Precision:       {best_precision:.9f}"
)

print(
    f"Recall:           {best_recall:.9f}"
)

print(
    f"F0.5:            {best_f05:.9f}"
)


# ---------------------------------------------------------------------
# IMPORTANT THRESHOLDS
# ---------------------------------------------------------------------

print("\nSelected thresholds:")

for threshold in [
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.82,
    0.85,
    0.88,
    0.90,
    0.92,
    0.95,
    0.97,
]:

    pred = (
        prob >= threshold
    ).astype(np.int8)

    precision = precision_score(
        y_val,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        pred,
        zero_division=0
    )

    f05 = fbeta_score(
        y_val,
        pred,
        beta=0.5,
        zero_division=0
    )

    print(
        f"  {threshold:.2f}  "
        f"P={precision:.6f}  "
        f"R={recall:.6f}  "
        f"F0.5={f05:.6f}"
    )


# ---------------------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------------------

artifact = {
    "model": model,
    "feature_names": feature_names,
    "threshold": best_threshold,
    "grouped_auc": auc,
    "grouped_f05": best_f05,
    "precision": best_precision,
    "recall": best_recall,
}


joblib.dump(
    artifact,
    MODEL_FILE
)


# ---------------------------------------------------------------------
# SAVE METADATA
# ---------------------------------------------------------------------

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "REAL RAW-FEATURE MODEL\n"
    )

    f.write(
        f"Rows: {len(rows)}\n"
    )

    f.write(
        f"Features: {len(feature_names)}\n"
    )

    f.write(
        f"Grouped ROC-AUC: {auc:.9f}\n"
    )

    f.write(
        f"Best threshold: {best_threshold:.6f}\n"
    )

    f.write(
        f"Precision: {best_precision:.9f}\n"
    )

    f.write(
        f"Recall: {best_recall:.9f}\n"
    )

    f.write(
        f"F0.5: {best_f05:.9f}\n"
    )

    f.write(
        "\nFEATURES\n"
    )

    for i, name in enumerate(
        feature_names,
        1
    ):

        f.write(
            f"{i:02d}. {name}\n"
        )


# ---------------------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------------------

elapsed = time.time() - start

print("\n" + "=" * 80)
print("STEP 14 COMPLETE")
print("=" * 80)

print(
    f"\nModel:\n{MODEL_FILE}"
)

print(
    f"\nMetadata:\n{METADATA_FILE}"
)

print(
    f"\nTotal runtime: {elapsed:.2f}s"
)


print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    "\nStep 9 baseline:"
)

print(
    "  Grouped ROC-AUC: 0.999841"
)

print(
    "  Grouped F0.5:    0.996704"
)


print(
    "\nStep 12:"
)

print(
    "  Grouped ROC-AUC: 0.999840"
)

print(
    "  Grouped F0.5:    0.996752"
)


print(
    "\nStep 14 real raw features:"
)

print(
    f"  Grouped ROC-AUC: {auc:.9f}"
)

print(
    f"  Grouped F0.5:    {best_f05:.9f}"
)


if best_f05 > 0.996752:

    print(
        "\nRESULT: NEW BEST MODEL"
    )

elif best_f05 > 0.996704:

    print(
        "\nRESULT: IMPROVES STEP 9, "
        "BUT DOES NOT BEAT STEP 12"
    )

else:

    print(
        "\nRESULT: STEP 12 REMAINS BEST"
    )