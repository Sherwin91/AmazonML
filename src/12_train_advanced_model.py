from __future__ import annotations

import csv
import os
import time
import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE,
    "output",
    "advanced_training_features.tsv",
)

MODEL_FILE = os.path.join(
    BASE,
    "output",
    "advanced_match_model.joblib",
)

META_FILE = os.path.join(
    BASE,
    "output",
    "advanced_model_metadata.txt",
)


# ============================================================
# F0.5
# ============================================================

def f05(precision, recall):

    beta = 0.5

    denominator = (
        beta * beta * precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta * beta)
        * precision
        * recall
        / denominator
    )


# ============================================================
# LOAD
# ============================================================

def main():

    start = time.time()

    print("=" * 80)
    print("STEP 12 — ADVANCED MODEL TRAINING")
    print("=" * 80)

    print()
    print(f"Input: {INPUT_FILE}")

    rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:
            rows.append(row)

    print(f"Rows loaded: {len(rows):,}")

    if not rows:
        raise RuntimeError(
            "No training data found."
        )

    # ========================================================
    # FEATURES
    # ========================================================

    excluded = {
        "s1_entity_id",
        "matched_entity_id",
        "matched_source",
        "label",
    }

    feature_names = [
        x
        for x in rows[0].keys()
        if x not in excluded
    ]

    print()
    print(
        f"Features: {len(feature_names)}"
    )

    print()
    print("Feature list:")

    for i, name in enumerate(
        feature_names,
        1,
    ):
        print(
            f"  {i:02d}. {name}"
        )

    # ========================================================
    # ARRAYS
    # ========================================================

    X = np.asarray(
        [
            [
                float(row[f])
                for f in feature_names
            ]
            for row in rows
        ],
        dtype=np.float32,
    )

    y = np.asarray(
        [
            int(row["label"])
            for row in rows
        ],
        dtype=np.int8,
    )

    groups = np.asarray(
        [
            row["s1_entity_id"]
            for row in rows
        ]
    )

    print()
    print(
        f"X shape: {X.shape}"
    )

    print(
        f"Positive: {np.sum(y == 1):,}"
    )

    print(
        f"Negative: {np.sum(y == 0):,}"
    )

    print(
        f"Unique S1 entities: "
        f"{len(np.unique(groups)):,}"
    )

    # ========================================================
    # GROUPED SPLIT
    # ========================================================

    print()
    print("=" * 80)
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
            groups,
        )
    )

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    train_groups = set(
        groups[train_idx]
    )

    val_groups = set(
        groups[val_idx]
    )

    print(
        f"Training rows:   "
        f"{len(train_idx):,}"
    )

    print(
        f"Validation rows: "
        f"{len(val_idx):,}"
    )

    print(
        f"Training S1:     "
        f"{len(train_groups):,}"
    )

    print(
        f"Validation S1:   "
        f"{len(val_groups):,}"
    )

    print(
        f"S1 overlap:      "
        f"{len(train_groups & val_groups)}"
    )

    if train_groups & val_groups:

        raise RuntimeError(
            "GROUP LEAKAGE DETECTED"
        )

    print(
        "GROUP LEAKAGE CHECK: PASSED"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    model = HistGradientBoostingClassifier(

        max_iter=350,

        learning_rate=0.055,

        max_leaf_nodes=31,

        min_samples_leaf=25,

        l2_regularization=1.5,

        random_state=42,
    )

    train_start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    train_time = (
        time.time()
        - train_start
    )

    print(
        f"Training time: "
        f"{train_time:.2f}s"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

    auc = roc_auc_score(
        y_val,
        probabilities,
    )

    print(
        f"Grouped ROC-AUC: "
        f"{auc:.6f}"
    )

    # ========================================================
    # THRESHOLD SEARCH
    # ========================================================

    print()
    print(
        "Threshold analysis:"
    )

    best = None

    thresholds = np.arange(
        0.50,
        0.991,
        0.005,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        score = f05(
            precision,
            recall,
        )

        if (
            best is None
            or score > best["f05"]
        ):

            best = {
                "threshold": float(
                    threshold
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f05": float(
                    score
                ),
            }

    print()
    print(
        f"Best threshold: "
        f"{best['threshold']:.3f}"
    )

    print(
        f"Precision:       "
        f"{best['precision']:.6f}"
    )

    print(
        f"Recall:          "
        f"{best['recall']:.6f}"
    )

    print(
        f"F0.5:            "
        f"{best['f05']:.6f}"
    )

    # ========================================================
    # FULL THRESHOLD TABLE
    # ========================================================

    print()
    print(
        "Selected thresholds:"
    )

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.82,
        0.85,
        0.90,
        0.92,
        0.95,
    ]:

        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        score = f05(
            precision,
            recall,
        )

        print(
            f"  {threshold:.2f}  "
            f"P={precision:.6f}  "
            f"R={recall:.6f}  "
            f"F0.5={score:.6f}"
        )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print()
    print(
        "Feature importance:"
    )

    try:

        importance = model.feature_importances_

        order = np.argsort(
            importance
        )[::-1]

        for idx in order:

            print(
                f"  "
                f"{feature_names[idx]:30s} "
                f"{importance[idx]:.6f}"
            )

    except Exception:

        print(
            "Feature importance unavailable "
            "for HistGradientBoostingClassifier."
        )

    # ========================================================
    # SAVE
    # ========================================================

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "threshold": best["threshold"],
        "validation_auc": auc,
        "validation_precision": best["precision"],
        "validation_recall": best["recall"],
        "validation_f05": best["f05"],
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
        compress=3,
    )

    metadata = f"""
STEP 12 — ADVANCED MODEL

Rows:
  {len(rows):,}

Features:
  {len(feature_names)}

Training rows:
  {len(train_idx):,}

Validation rows:
  {len(val_idx):,}

Training S1:
  {len(train_groups):,}

Validation S1:
  {len(val_groups):,}

Group overlap:
  0

Grouped ROC-AUC:
  {auc:.8f}

Best threshold:
  {best["threshold"]:.4f}

Precision:
  {best["precision"]:.8f}

Recall:
  {best["recall"]:.8f}

F0.5:
  {best["f05"]:.8f}

Training time:
  {train_time:.2f}s

Total runtime:
  {time.time() - start:.2f}s
"""

    with open(
        META_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(metadata)

    print()
    print("=" * 80)
    print("STEP 12 COMPLETE")
    print("=" * 80)

    print(metadata)

    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Metadata: {META_FILE}"
    )

    print()
    print("BASELINE TO BEAT:")
    print("  Grouped ROC-AUC: 0.999841")
    print("  Grouped F0.5:    0.996704")


if __name__ == "__main__":
    main()
