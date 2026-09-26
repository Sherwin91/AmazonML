#!/usr/bin/env python3

"""
STEP 8 — BASELINE ENTITY MATCHING MODEL

Input:
    output/training_features.tsv

Output:
    output/entity_match_model.joblib
    output/model_metadata.txt

The model is deliberately lightweight because the machine has
limited RAM.

Evaluation:
    Precision
    Recall
    F0.5
    ROC-AUC

Threshold selection:
    Selects the validation threshold with the best F0.5.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "output" / "training_features.tsv"
MODEL_FILE = BASE_DIR / "output" / "entity_match_model.joblib"
METADATA_FILE = BASE_DIR / "output" / "model_metadata.txt"


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_SEED = 42

TEST_SIZE = 0.20

BETA = 0.5

FEATURE_COLUMNS = [
    "same_country",
    "name_exact",
    "name_compact_exact",
    "name_similarity",
    "name_token_similarity",
    "address_exact",
    "address_compact_exact",
    "address_similarity",
    "address_token_similarity",
    "address_digits_exact",
    "name_length_diff",
    "address_length_diff",
    "name_missing",
    "address_missing",
]


# =============================================================================
# F0.5 THRESHOLD SEARCH
# =============================================================================

def find_best_threshold(
    y_true,
    probabilities,
):

    best_threshold = 0.50
    best_fbeta = -1.0

    results = []

    # Fine threshold search.
    thresholds = np.arange(
        0.05,
        0.996,
        0.005,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        fbeta = fbeta_score(
            y_true,
            predictions,
            beta=BETA,
            zero_division=0,
        )

        results.append(
            (
                float(threshold),
                float(precision),
                float(recall),
                float(fbeta),
            )
        )

        if fbeta > best_fbeta:

            best_fbeta = fbeta
            best_threshold = threshold

    return (
        best_threshold,
        best_fbeta,
        results,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    overall_start = time.time()

    print("=" * 80)
    print("STEP 8 — BASELINE ENTITY MATCHING MODEL")
    print("=" * 80)

    # =========================================================================
    # LOAD
    # =========================================================================

    print()
    print("=" * 80)
    print("LOADING TRAINING FEATURES")
    print("=" * 80)

    print()
    print(f"Input: {INPUT_FILE}")

    start = time.time()

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
    )

    print()
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Load time: {time.time() - start:.2f}s")

    # =========================================================================
    # VERIFY
    # =========================================================================

    print()
    print("=" * 80)
    print("VERIFYING FEATURES")
    print("=" * 80)

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Missing required feature columns:\n"
            + "\n".join(missing_columns)
        )

    if "label" not in df.columns:

        raise RuntimeError(
            "label column missing."
        )

    print()
    print("All required features found.")

    print()
    print("Feature columns:")

    for column in FEATURE_COLUMNS:
        print(f"  {column}")

    # =========================================================================
    # CLEAN NUMERIC DATA
    # =========================================================================

    print()
    print("=" * 80)
    print("PREPARING FEATURES")
    print("=" * 80)

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        "label"
    ].astype(int)

    # Ensure everything is numeric.
    for column in FEATURE_COLUMNS:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # Replace invalid numeric values.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0.0)

    print()
    print(f"Feature matrix: {X.shape}")
    print(f"Positive labels: {(y == 1).sum():,}")
    print(f"Negative labels: {(y == 0).sum():,}")

    # =========================================================================
    # TRAIN / VALIDATION SPLIT
    # =========================================================================

    print()
    print("=" * 80)
    print("TRAIN / VALIDATION SPLIT")
    print("=" * 80)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print()
    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")

    print()
    print("Training labels:")
    print(f"  Positive: {(y_train == 1).sum():,}")
    print(f"  Negative: {(y_train == 0).sum():,}")

    print()
    print("Validation labels:")
    print(f"  Positive: {(y_valid == 1).sum():,}")
    print(f"  Negative: {(y_valid == 0).sum():,}")

    # =========================================================================
    # TRAIN MODEL
    # =========================================================================

    print()
    print("=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)

    print()
    print("Model:")
    print("  HistGradientBoostingClassifier")
    print("  max_iter = 200")
    print("  learning_rate = 0.08")
    print("  max_leaf_nodes = 31")

    model = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    training_time = time.time() - start

    print()
    print(
        f"Training completed in "
        f"{training_time:.2f} sec"
    )

    # =========================================================================
    # VALIDATION PROBABILITIES
    # =========================================================================

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    # =========================================================================
    # ROC-AUC
    # =========================================================================

    roc_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    print()
    print(f"ROC-AUC: {roc_auc:.6f}")

    # =========================================================================
    # THRESHOLD SEARCH
    # =========================================================================

    print()
    print("=" * 80)
    print("F0.5 THRESHOLD SEARCH")
    print("=" * 80)

    (
        best_threshold,
        best_fbeta,
        threshold_results,
    ) = find_best_threshold(
        y_valid.to_numpy(),
        probabilities,
    )

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    f05 = fbeta_score(
        y_valid,
        predictions,
        beta=BETA,
        zero_division=0,
    )

    print()
    print(f"Best threshold: {best_threshold:.3f}")
    print()
    print(f"Precision: {precision:.6f}")
    print(f"Recall:    {recall:.6f}")
    print(f"F0.5:      {f05:.6f}")
    print(f"ROC-AUC:   {roc_auc:.6f}")

    # =========================================================================
    # SHOW HIGH-PRECISION THRESHOLDS
    # =========================================================================

    print()
    print("=" * 80)
    print("SELECTED THRESHOLD ANALYSIS")
    print("=" * 80)

    interesting_thresholds = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.85,
        0.90,
        0.95,
    ]

    print()
    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'F0.5':>12}"
    )

    print("-" * 50)

    for threshold in interesting_thresholds:

        preds = (
            probabilities >= threshold
        ).astype(int)

        p = precision_score(
            y_valid,
            preds,
            zero_division=0,
        )

        r = recall_score(
            y_valid,
            preds,
            zero_division=0,
        )

        f = fbeta_score(
            y_valid,
            preds,
            beta=BETA,
            zero_division=0,
        )

        print(
            f"{threshold:>10.2f} "
            f"{p:>12.6f} "
            f"{r:>12.6f} "
            f"{f:>12.6f}"
        )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    print()
    print("=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)

    try:

        importance = model.feature_importances_

        feature_importance = sorted(
            zip(
                FEATURE_COLUMNS,
                importance,
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        print()

        for feature, value in feature_importance:

            print(
                f"{feature:35s} "
                f"{value:.6f}"
            )

    except Exception:

        print()
        print(
            "Feature importance is not available "
            "for this estimator."
        )

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    print()
    print("=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "threshold": float(best_threshold),
        "beta": BETA,
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f0.5": float(f05),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "random_seed": RANDOM_SEED,
    }

    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": float(best_threshold),
            "metadata": metadata,
        },
        MODEL_FILE,
    )

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "ENTITY MATCHING MODEL\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Rows: {len(df):,}\n"
        )

        file.write(
            f"ROC-AUC: {roc_auc:.6f}\n"
        )

        file.write(
            f"Precision: {precision:.6f}\n"
        )

        file.write(
            f"Recall: {recall:.6f}\n"
        )

        file.write(
            f"F0.5: {f05:.6f}\n"
        )

        file.write(
            f"Threshold: {best_threshold:.3f}\n\n"
        )

        file.write(
            "FEATURES\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for feature in FEATURE_COLUMNS:

            file.write(
                f"{feature}\n"
            )

    print()
    print(f"Model saved:")
    print(f"  {MODEL_FILE}")

    print()
    print(f"Metadata saved:")
    print(f"  {METADATA_FILE}")

    # =========================================================================
    # COMPLETE
    # =========================================================================

    total_time = time.time() - overall_start

    print()
    print("=" * 80)
    print("STEP 8 COMPLETE")
    print("=" * 80)

    print()
    print(f"ROC-AUC:        {roc_auc:.6f}")
    print(f"Precision:      {precision:.6f}")
    print(f"Recall:         {recall:.6f}")
    print(f"F0.5:           {f05:.6f}")
    print(f"Threshold:      {best_threshold:.3f}")

    print()
    print(f"Total runtime: {total_time:.2f} sec")

    print()
    print("NEXT:")
    print("  Inspect the validation results.")
    print("  Then we will build hard-negative mining.")
    print()


if __name__ == "__main__":
    main()
