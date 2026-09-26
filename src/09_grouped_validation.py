#!/usr/bin/env python3

"""
STEP 9 — GROUPED ENTITY VALIDATION

Purpose:
    Validate the entity matching model without allowing the same S1
    entity to appear in both training and validation.

Why:
    Pair-level random splitting can leak S1-specific patterns between
    train and validation.

Input:
    output/training_features.tsv

Output:
    output/grouped_validation_model.joblib
    output/grouped_validation_metadata.txt
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
from sklearn.model_selection import GroupShuffleSplit


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "output"
    / "training_features.tsv"
)

MODEL_FILE = (
    BASE_DIR
    / "output"
    / "grouped_validation_model.joblib"
)

METADATA_FILE = (
    BASE_DIR
    / "output"
    / "grouped_validation_metadata.txt"
)


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_SEED = 42

VALIDATION_SIZE = 0.20

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
# THRESHOLD SEARCH
# =============================================================================

def find_best_threshold(y_true, probabilities):

    best_threshold = 0.50
    best_f05 = -1.0

    results = []

    thresholds = np.arange(
        0.05,
        0.996,
        0.005,
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(np.int8)

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

        f05 = fbeta_score(
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
                float(f05),
            )
        )

        if f05 > best_f05:

            best_f05 = f05
            best_threshold = float(threshold)

    return (
        best_threshold,
        best_f05,
        results,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    overall_start = time.time()

    print("=" * 80)
    print("STEP 9 — GROUPED ENTITY VALIDATION")
    print("=" * 80)

    # =========================================================================
    # LOAD
    # =========================================================================

    print()
    print("=" * 80)
    print("LOADING FEATURES")
    print("=" * 80)

    print()
    print(f"Input: {INPUT_FILE}")

    start = time.time()

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
    )

    print()
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(
        f"Load time: {time.time() - start:.2f}s"
    )

    # =========================================================================
    # VERIFY
    # =========================================================================

    print()
    print("=" * 80)
    print("VERIFYING DATA")
    print("=" * 80)

    required = (
        ["s1_entity_id", "label"]
        + FEATURE_COLUMNS
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )

    print()
    print("All required columns found.")

    # =========================================================================
    # PREPARE
    # =========================================================================

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        "label"
    ].astype(np.int8)

    groups = df[
        "s1_entity_id"
    ].astype(str)

    for column in FEATURE_COLUMNS:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0.0)

    print()
    print(f"Feature matrix: {X.shape}")
    print(
        f"Unique S1 entities: "
        f"{groups.nunique():,}"
    )

    print()
    print(
        f"Positive pairs: {(y == 1).sum():,}"
    )

    print(
        f"Negative pairs: {(y == 0).sum():,}"
    )

    # =========================================================================
    # GROUPED SPLIT
    # =========================================================================

    print()
    print("=" * 80)
    print("GROUPED TRAIN / VALIDATION SPLIT")
    print("=" * 80)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
    )

    train_indices, valid_indices = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[
        train_indices
    ]

    X_valid = X.iloc[
        valid_indices
    ]

    y_train = y.iloc[
        train_indices
    ]

    y_valid = y.iloc[
        valid_indices
    ]

    train_groups = set(
        groups.iloc[
            train_indices
        ]
    )

    valid_groups = set(
        groups.iloc[
            valid_indices
        ]
    )

    overlap = (
        train_groups
        & valid_groups
    )

    print()
    print(
        f"Training pairs:   {len(X_train):,}"
    )

    print(
        f"Validation pairs: {len(X_valid):,}"
    )

    print(
        f"Training S1 entities: "
        f"{len(train_groups):,}"
    )

    print(
        f"Validation S1 entities: "
        f"{len(valid_groups):,}"
    )

    print(
        f"S1 overlap: {len(overlap):,}"
    )

    if overlap:

        raise RuntimeError(
            "GROUP LEAKAGE DETECTED."
        )

    print()
    print("GROUP LEAKAGE CHECK: PASSED")

    print()
    print("Training labels:")

    print(
        f"  Positive: {(y_train == 1).sum():,}"
    )

    print(
        f"  Negative: {(y_train == 0).sum():,}"
    )

    print()
    print("Validation labels:")

    print(
        f"  Positive: {(y_valid == 1).sum():,}"
    )

    print(
        f"  Negative: {(y_valid == 0).sum():,}"
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    print()
    print("=" * 80)
    print("TRAINING GROUPED MODEL")
    print("=" * 80)

    model = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )

    print()
    print("Model:")
    print("  HistGradientBoostingClassifier")
    print("  max_iter = 250")
    print("  learning_rate = 0.06")
    print("  max_leaf_nodes = 31")
    print("  min_samples_leaf = 30")

    start = time.time()

    model.fit(
        X_train,
        y_train,
    )

    training_time = time.time() - start

    print()
    print(
        f"Training time: "
        f"{training_time:.2f}s"
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    print()
    print("=" * 80)
    print("GROUPED VALIDATION RESULTS")
    print("=" * 80)

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    roc_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    (
        best_threshold,
        best_f05,
        threshold_results,
    ) = find_best_threshold(
        y_valid.to_numpy(),
        probabilities,
    )

    predictions = (
        probabilities >= best_threshold
    ).astype(np.int8)

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
    print(
        f"ROC-AUC:   {roc_auc:.6f}"
    )

    print(
        f"Threshold: {best_threshold:.3f}"
    )

    print(
        f"Precision: {precision:.6f}"
    )

    print(
        f"Recall:    {recall:.6f}"
    )

    print(
        f"F0.5:      {f05:.6f}"
    )

    # =========================================================================
    # THRESHOLD TABLE
    # =========================================================================

    print()
    print("=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)

    print()
    print(
        f"{'Threshold':>10} "
        f"{'Precision':>12} "
        f"{'Recall':>12} "
        f"{'F0.5':>12}"
    )

    print("-" * 50)

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.82,
        0.85,
        0.90,
        0.95,
    ]:

        preds = (
            probabilities >= threshold
        ).astype(np.int8)

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
    # COMPARE WITH STEP 8
    # =========================================================================

    old_f05 = 0.997082
    old_auc = 0.999878

    print()
    print("=" * 80)
    print("COMPARISON WITH STEP 8")
    print("=" * 80)

    print()
    print(
        f"{'Metric':<15}"
        f"{'Step 8':>15}"
        f"{'Step 9':>15}"
        f"{'Change':>15}"
    )

    print("-" * 60)

    auc_change = roc_auc - old_auc
    f05_change = f05 - old_f05

    print(
        f"{'ROC-AUC':<15}"
        f"{old_auc:>15.6f}"
        f"{roc_auc:>15.6f}"
        f"{auc_change:>+15.6f}"
    )

    print(
        f"{'F0.5':<15}"
        f"{old_f05:>15.6f}"
        f"{f05:>15.6f}"
        f"{f05_change:>+15.6f}"
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    print()
    print("=" * 80)
    print("SAVING GROUPED MODEL")
    print("=" * 80)

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "threshold": float(best_threshold),
        "beta": BETA,
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f0.5": float(f05),
        "training_pairs": int(len(X_train)),
        "validation_pairs": int(len(X_valid)),
        "training_s1_entities": int(
            len(train_groups)
        ),
        "validation_s1_entities": int(
            len(valid_groups)
        ),
        "s1_overlap": int(len(overlap)),
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
            "GROUPED ENTITY MATCHING MODEL\n"
        )

        file.write(
            "=" * 60 + "\n\n"
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
            f"Threshold: {best_threshold:.3f}\n"
        )

        file.write(
            f"Training pairs: {len(X_train):,}\n"
        )

        file.write(
            f"Validation pairs: {len(X_valid):,}\n"
        )

        file.write(
            f"Training S1 entities: "
            f"{len(train_groups):,}\n"
        )

        file.write(
            f"Validation S1 entities: "
            f"{len(valid_groups):,}\n"
        )

        file.write(
            f"S1 overlap: {len(overlap):,}\n"
        )

    print()
    print(f"Model:")
    print(f"  {MODEL_FILE}")

    print()
    print(f"Metadata:")
    print(f"  {METADATA_FILE}")

    # =========================================================================
    # COMPLETE
    # =========================================================================

    total_time = time.time() - overall_start

    print()
    print("=" * 80)
    print("STEP 9 COMPLETE")
    print("=" * 80)

    print()
    print(f"Grouped ROC-AUC: {roc_auc:.6f}")
    print(f"Grouped F0.5:    {f05:.6f}")
    print(f"Threshold:       {best_threshold:.3f}")

    print()
    print(
        f"Total runtime: "
        f"{total_time:.2f}s"
    )

    print()
    print("NEXT:")
    print("  Compare grouped F0.5 with Step 8.")
    print("  Then proceed to hard-negative mining.")


if __name__ == "__main__":
    main()
