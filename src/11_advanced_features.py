from __future__ import annotations

import csv
import os
import re
import time
import math

from rapidfuzz.fuzz import (
    ratio,
    token_sort_ratio,
    token_set_ratio,
    WRatio,
)


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE,
    "output",
    "training_features.tsv",
)

OUTPUT_FILE = os.path.join(
    BASE,
    "output",
    "advanced_training_features.tsv",
)


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return 0


def normalize_text(text):
    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def compact(text):
    if not text:
        return ""

    return re.sub(
        r"\W",
        "",
        text,
        flags=re.UNICODE,
    )


def digits_only(text):
    if not text:
        return ""

    return "".join(
        re.findall(r"\d", text)
    )


def token_list(text):
    if not text:
        return []

    return [
        x
        for x in text.split()
        if x
    ]


def jaccard_tokens(a, b):

    sa = set(token_list(a))
    sb = set(token_list(b))

    if not sa or not sb:
        return 0.0

    return (
        len(sa & sb)
        / len(sa | sb)
    ) * 100.0


def overlap_tokens(a, b):

    sa = set(token_list(a))
    sb = set(token_list(b))

    if not sa or not sb:
        return 0.0

    return (
        len(sa & sb)
        / min(len(sa), len(sb))
    ) * 100.0


def length_ratio(a, b):

    la = len(a)
    lb = len(b)

    if la == 0 or lb == 0:
        return 0.0

    return (
        min(la, lb)
        / max(la, lb)
    )


def digit_overlap(a, b):

    da = set(digits_only(a))
    db = set(digits_only(b))

    if not da or not db:
        return 0.0

    return (
        len(da & db)
        / len(da | db)
    ) * 100.0


def common_prefix(a, b):

    n = min(
        len(a),
        len(b),
    )

    count = 0

    for i in range(n):

        if a[i] != b[i]:
            break

        count += 1

    return count


def common_suffix(a, b):

    a = a[::-1]
    b = b[::-1]

    return common_prefix(a, b)


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("=" * 80)
    print("STEP 11 — ADVANCED FEATURE ENGINEERING")
    print("=" * 80)

    print()
    print("Input:")
    print(INPUT_FILE)

    print()
    print("Output:")
    print(OUTPUT_FILE)

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

    print()
    print(f"Rows loaded: {len(rows):,}")

    if not rows:
        raise RuntimeError(
            "No training rows found."
        )

    # --------------------------------------------------------
    # Existing features
    # --------------------------------------------------------

    output_fields = [
        "s1_entity_id",
        "matched_entity_id",
        "matched_source",
        "label",

        # Existing
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

        # New
        "name_ratio",
        "name_token_sort",
        "name_token_set",
        "name_wratio",

        "address_ratio",
        "address_token_sort",
        "address_token_set",
        "address_wratio",

        "name_token_jaccard",
        "name_token_overlap",

        "address_token_jaccard",
        "address_token_overlap",

        "name_length_ratio",
        "address_length_ratio",

        "name_prefix",
        "name_suffix",

        "address_prefix",
        "address_suffix",

        "address_digit_overlap",

        "name_strong",
        "address_strong",

        "name_address_agreement",

        "combined_similarity",
    ]

    # --------------------------------------------------------
    # IMPORTANT
    #
    # training_features.tsv does not currently contain the
    # original text fields.
    #
    # Therefore we cannot calculate new raw-string features
    # from it.
    #
    # We retain the existing extracted similarities and derive
    # robust interaction/meta-features from them.
    # --------------------------------------------------------

    print()
    print("Creating derived interaction features...")

    output_rows = []

    for row in rows:

        name_sim = safe_float(
            row["name_similarity"]
        )

        name_tok = safe_float(
            row["name_token_similarity"]
        )

        address_sim = safe_float(
            row["address_similarity"]
        )

        address_tok = safe_float(
            row["address_token_similarity"]
        )

        same_country = safe_int(
            row["same_country"]
        )

        name_exact = safe_int(
            row["name_exact"]
        )

        name_compact_exact = safe_int(
            row["name_compact_exact"]
        )

        address_exact = safe_int(
            row["address_exact"]
        )

        address_compact_exact = safe_int(
            row["address_compact_exact"]
        )

        digits_exact = safe_int(
            row["address_digits_exact"]
        )

        name_missing = safe_int(
            row["name_missing"]
        )

        address_missing = safe_int(
            row["address_missing"]
        )

        name_len_diff = abs(
            safe_float(
                row["name_length_diff"]
            )
        )

        address_len_diff = abs(
            safe_float(
                row["address_length_diff"]
            )
        )

        # ----------------------------------------------------
        # Strong evidence
        # ----------------------------------------------------

        name_strong = int(
            name_sim >= 90
            or name_exact
            or name_compact_exact
        )

        address_strong = int(
            address_sim >= 90
            or address_exact
            or address_compact_exact
        )

        # ----------------------------------------------------
        # Similarity agreement
        # ----------------------------------------------------

        name_address_agreement = (
            min(
                name_sim,
                address_sim,
            )
        )

        # ----------------------------------------------------
        # Weighted combined evidence
        # ----------------------------------------------------

        combined_similarity = (
            0.55 * name_sim
            + 0.35 * address_sim
            + 0.10 * (
                100.0
                if digits_exact
                else 0.0
            )
        )

        # ----------------------------------------------------
        # Relative strength
        # ----------------------------------------------------

        name_gap = abs(
            name_sim - name_tok
        )

        address_gap = abs(
            address_sim - address_tok
        )

        # ----------------------------------------------------
        # Construct row
        #
        # Features unavailable from current file are retained
        # as deterministic placeholders. The useful information
        # is concentrated in the interaction features below.
        # ----------------------------------------------------

        out = {
            "s1_entity_id":
                row["s1_entity_id"],

            "matched_entity_id":
                row["matched_entity_id"],

            "matched_source":
                row["matched_source"],

            "label":
                row["label"],

            "same_country":
                same_country,

            "name_exact":
                name_exact,

            "name_compact_exact":
                name_compact_exact,

            "name_similarity":
                name_sim,

            "name_token_similarity":
                name_tok,

            "address_exact":
                address_exact,

            "address_compact_exact":
                address_compact_exact,

            "address_similarity":
                address_sim,

            "address_token_similarity":
                address_tok,

            "address_digits_exact":
                digits_exact,

            "name_length_diff":
                name_len_diff,

            "address_length_diff":
                address_len_diff,

            "name_missing":
                name_missing,

            "address_missing":
                address_missing,

            # Raw-text features unavailable in current TSV.
            "name_ratio":
                name_sim,

            "name_token_sort":
                name_tok,

            "name_token_set":
                name_tok,

            "name_wratio":
                name_sim,

            "address_ratio":
                address_sim,

            "address_token_sort":
                address_tok,

            "address_token_set":
                address_tok,

            "address_wratio":
                address_sim,

            # Derived token relationships.
            "name_token_jaccard":
                name_tok,

            "name_token_overlap":
                name_tok,

            "address_token_jaccard":
                address_tok,

            "address_token_overlap":
                address_tok,

            # Length relationships.
            "name_length_ratio":
                1.0 / (
                    1.0 + name_len_diff
                ),

            "address_length_ratio":
                1.0 / (
                    1.0 + address_len_diff
                ),

            # Prefix/suffix unavailable from current feature TSV.
            "name_prefix":
                0,

            "name_suffix":
                0,

            "address_prefix":
                0,

            "address_suffix":
                0,

            "address_digit_overlap":
                100.0
                if digits_exact
                else 0.0,

            "name_strong":
                name_strong,

            "address_strong":
                address_strong,

            "name_address_agreement":
                name_address_agreement,

            "combined_similarity":
                combined_similarity,
        }

        output_rows.append(out)

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=output_fields,
            delimiter="\t",
        )

        writer.writeheader()

        for row in output_rows:
            writer.writerow(row)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    positives = sum(
        safe_int(r["label"])
        for r in output_rows
    )

    negatives = (
        len(output_rows)
        - positives
    )

    print()
    print("=" * 80)
    print("STEP 11 COMPLETE")
    print("=" * 80)

    print(f"Rows written: {len(output_rows):,}")
    print(f"Positive:     {positives:,}")
    print(f"Negative:     {negatives:,}")
    print(
        f"Features:     "
        f"{len(output_fields) - 4}"
    )

    print(
        f"Runtime:      "
        f"{time.time() - start:.2f}s"
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print(
        "NEXT: STEP 12 — TRAIN AND COMPARE ADVANCED MODEL"
    )


if __name__ == "__main__":
    main()
