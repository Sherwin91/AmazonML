#!/usr/bin/env python3

from __future__ import annotations

import csv
import time
from pathlib import Path

from rapidfuzz import fuzz


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

SAMPLE_FILE = OUTPUT_DIR / "step18_missed_sample.tsv"

FEATURE_FILE = OUTPUT_DIR / "step19_actual_missed_features.tsv"
REPORT_FILE = OUTPUT_DIR / "step19_actual_missed_analysis.txt"


# ============================================================
# NORMALIZATION
# ============================================================

def clean_text(value):
    if not value:
        return ""

    value = value.lower()

    out = []

    for ch in value:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
        else:
            out.append(" ")

    text = "".join(out)

    return " ".join(text.split())


def compact(value):
    if not value:
        return ""

    return "".join(
        ch for ch in value.lower()
        if ch.isalnum()
    )


def tokens(value):
    return {
        x
        for x in clean_text(value).split()
        if len(x) >= 3
    }


def digits(value):
    result = set()

    if not value:
        return result

    current = ""

    for ch in value:

        if ch.isdigit():
            current += ch

        else:

            if current:
                result.add(current)
                current = ""

    if current:
        result.add(current)

    return result


def token_jaccard(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def token_overlap(a, b):

    if not a or not b:
        return 0.0

    return len(a & b) / min(
        len(a),
        len(b),
    )


def char_jaccard(a, b):

    x = set(a)
    y = set(b)

    if not x and not y:
        return 1.0

    if not x or not y:
        return 0.0

    return len(x & y) / len(x | y)


# ============================================================
# FEATURES
# ============================================================

def compute_features(
    s1_name,
    s1_addr,
    s2_name,
    s2_addr,
):

    n1 = clean_text(s1_name)
    n2 = clean_text(s2_name)

    a1 = clean_text(s1_addr)
    a2 = clean_text(s2_addr)

    n1c = compact(n1)
    n2c = compact(n2)

    a1c = compact(a1)
    a2c = compact(a2)

    nt1 = tokens(n1)
    nt2 = tokens(n2)

    at1 = tokens(a1)
    at2 = tokens(a2)

    d1 = digits(a1)
    d2 = digits(a2)

    name_ratio = fuzz.ratio(
        n1,
        n2,
    ) / 100.0

    name_sort = fuzz.token_sort_ratio(
        n1,
        n2,
    ) / 100.0

    name_set = fuzz.token_set_ratio(
        n1,
        n2,
    ) / 100.0

    name_partial = fuzz.partial_ratio(
        n1,
        n2,
    ) / 100.0

    name_wratio = fuzz.WRatio(
        n1,
        n2,
    ) / 100.0

    addr_ratio = fuzz.ratio(
        a1,
        a2,
    ) / 100.0 if a1 and a2 else 0.0

    addr_sort = fuzz.token_sort_ratio(
        a1,
        a2,
    ) / 100.0 if a1 and a2 else 0.0

    addr_set = fuzz.token_set_ratio(
        a1,
        a2,
    ) / 100.0 if a1 and a2 else 0.0

    addr_partial = fuzz.partial_ratio(
        a1,
        a2,
    ) / 100.0 if a1 and a2 else 0.0

    addr_wratio = fuzz.WRatio(
        a1,
        a2,
    ) / 100.0 if a1 and a2 else 0.0

    digit_overlap = (
        len(d1 & d2) /
        max(1, len(d1 | d2))
    )

    return {
        "same_country": 1,

        "name_exact":
            int(n1 == n2 and bool(n1)),

        "name_compact_exact":
            int(
                n1c == n2c
                and bool(n1c)
            ),

        "name_ratio":
            name_ratio,

        "name_token_sort":
            name_sort,

        "name_token_set":
            name_set,

        "name_wratio":
            name_wratio,

        "name_partial":
            name_partial,

        "address_exact":
            int(
                a1 == a2
                and bool(a1)
                and bool(a2)
            ),

        "address_compact_exact":
            int(
                a1c == a2c
                and bool(a1c)
                and bool(a2c)
            ),

        "address_ratio":
            addr_ratio,

        "address_token_sort":
            addr_sort,

        "address_token_set":
            addr_set,

        "address_wratio":
            addr_wratio,

        "address_partial":
            addr_partial,

        "name_token_jaccard":
            token_jaccard(
                nt1,
                nt2,
            ),

        "name_token_overlap":
            token_overlap(
                nt1,
                nt2,
            ),

        "address_token_jaccard":
            token_jaccard(
                at1,
                at2,
            ),

        "address_token_overlap":
            token_overlap(
                at1,
                at2,
            ),

        "name_char_jaccard":
            char_jaccard(
                n1,
                n2,
            ),

        "address_char_jaccard":
            char_jaccard(
                a1,
                a2,
            ),

        "name_length_diff":
            abs(len(n1) - len(n2)),

        "address_length_diff":
            abs(len(a1) - len(a2)),

        "name_length_ratio":
            min(
                len(n1),
                len(n2),
            ) / max(
                1,
                max(
                    len(n1),
                    len(n2),
                ),
            ),

        "address_length_ratio":
            min(
                len(a1),
                len(a2),
            ) / max(
                1,
                max(
                    len(a1),
                    len(a2),
                ),
            ),

        "name_prefix":
            int(
                bool(n1)
                and bool(n2)
                and (
                    n1[:6] == n2[:6]
                )
            ),

        "name_suffix":
            int(
                bool(n1)
                and bool(n2)
                and (
                    n1[-6:] == n2[-6:]
                )
            ),

        "address_prefix":
            int(
                bool(a1)
                and bool(a2)
                and (
                    a1[:8] == a2[:8]
                )
            ),

        "address_suffix":
            int(
                bool(a1)
                and bool(a2)
                and (
                    a1[-8:] == a2[-8:]
                )
            ),

        "address_digit_overlap":
            digit_overlap,

        "name_missing":
            int(not bool(n2)),

        "address_missing":
            int(not bool(a2)),

        "name_strong":
            int(
                name_set >= 0.85
                or name_sort >= 0.85
            ),

        "address_strong":
            int(
                addr_set >= 0.85
                or addr_sort >= 0.85
            ),

        "name_address_agreement":
            (
                name_set *
                addr_set
            ),

        "combined_similarity":
            (
                0.6 * name_set +
                0.4 * addr_set
            ),

        "exact_name_and_country":
            int(
                n1 == n2
                and bool(n1)
            ),

        "exact_address_and_country":
            int(
                a1 == a2
                and bool(a1)
                and bool(a2)
            ),

        "strong_name_and_address":
            int(
                name_set >= 0.80
                and addr_set >= 0.70
            ),

        "name_address_max":
            max(
                name_set,
                addr_set,
            ),

        "name_address_min":
            min(
                name_set,
                addr_set,
            ),
    }


# ============================================================
# LOAD SAMPLE PAIRS
# ============================================================

def load_pairs():

    pairs = []

    s1_ids = set()
    target_ids = set()

    with SAMPLE_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split(
                "\t"
            )

            if len(parts) < 2:
                continue

            s1 = parts[0]
            target = parts[1]

            if target.startswith(
                "S2-"
            ):
                source = "S2"

            elif target.startswith(
                "S3-"
            ):
                source = "S3"

            else:
                continue

            pairs.append(
                (
                    s1,
                    target,
                    source,
                )
            )

            s1_ids.add(s1)
            target_ids.add(target)

    print(
        f"Sample pairs: {len(pairs):,}"
    )

    print(
        f"Required S1: {len(s1_ids):,}"
    )

    print(
        f"Required targets: {len(target_ids):,}"
    )

    return (
        pairs,
        s1_ids,
        target_ids,
    )


# ============================================================
# STREAM RECORDS
# ============================================================

def load_records(
    path,
    required_ids,
):

    records = {}

    scanned = 0

    start = time.time()

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            scanned += 1

            entity_id = (
                row.get(
                    "entity_id"
                )
                or ""
            ).strip()

            if entity_id in required_ids:

                records[entity_id] = (
                    row.get(
                        "business_name"
                    ) or "",
                    row.get(
                        "business_address"
                    ) or "",
                    row.get(
                        "country"
                    ) or "",
                )

            if scanned % 1_000_000 == 0:

                print(
                    f"  scanned: "
                    f"{scanned:,} | "
                    f"found: "
                    f"{len(records):,}",
                    flush=True,
                )

    print(
        f"Finished {path.name}: "
        f"scanned={scanned:,}, "
        f"found={len(records):,}, "
        f"time={time.time()-start:.2f}s"
    )

    return records


# ============================================================
# ANALYSIS
# ============================================================

def main():

    start = time.time()

    print("=" * 72)
    print("STEP 19 — ACTUAL MISSED-PAIR ANALYSIS")
    print("=" * 72)

    if not SAMPLE_FILE.exists():

        raise FileNotFoundError(
            f"Missing:\n{SAMPLE_FILE}"
        )

    pairs, s1_ids, target_ids = (
        load_pairs()
    )

    s1_records = load_records(
        TRAIN_DIR /
        "train_source1.tsv",
        s1_ids,
    )

    s2_ids = {
        x
        for x in target_ids
        if x.startswith("S2-")
    }

    s3_ids = {
        x
        for x in target_ids
        if x.startswith("S3-")
    }

    s2_records = load_records(
        TRAIN_DIR /
        "train_source2.tsv",
        s2_ids,
    )

    s3_records = load_records(
        TRAIN_DIR /
        "train_source3.tsv",
        s3_ids,
    )

    fields = None

    counts = {}

    numeric_values = {}

    valid = 0

    with FEATURE_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as out:

        writer = None

        for index, (
            s1_id,
            target_id,
            source,
        ) in enumerate(pairs, 1):

            s1 = s1_records.get(
                s1_id
            )

            if source == "S2":

                target = s2_records.get(
                    target_id
                )

            else:

                target = s3_records.get(
                    target_id
                )

            if s1 is None or target is None:
                continue

            features = compute_features(
                s1[0],
                s1[1],
                target[0],
                target[1],
            )

            if fields is None:

                fields = list(
                    features.keys()
                )

                writer = csv.writer(
                    out,
                    delimiter="\t",
                    lineterminator="\n",
                )

                writer.writerow(
                    [
                        "s1_entity_id",
                        "matched_entity_id",
                        "source",
                    ]
                    + fields
                )

                for field in fields:

                    counts[field] = 0
                    numeric_values[field] = []

            row = [
                s1_id,
                target_id,
                source,
            ]

            for field in fields:

                value = features[field]

                row.append(value)

                if isinstance(
                    value,
                    (int, float),
                ):

                    numeric_values[
                        field
                    ].append(
                        float(value)
                    )

            writer.writerow(row)

            valid += 1

            for field in fields:

                value = features[field]

                if (
                    field.endswith(
                        "missing"
                    )
                ):

                    threshold_value = (
                        value == 1
                    )

                else:

                    threshold_value = (
                        value >= 0.70
                    )

                if threshold_value:

                    counts[field] += 1

            if (
                valid % 20_000
                == 0
            ):

                print(
                    f"  analyzed: "
                    f"{valid:,}/{len(pairs):,}",
                    flush=True,
                )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "STEP 19 — ACTUAL MISSED-PAIR ANALYSIS"
    )

    report.append("=" * 72)

    report.append(
        f"Diagnostic sample: {len(pairs):,}"
    )

    report.append(
        f"Valid records:     {valid:,}"
    )

    report.append("")

    report.append(
        "FEATURE COVERAGE"
    )

    report.append(
        "-" * 72
    )

    ranked = sorted(
        counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for field, count in ranked:

        percentage = (
            100.0 * count /
            max(1, valid)
        )

        report.append(
            f"{field:<32}"
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    report.append("")

    report.append(
        "KEY BLOCKING SIGNALS"
    )

    report.append(
        "-" * 72
    )

    key_rules = {
        "name_prefix":
            lambda f:
            f["name_prefix"] >= 1,

        "name_compact_exact":
            lambda f:
            f["name_compact_exact"] >= 1,

        "name_set_0.70":
            lambda f:
            f["name_token_set"] >= 0.70,

        "name_set_0.80":
            lambda f:
            f["name_token_set"] >= 0.80,

        "name_set_0.90":
            lambda f:
            f["name_token_set"] >= 0.90,

        "address_set_0.70":
            lambda f:
            f["address_token_set"] >= 0.70,

        "address_set_0.80":
            lambda f:
            f["address_token_set"] >= 0.80,

        "address_set_0.90":
            lambda f:
            f["address_token_set"] >= 0.90,

        "digit_overlap_0.50":
            lambda f:
            f["address_digit_overlap"] >= 0.50,

        "strong_name_address":
            lambda f:
            (
                f["name_token_set"] >= 0.80
                and
                f["address_token_set"] >= 0.70
            ),

        "name70_address85":
            lambda f:
            (
                f["name_token_set"] >= 0.70
                and
                f["address_token_set"] >= 0.85
            ),

        "name85_address70":
            lambda f:
            (
                f["name_token_set"] >= 0.85
                and
                f["address_token_set"] >= 0.70
            ),
    }

    # Re-read the generated feature file to calculate
    # combined rules without storing the full dataset.

    rule_counts = {
        key: 0
        for key in key_rules
    }

    with FEATURE_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            feature_row = {
                key: float(
                    row[key]
                )
                for key in row
                if key not in (
                    "s1_entity_id",
                    "matched_entity_id",
                    "source",
                )
            }

            for key, rule in key_rules.items():

                if rule(feature_row):

                    rule_counts[key] += 1

    report.append("")

    for key, count in sorted(
        rule_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    ):

        percentage = (
            100.0 * count /
            max(1, valid)
        )

        report.append(
            f"{key:<32}"
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    REPORT_FILE.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("STEP 19 COMPLETE")
    print("=" * 72)

    print(
        f"Valid analyzed: "
        f"{valid:,}"
    )

    print(
        f"Feature file: "
        f"{FEATURE_FILE}"
    )

    print(
        f"Report: "
        f"{REPORT_FILE}"
    )

    print(
        f"Runtime: "
        f"{time.time() - start:.2f}s"
    )


if __name__ == "__main__":
    main()
