#!/usr/bin/env python3

"""
STEP 17 — RAM-SAFE MISSED TRUE-PAIR DIAGNOSTICS

Purpose:
    Find true pairs missed by Step 15 and analyze a 100,000-pair
    deterministic sample to discover what blocking patterns are
    needed for the next high-recall candidate generator.

Designed for low-RAM machines.
"""

from __future__ import annotations

import csv
import hashlib
import re
import time
from pathlib import Path

from rapidfuzz import fuzz


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"
CANDIDATE_FILE = OUTPUT_DIR / "step15_candidates.tsv"

MISSED_FILE = OUTPUT_DIR / "step17_missed_pairs.tsv"
SAMPLE_FILE = OUTPUT_DIR / "step17_missed_sample.tsv"
DETAIL_FILE = OUTPUT_DIR / "step17_feature_sample.tsv"
REPORT_FILE = OUTPUT_DIR / "step17_missed_pair_analysis.txt"


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 100_000


# ============================================================
# NORMALIZATION
# ============================================================

def clean_text(value: str) -> str:
    if not value:
        return ""

    value = value.lower()

    value = re.sub(
        r"[^\w\s]+",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def compact_text(value: str) -> str:
    return re.sub(
        r"[^\w]+",
        "",
        (value or "").lower(),
        flags=re.UNICODE,
    )


def tokens(value: str) -> set[str]:
    value = clean_text(value)

    if not value:
        return set()

    return {
        token
        for token in value.split()
        if len(token) >= 2
    }


def digits(value: str) -> set[str]:
    if not value:
        return set()

    return {
        x
        for x in re.findall(r"\d+", value)
        if x
    }


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b

    if not union:
        return 0.0

    return len(a & b) / len(union)


# ============================================================
# STEP 1 — FIND MISSED TRUE PAIRS
# ============================================================

def build_missed_pair_file():

    print("=" * 72)
    print("STEP 17 — FIND MISSED TRUE PAIRS")
    print("=" * 72)

    start = time.time()

    print()
    print("Scanning Step 15 candidate file...")

    candidate = CANDIDATE_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
    )

    candidate_line = candidate.readline()

    missed_count = 0
    gt_pairs = 0

    with GT_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as gt:

        reader = csv.DictReader(
            gt,
            delimiter="\t",
        )

        with MISSED_FILE.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as out:

            writer = csv.writer(
                out,
                delimiter="\t",
            )

            writer.writerow([
                "source1_entity_id",
                "matched_entity_id",
            ])

            for row in reader:

                s1 = row.get(
                    "source1_entity_id",
                    "",
                )

                matched = row.get(
                    "matched_entity_ids",
                    "",
                )

                if not s1 or not matched:
                    continue

                for target in matched.split(","):

                    target = target.strip()

                    if not target:
                        continue

                    gt_pairs += 1

                    target_pair = f"{s1}\t{target}"

                    while (
                        candidate_line
                        and candidate_line.rstrip("\n")
                        < target_pair
                    ):
                        candidate_line = candidate.readline()

                    if (
                        not candidate_line
                        or candidate_line.rstrip("\n")
                        != target_pair
                    ):

                        writer.writerow([
                            s1,
                            target,
                        ])

                        missed_count += 1

    candidate.close()

    elapsed = time.time() - start

    print()
    print(f"Ground-truth pairs: {gt_pairs:,}")
    print(f"Missed pairs:       {missed_count:,}")
    print(f"Saved:              {MISSED_FILE}")
    print(f"Time:               {elapsed:.2f} sec")

    return missed_count


# ============================================================
# STEP 2 — CREATE DETERMINISTIC SAMPLE
# ============================================================

def sample_missed_pairs():

    print()
    print("=" * 72)
    print("CREATING RAM-SAFE DIAGNOSTIC SAMPLE")
    print("=" * 72)

    start = time.time()

    selected = []

    with MISSED_FILE.open(
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

            s1 = row["source1_entity_id"]
            target = row["matched_entity_id"]

            key = f"{s1}|{target}".encode("utf-8")

            digest = hashlib.md5(key).digest()

            score = int.from_bytes(
                digest[:8],
                "big",
            )

            selected.append(
                (
                    score,
                    s1,
                    target,
                )
            )

    selected.sort(key=lambda x: x[0])

    selected = selected[:SAMPLE_SIZE]

    with SAMPLE_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow([
            "source1_entity_id",
            "matched_entity_id",
        ])

        for _, s1, target in selected:

            writer.writerow([
                s1,
                target,
            ])

    elapsed = time.time() - start

    print(
        f"Sample selected: {len(selected):,}"
    )

    print(
        f"Saved: {SAMPLE_FILE}"
    )

    print(
        f"Time: {elapsed:.2f} sec"
    )

    return [
        (s1, target)
        for _, s1, target in selected
    ]


# ============================================================
# STEP 3 — LOAD ONLY SAMPLE RECORDS
# ============================================================

def load_sample_records(pairs):

    print()
    print("=" * 72)
    print("LOADING ONLY SAMPLE RECORDS")
    print("=" * 72)

    s1_needed = {
        s1
        for s1, _ in pairs
    }

    target_needed = {
        target
        for _, target in pairs
    }

    s1_records = {}
    target_records = {}

    print(
        f"Required S1 records:     {len(s1_needed):,}"
    )

    print(
        f"Required target records: {len(target_needed):,}"
    )

    # --------------------------------------------------------
    # SOURCE 1
    # --------------------------------------------------------

    path = TRAIN_DIR / "train_source1.tsv"

    start = time.time()
    rows = 0

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

            rows += 1

            entity_id = row.get(
                "entity_id",
                "",
            )

            if entity_id in s1_needed:

                s1_records[entity_id] = {
                    "name": row.get(
                        "business_name"
                    ) or "",
                    "address": row.get(
                        "business_address"
                    ) or "",
                    "country": row.get(
                        "country"
                    ) or "",
                }

                if len(s1_records) == len(s1_needed):
                    break

    print(
        f"S1 loaded: {len(s1_records):,} "
        f"(scanned {rows:,}) "
        f"in {time.time() - start:.2f}s"
    )

    # --------------------------------------------------------
    # SOURCE 2
    # --------------------------------------------------------

    path = TRAIN_DIR / "train_source2.tsv"

    start = time.time()
    rows = 0
    s2_found = 0

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

            rows += 1

            entity_id = row.get(
                "entity_id",
                "",
            )

            if entity_id in target_needed:

                if entity_id not in target_records:

                    target_records[entity_id] = {
                        "name": row.get(
                            "business_name"
                        ) or "",
                        "address": row.get(
                            "business_address"
                        ) or "",
                        "country": row.get(
                            "country"
                        ) or "",
                        "source": "S2",
                    }

                    s2_found += 1

    print(
        f"S2 targets loaded: {s2_found:,} "
        f"(scanned {rows:,}) "
        f"in {time.time() - start:.2f}s"
    )

    # --------------------------------------------------------
    # SOURCE 3
    # --------------------------------------------------------

    path = TRAIN_DIR / "train_source3.tsv"

    start = time.time()
    rows = 0
    s3_found = 0

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

            rows += 1

            entity_id = row.get(
                "entity_id",
                "",
            )

            if entity_id in target_needed:

                if entity_id not in target_records:

                    target_records[entity_id] = {
                        "name": row.get(
                            "business_name"
                        ) or "",
                        "address": row.get(
                            "business_address"
                        ) or "",
                        "country": row.get(
                            "country"
                        ) or "",
                        "source": "S3",
                    }

                    s3_found += 1

    print(
        f"S3 targets loaded: {s3_found:,} "
        f"(scanned {rows:,}) "
        f"in {time.time() - start:.2f}s"
    )

    print(
        f"Total target records loaded: "
        f"{len(target_records):,}"
    )

    return s1_records, target_records


# ============================================================
# STEP 4 — FEATURE CALCULATION
# ============================================================

def calculate_features(s1, target):

    n1 = clean_text(
        s1["name"]
    )

    n2 = clean_text(
        target["name"]
    )

    a1 = clean_text(
        s1["address"]
    )

    a2 = clean_text(
        target["address"]
    )

    n1_tokens = tokens(
        s1["name"]
    )

    n2_tokens = tokens(
        target["name"]
    )

    a1_tokens = tokens(
        s1["address"]
    )

    a2_tokens = tokens(
        target["address"]
    )

    d1 = digits(
        s1["address"]
    )

    d2 = digits(
        target["address"]
    )

    name_ratio = (
        fuzz.ratio(n1, n2) / 100.0
    )

    name_sort = (
        fuzz.token_sort_ratio(n1, n2)
        / 100.0
    )

    name_set = (
        fuzz.token_set_ratio(n1, n2)
        / 100.0
    )

    name_partial = (
        fuzz.partial_ratio(n1, n2)
        / 100.0
    )

    address_ratio = (
        fuzz.ratio(a1, a2) / 100.0
    )

    address_sort = (
        fuzz.token_sort_ratio(a1, a2)
        / 100.0
    )

    address_set = (
        fuzz.token_set_ratio(a1, a2)
        / 100.0
    )

    address_partial = (
        fuzz.partial_ratio(a1, a2)
        / 100.0
    )

    digit_overlap = (
        len(d1 & d2)
        / max(1, len(d1 | d2))
    )

    name_jaccard = jaccard(
        n1_tokens,
        n2_tokens,
    )

    address_jaccard = jaccard(
        a1_tokens,
        a2_tokens,
    )

    return {
        "same_country": (
            s1["country"].strip().lower()
            ==
            target["country"].strip().lower()
        ),

        "name_exact": (
            bool(n1)
            and n1 == n2
        ),

        "name_compact_exact": (
            bool(compact_text(s1["name"]))
            and
            compact_text(s1["name"])
            ==
            compact_text(target["name"])
        ),

        "name_ratio": name_ratio,
        "name_sort": name_sort,
        "name_set": name_set,
        "name_partial": name_partial,
        "name_jaccard": name_jaccard,

        "address_exact": (
            bool(a1)
            and a1 == a2
        ),

        "address_compact_exact": (
            bool(compact_text(s1["address"]))
            and
            compact_text(s1["address"])
            ==
            compact_text(target["address"])
        ),

        "address_ratio": address_ratio,
        "address_sort": address_sort,
        "address_set": address_set,
        "address_partial": address_partial,
        "address_jaccard": address_jaccard,

        "digit_overlap": digit_overlap,

        "name_prefix": (
            len(n1) >= 6
            and len(n2) >= 6
            and n1[:6] == n2[:6]
        ),

        "name_suffix": (
            len(n1) >= 6
            and len(n2) >= 6
            and n1[-6:] == n2[-6:]
        ),

        "address_prefix": (
            len(a1) >= 8
            and len(a2) >= 8
            and a1[:8] == a2[:8]
        ),

        "address_suffix": (
            len(a1) >= 8
            and len(a2) >= 8
            and a1[-8:] == a2[-8:]
        ),
    }


# ============================================================
# STEP 5 — ANALYSIS
# ============================================================

def analyze(
    pairs,
    s1_records,
    target_records,
):

    print()
    print("=" * 72)
    print("ANALYZING MISSED TRUE MATCH SAMPLE")
    print("=" * 72)

    counters = {}

    def inc(name):
        counters[name] = (
            counters.get(name, 0) + 1
        )

    detail_rows = []

    valid = 0
    missing = 0

    for index, (s1_id, target_id) in enumerate(
        pairs,
        start=1,
    ):

        s1 = s1_records.get(s1_id)
        target = target_records.get(target_id)

        if not s1 or not target:
            missing += 1
            continue

        valid += 1

        f = calculate_features(
            s1,
            target,
        )

        # ----------------------------------------------------
        # BASIC
        # ----------------------------------------------------

        if f["same_country"]:
            inc("same_country")

        if f["name_exact"]:
            inc("name_exact")

        if f["name_compact_exact"]:
            inc("name_compact_exact")

        if f["address_exact"]:
            inc("address_exact")

        if f["address_compact_exact"]:
            inc("address_compact_exact")

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        thresholds = (
            0.95,
            0.90,
            0.85,
            0.80,
            0.70,
        )

        for threshold in thresholds:

            if f["name_ratio"] >= threshold:
                inc(
                    f"name_ratio_{threshold:.2f}"
                )

            if f["name_sort"] >= threshold:
                inc(
                    f"name_sort_{threshold:.2f}"
                )

            if f["name_set"] >= threshold:
                inc(
                    f"name_set_{threshold:.2f}"
                )

        # ----------------------------------------------------
        # ADDRESS
        # ----------------------------------------------------

        for threshold in thresholds:

            if f["address_ratio"] >= threshold:
                inc(
                    f"address_ratio_{threshold:.2f}"
                )

            if f["address_sort"] >= threshold:
                inc(
                    f"address_sort_{threshold:.2f}"
                )

            if f["address_set"] >= threshold:
                inc(
                    f"address_set_{threshold:.2f}"
                )

        # ----------------------------------------------------
        # TOKEN / DIGIT
        # ----------------------------------------------------

        if f["name_jaccard"] >= 0.50:
            inc("name_jaccard_050")

        if f["name_jaccard"] >= 0.75:
            inc("name_jaccard_075")

        if f["address_jaccard"] >= 0.50:
            inc("address_jaccard_050")

        if f["address_jaccard"] >= 0.75:
            inc("address_jaccard_075")

        if f["digit_overlap"] > 0:
            inc("digit_overlap")

        if f["digit_overlap"] >= 0.50:
            inc("digit_overlap_050")

        # ----------------------------------------------------
        # PREFIX / SUFFIX
        # ----------------------------------------------------

        if f["name_prefix"]:
            inc("name_prefix")

        if f["name_suffix"]:
            inc("name_suffix")

        if f["address_prefix"]:
            inc("address_prefix")

        if f["address_suffix"]:
            inc("address_suffix")

        # ----------------------------------------------------
        # STRONG COMBINATIONS
        # ----------------------------------------------------

        strong_name = (
            f["name_set"] >= 0.90
            or
            f["name_sort"] >= 0.90
        )

        strong_address = (
            f["address_set"] >= 0.85
            or
            f["address_sort"] >= 0.85
        )

        if strong_name:
            inc("strong_name")

        if strong_address:
            inc("strong_address")

        if strong_name and strong_address:
            inc("strong_name_and_address")

        if strong_name and not strong_address:
            inc("strong_name_only")

        if strong_address and not strong_name:
            inc("strong_address_only")

        if (
            f["name_set"] >= 0.80
            and
            f["address_set"] >= 0.70
        ):
            inc("name80_address70")

        if (
            f["name_set"] >= 0.85
            and
            f["address_set"] >= 0.75
        ):
            inc("name85_address75")

        if (
            f["name_set"] >= 0.70
            and
            f["address_set"] >= 0.85
        ):
            inc("name70_address85")

        # ----------------------------------------------------
        # DETAIL ROW
        # ----------------------------------------------------

        detail_rows.append({
            "s1_entity_id": s1_id,
            "target_entity_id": target_id,
            "target_source": target["source"],

            "same_country": int(
                f["same_country"]
            ),

            "name_ratio": round(
                f["name_ratio"], 4
            ),

            "name_sort": round(
                f["name_sort"], 4
            ),

            "name_set": round(
                f["name_set"], 4
            ),

            "name_partial": round(
                f["name_partial"], 4
            ),

            "name_jaccard": round(
                f["name_jaccard"], 4
            ),

            "address_ratio": round(
                f["address_ratio"], 4
            ),

            "address_sort": round(
                f["address_sort"], 4
            ),

            "address_set": round(
                f["address_set"], 4
            ),

            "address_partial": round(
                f["address_partial"], 4
            ),

            "address_jaccard": round(
                f["address_jaccard"], 4
            ),

            "digit_overlap": round(
                f["digit_overlap"], 4
            ),

            "name_exact": int(
                f["name_exact"]
            ),

            "name_compact_exact": int(
                f["name_compact_exact"]
            ),

            "address_exact": int(
                f["address_exact"]
            ),

            "address_compact_exact": int(
                f["address_compact_exact"]
            ),
        })

        if index % 20_000 == 0:

            print(
                f"  analyzed: "
                f"{index:,}/{len(pairs):,}",
                flush=True,
            )

    # ========================================================
    # SAVE DETAIL FILE
    # ========================================================

    if detail_rows:

        with DETAIL_FILE.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    detail_rows[0].keys()
                ),
                delimiter="\t",
            )

            writer.writeheader()
            writer.writerows(detail_rows)

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "STEP 17 — MISSED TRUE PAIR ANALYSIS"
    )

    report.append("=" * 72)

    report.append(
        f"Diagnostic sample: {len(pairs):,}"
    )

    report.append(
        f"Valid records:     {valid:,}"
    )

    report.append(
        f"Missing records:   {missing:,}"
    )

    report.append("")

    report.append(
        "FEATURE COVERAGE"
    )

    report.append("-" * 72)

    ordered = sorted(
        counters.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for name, count in ordered:

        pct = (
            count / valid * 100
            if valid
            else 0
        )

        report.append(
            f"{name:<32}"
            f"{count:>10,}"
            f"  ({pct:6.2f}%)"
        )

    REPORT_FILE.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("STEP 17 COMPLETE")
    print("=" * 72)

    print(
        f"Sample analyzed: {valid:,}"
    )

    print(
        f"Missing records: {missing:,}"
    )

    print()
    print(
        f"Details: {DETAIL_FILE}"
    )

    print(
        f"Report:  {REPORT_FILE}"
    )

    return counters


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    missed_count = build_missed_pair_file()

    if missed_count == 0:

        print(
            "No missed pairs. "
            "Candidate recall is complete."
        )

        return

    pairs = sample_missed_pairs()

    s1_records, target_records = (
        load_sample_records(pairs)
    )

    analyze(
        pairs,
        s1_records,
        target_records,
    )

    print()
    print(
        f"Total runtime: "
        f"{time.time() - overall_start:.2f} sec"
    )


if __name__ == "__main__":
    main()