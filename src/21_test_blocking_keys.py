#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

MISSED_SAMPLE = OUTPUT_DIR / "step18_missed_sample.tsv"
REPORT = OUTPUT_DIR / "step21_blocking_key_report.txt"


# ============================================================
# NORMALIZATION
# ============================================================

def clean_text(value: str) -> str:
    if not value:
        return ""

    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]", " ", value)

    return " ".join(value.split())


def compact(value: str) -> str:
    if not value:
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        value.lower(),
    )


def tokens(value: str):
    text = clean_text(value)

    return [
        token
        for token in text.split()
        if len(token) >= 3
    ]


def digit_string(value: str) -> str:
    if not value:
        return ""

    return "".join(
        ch
        for ch in value
        if ch.isdigit()
    )


# ============================================================
# BLOCKING KEYS
# ============================================================

def make_keys(
    name: str,
    address: str,
    country: str,
):

    name_clean = clean_text(name)
    address_clean = clean_text(address)

    name_tokens = tokens(name)
    address_tokens = tokens(address)

    keys = {}

    # --------------------------------------------------------
    # NAME CHARACTER PREFIX
    # --------------------------------------------------------

    for n in (3, 4, 5, 6, 7):

        if len(name_clean) >= n:

            keys[f"name_prefix_{n}"] = (
                country,
                name_clean[:n],
            )

    # --------------------------------------------------------
    # COMPACT NAME PREFIX
    # --------------------------------------------------------

    name_compact = compact(name)

    for n in (4, 5, 6, 7):

        if len(name_compact) >= n:

            keys[f"name_compact_prefix_{n}"] = (
                country,
                name_compact[:n],
            )

    # --------------------------------------------------------
    # FIRST NAME TOKEN PREFIX
    # --------------------------------------------------------

    if name_tokens:

        first = name_tokens[0]

        for n in (3, 4, 5, 6):

            if len(first) >= n:

                keys[f"first_name_token_{n}"] = (
                    country,
                    first[:n],
                )

    # --------------------------------------------------------
    # LAST NAME TOKEN PREFIX
    # --------------------------------------------------------

    if len(name_tokens) >= 2:

        last = name_tokens[-1]

        for n in (3, 4, 5, 6):

            if len(last) >= n:

                keys[f"last_name_token_{n}"] = (
                    country,
                    last[:n],
                )

    # --------------------------------------------------------
    # FIRST TWO NAME TOKENS
    # --------------------------------------------------------

    if len(name_tokens) >= 2:

        first_two = sorted(
            name_tokens[:2]
        )

        keys["name_first2_sorted"] = (
            country,
            "|".join(first_two),
        )

    # --------------------------------------------------------
    # LONGEST NAME TOKEN PREFIX
    # --------------------------------------------------------

    if name_tokens:

        longest = max(
            name_tokens,
            key=len,
        )

        for n in (4, 5, 6):

            if len(longest) >= n:

                keys[f"longest_name_token_{n}"] = (
                    country,
                    longest[:n],
                )

    # --------------------------------------------------------
    # ADDRESS CHARACTER PREFIX
    # --------------------------------------------------------

    for n in (4, 5, 6, 7, 8, 10):

        if len(address_clean) >= n:

            keys[f"address_prefix_{n}"] = (
                country,
                address_clean[:n],
            )

    # --------------------------------------------------------
    # COMPACT ADDRESS PREFIX
    # --------------------------------------------------------

    address_compact = compact(address)

    for n in (5, 6, 7, 8):

        if len(address_compact) >= n:

            keys[f"address_compact_prefix_{n}"] = (
                country,
                address_compact[:n],
            )

    # --------------------------------------------------------
    # FIRST ADDRESS TOKEN
    # --------------------------------------------------------

    if address_tokens:

        first = address_tokens[0]

        for n in (3, 4, 5, 6):

            if len(first) >= n:

                keys[f"first_address_token_{n}"] = (
                    country,
                    first[:n],
                )

    # --------------------------------------------------------
    # LONGEST ADDRESS TOKEN
    # --------------------------------------------------------

    if address_tokens:

        longest = max(
            address_tokens,
            key=len,
        )

        for n in (4, 5, 6):

            if len(longest) >= n:

                keys[f"longest_address_token_{n}"] = (
                    country,
                    longest[:n],
                )

    # --------------------------------------------------------
    # ADDRESS DIGITS
    # --------------------------------------------------------

    digits = digit_string(address)

    if len(digits) >= 2:

        keys["address_digits_2"] = (
            country,
            digits[:2],
        )

    if len(digits) >= 3:

        keys["address_digits_3"] = (
            country,
            digits[:3],
        )

    if len(digits) >= 4:

        keys["address_digits_4"] = (
            country,
            digits[:4],
        )

    # --------------------------------------------------------
    # NAME + ADDRESS COMBINATIONS
    # --------------------------------------------------------

    if len(name_clean) >= 4 and len(address_clean) >= 4:

        keys["name4_address4"] = (
            country,
            name_clean[:4],
            address_clean[:4],
        )

    if len(name_clean) >= 5 and len(address_clean) >= 5:

        keys["name5_address5"] = (
            country,
            name_clean[:5],
            address_clean[:5],
        )

    if name_tokens and address_tokens:

        keys["first_name_first_address"] = (
            country,
            name_tokens[0][:4],
            address_tokens[0][:4],
        )

    return keys


# ============================================================
# LOAD MISSED PAIRS
# ============================================================

def load_missed_pairs():

    pairs = []

    with MISSED_SAMPLE.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        reader = csv.reader(
            f,
            delimiter="\t",
        )

        for row in reader:

            if len(row) < 2:
                continue

            s1_id = row[0].strip()
            target_id = row[1].strip()

            if not s1_id or not target_id:
                continue

            pairs.append(
                (
                    s1_id,
                    target_id,
                )
            )

    return pairs


# ============================================================
# LOAD ONLY REQUIRED RECORDS
# ============================================================

def load_required_records(pairs):

    required_s1 = {
        s1
        for s1, _ in pairs
    }

    required_s2 = {
        target
        for _, target in pairs
        if target.startswith("S2-")
    }

    required_s3 = {
        target
        for _, target in pairs
        if target.startswith("S3-")
    }

    records = {}

    sources = [
        (
            "S1",
            TRAIN_DIR / "train_source1.tsv",
            required_s1,
        ),
        (
            "S2",
            TRAIN_DIR / "train_source2.tsv",
            required_s2,
        ),
        (
            "S3",
            TRAIN_DIR / "train_source3.tsv",
            required_s3,
        ),
    ]

    for source, path, required in sources:

        print()
        print("=" * 72)
        print(f"SCANNING {source}")
        print("=" * 72)

        start = time.time()

        scanned = 0
        found = 0

        if not required:
            print("Nothing required.")
            continue

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
                    row.get("entity_id")
                    or ""
                ).strip()

                if entity_id not in required:
                    continue

                records[entity_id] = {
                    "name": (
                        row.get("business_name")
                        or ""
                    ),
                    "address": (
                        row.get("business_address")
                        or ""
                    ),
                    "country": (
                        row.get("country")
                        or ""
                    ).strip().lower(),
                }

                found += 1

                if found == len(required):
                    break

        elapsed = time.time() - start

        print(
            f"Scanned: {scanned:,}"
        )

        print(
            f"Found: {found:,}/{len(required):,}"
        )

        print(
            f"Time: {elapsed:.2f}s"
        )

    return records


# ============================================================
# ANALYZE
# ============================================================

def analyze_pairs(
    pairs,
    records,
):

    coverage = defaultdict(int)

    valid = 0

    total = len(pairs)

    for i, (s1_id, target_id) in enumerate(
        pairs,
        start=1,
    ):

        s1 = records.get(s1_id)
        target = records.get(target_id)

        if s1 is None or target is None:
            continue

        valid += 1

        s1_keys = make_keys(
            s1["name"],
            s1["address"],
            s1["country"],
        )

        target_keys = make_keys(
            target["name"],
            target["address"],
            target["country"],
        )

        for key_name, key_value in s1_keys.items():

            if (
                key_name in target_keys
                and key_value == target_keys[key_name]
            ):

                coverage[key_name] += 1

        if i % 10_000 == 0:

            print(
                f"Analyzed "
                f"{i:,}/{total:,}",
                flush=True,
            )

    return valid, coverage


# ============================================================
# REPORT
# ============================================================

def write_report(
    pairs_count,
    valid,
    coverage,
    runtime,
):

    results = []

    for key, count in coverage.items():

        percentage = (
            count /
            max(1, valid)
            * 100.0
        )

        results.append(
            (
                percentage,
                count,
                key,
            )
        )

    results.sort(
        key=lambda x: (
            -x[0],
            -x[1],
            x[2],
        )
    )

    lines = []

    lines.append(
        "STEP 21 — BLOCKING KEY DIAGNOSTIC"
    )

    lines.append(
        "=" * 72
    )

    lines.append(
        f"Missed-pair sample: {pairs_count:,}"
    )

    lines.append(
        f"Valid pairs:         {valid:,}"
    )

    lines.append("")

    lines.append(
        "ALL BLOCKING KEYS"
    )

    lines.append(
        "-" * 72
    )

    for percentage, count, key in results:

        lines.append(
            f"{key:<36}"
            f"{count:>9,} "
            f"({percentage:>6.2f}%)"
        )

    lines.append("")

    lines.append(
        "TOP 15"
    )

    lines.append(
        "-" * 72
    )

    for percentage, count, key in results[:15]:

        lines.append(
            f"{key:<36}"
            f"{count:>9,} "
            f"({percentage:>6.2f}%)"
        )

    lines.append("")

    lines.append(
        f"Runtime: {runtime:.2f}s"
    )

    REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    print("=" * 72)
    print(
        "STEP 21 — BLOCKING KEY DIAGNOSTIC"
    )
    print("=" * 72)

    if not MISSED_SAMPLE.exists():

        raise FileNotFoundError(
            f"Missing file:\n{MISSED_SAMPLE}"
        )

    print()
    print(
        "Loading actual missed pairs..."
    )

    pairs = load_missed_pairs()

    print(
        f"Loaded: {len(pairs):,}"
    )

    print()
    print(
        "Loading required records..."
    )

    records = load_required_records(
        pairs
    )

    print()
    print(
        "Testing blocking keys..."
    )

    valid, coverage = analyze_pairs(
        pairs,
        records,
    )

    runtime = time.time() - start

    results = write_report(
        len(pairs),
        valid,
        coverage,
        runtime,
    )

    print()
    print("=" * 72)
    print("STEP 21 COMPLETE")
    print("=" * 72)

    print(
        f"Valid pairs: {valid:,}"
    )

    print()

    print(
        "TOP BLOCKING KEYS:"
    )

    for percentage, count, key in results[:15]:

        print(
            f"{key:<36}"
            f"{count:>9,} "
            f"({percentage:>6.2f}%)"
        )

    print()
    print(
        f"Report: {REPORT}"
    )

    print(
        f"Runtime: {runtime:.2f}s"
    )


if __name__ == "__main__":
    main()