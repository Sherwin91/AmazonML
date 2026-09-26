#!/usr/bin/env python3

"""
STEP 18A — BLOCKING KEY FREQUENCY ANALYSIS

Analyze candidate blocking-key frequencies on the full training
dataset WITHOUT generating pair candidates.

This tells us which blocking keys are useful and which keys
cause candidate explosions.

Keys analyzed:

    1. name prefix
    2. compact name
    3. name tokens
    4. address prefix
    5. address tokens
    6. address digits
    7. country + name prefix
    8. country + address prefix
    9. country + compact name

No candidate pairs are generated.
No large pair matrix is created.
"""

from __future__ import annotations

import csv
import re
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

REPORT_FILE = (
    OUTPUT_DIR /
    "step18_block_key_frequency_report.txt"
)


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


def name_prefix(value: str) -> str:

    value = clean_text(value)

    if not value:
        return ""

    # First 6 normalized characters.
    return value[:6]


def address_prefix(value: str) -> str:

    value = clean_text(value)

    if not value:
        return ""

    return value[:8]


def name_token_key(value: str) -> str:

    tokens = {
        token
        for token in clean_text(value).split()
        if len(token) >= 3
    }

    if not tokens:
        return ""

    return " ".join(
        sorted(tokens)
    )


def address_token_key(value: str) -> str:

    tokens = {
        token
        for token in clean_text(value).split()
        if len(token) >= 3
    }

    if not tokens:
        return ""

    return " ".join(
        sorted(tokens)
    )


def address_digit_key(value: str) -> str:

    if not value:
        return ""

    nums = re.findall(
        r"\d+",
        value,
    )

    if not nums:
        return ""

    return " ".join(
        sorted(set(nums))
    )


# ============================================================
# STATISTICS
# ============================================================

def summarize(
    counter: Counter,
    name: str,
):

    if not counter:

        print(
            f"\n{name}: EMPTY"
        )

        return []

    values = sorted(
        counter.values(),
        reverse=True,
    )

    total_occurrences = sum(values)

    unique_keys = len(values)

    blocks_1 = sum(
        1 for x in values
        if x == 1
    )

    blocks_5 = sum(
        1 for x in values
        if x <= 5
    )

    blocks_10 = sum(
        1 for x in values
        if x <= 10
    )

    blocks_25 = sum(
        1 for x in values
        if x <= 25
    )

    blocks_50 = sum(
        1 for x in values
        if x <= 50
    )

    blocks_100 = sum(
        1 for x in values
        if x <= 100
    )

    blocks_250 = sum(
        1 for x in values
        if x <= 250
    )

    blocks_500 = sum(
        1 for x in values
        if x <= 500
    )

    blocks_1000 = sum(
        1 for x in values
        if x <= 1000
    )

    max_block = max(values)

    # Candidate pairs produced by each block:
    # n * (n - 1) / 2
    candidate_pairs = sum(
        n * (n - 1) // 2
        for n in values
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

    print(
        f"Unique keys:             {unique_keys:,}"
    )

    print(
        f"Total occurrences:       {total_occurrences:,}"
    )

    print(
        f"Singleton keys:           {blocks_1:,}"
    )

    print(
        f"Keys <= 5:                {blocks_5:,}"
    )

    print(
        f"Keys <= 10:               {blocks_10:,}"
    )

    print(
        f"Keys <= 25:               {blocks_25:,}"
    )

    print(
        f"Keys <= 50:               {blocks_50:,}"
    )

    print(
        f"Keys <= 100:              {blocks_100:,}"
    )

    print(
        f"Keys <= 250:              {blocks_250:,}"
    )

    print(
        f"Keys <= 500:              {blocks_500:,}"
    )

    print(
        f"Keys <= 1000:             {blocks_1000:,}"
    )

    print(
        f"Maximum block:            {max_block:,}"
    )

    print(
        f"Raw same-key pairs:       {candidate_pairs:,}"
    )

    print()
    print("Largest blocks:")

    for rank, (key, count) in enumerate(
        counter.most_common(20),
        start=1,
    ):

        print(
            f"{rank:2d}. "
            f"{count:>8,}  "
            f"{key[:80]}"
        )

    return [
        (
            unique_keys,
            total_occurrences,
            candidate_pairs,
            max_block,
        )
    ]


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(
    path: Path,
    source_name: str,
):

    print()
    print(
        f"Processing {source_name}: "
        f"{path.name}"
    )

    start = time.time()

    counters = {
        "name_prefix": Counter(),
        "compact_name": Counter(),
        "name_tokens": Counter(),
        "address_prefix": Counter(),
        "address_tokens": Counter(),
        "address_digits": Counter(),
        "country_name_prefix": Counter(),
        "country_address_prefix": Counter(),
        "country_compact_name": Counter(),
    }

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

            name = row.get(
                "business_name"
            ) or ""

            address = row.get(
                "business_address"
            ) or ""

            country = (
                row.get("country")
                or ""
            ).strip().lower()

            np = name_prefix(name)
            cn = compact_text(name)
            nt = name_token_key(name)

            ap = address_prefix(address)
            at = address_token_key(address)
            ad = address_digit_key(address)

            if np:
                counters[
                    "name_prefix"
                ][np] += 1

                if country:
                    counters[
                        "country_name_prefix"
                    ][
                        f"{country}|{np}"
                    ] += 1

            if cn:
                counters[
                    "compact_name"
                ][cn] += 1

                if country:
                    counters[
                        "country_compact_name"
                    ][
                        f"{country}|{cn}"
                    ] += 1

            if nt:
                counters[
                    "name_tokens"
                ][nt] += 1

            if ap:
                counters[
                    "address_prefix"
                ][ap] += 1

                if country:
                    counters[
                        "country_address_prefix"
                    ][
                        f"{country}|{ap}"
                    ] += 1

            if at:
                counters[
                    "address_tokens"
                ][at] += 1

            if ad:
                counters[
                    "address_digits"
                ][ad] += 1

            if rows % 1_000_000 == 0:

                print(
                    f"  rows: {rows:,}",
                    flush=True,
                )

    print(
        f"Rows processed: {rows:,}"
    )

    print(
        f"Time: {time.time() - start:.2f}s"
    )

    return counters


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    all_counters = {}

    sources = [
        (
            "S1",
            TRAIN_DIR /
            "train_source1.tsv",
        ),
        (
            "S2",
            TRAIN_DIR /
            "train_source2.tsv",
        ),
        (
            "S3",
            TRAIN_DIR /
            "train_source3.tsv",
        ),
    ]

    for source_name, path in sources:

        all_counters[source_name] = (
            process_source(
                path,
                source_name,
            )
        )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "STEP 18A — BLOCKING KEY FREQUENCY ANALYSIS"
    )

    report.append("=" * 72)

    report.append("")

    report.append(
        "This report measures block sizes before "
        "candidate generation."
    )

    report.append("")

    for source_name in (
        "S1",
        "S2",
        "S3",
    ):

        report.append("")
        report.append(
            "#" * 72
        )

        report.append(
            f"{source_name}"
        )

        report.append(
            "#" * 72
        )

        for key_name, counter in (
            all_counters[
                source_name
            ].items()
        ):

            if not counter:
                continue

            values = sorted(
                counter.values(),
                reverse=True,
            )

            total_occurrences = sum(
                values
            )

            unique_keys = len(values)

            candidate_pairs = sum(
                n * (n - 1) // 2
                for n in values
            )

            max_block = max(values)

            report.append("")
            report.append(
                f"[{key_name}]"
            )

            report.append(
                f"unique_keys={unique_keys:,}"
            )

            report.append(
                f"occurrences={total_occurrences:,}"
            )

            report.append(
                f"max_block={max_block:,}"
            )

            report.append(
                f"raw_pairs={candidate_pairs:,}"
            )

            for limit in (
                10,
                25,
                50,
                100,
                250,
                500,
                1000,
                1500,
            ):

                count = sum(
                    1
                    for x in values
                    if x <= limit
                )

                report.append(
                    f"keys<= {limit:4d}: "
                    f"{count:,}"
                )

            report.append(
                "largest:"
            )

            for key, count in counter.most_common(
                10
            ):

                report.append(
                    f"  {count:>8,} "
                    f"{key[:80]}"
                )

    elapsed = (
        time.time()
        - overall_start
    )

    report.append("")
    report.append(
        f"Total runtime: {elapsed:.2f}s"
    )

    REPORT_FILE.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("STEP 18A COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Report: {REPORT_FILE}"
    )

    print(
        f"Total runtime: {elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
