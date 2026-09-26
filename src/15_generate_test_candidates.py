#!/usr/bin/env python3

"""
STEP 15
Disk-backed blocking recall benchmark.

Purpose:
    Measure how many real TRAINING matches are recovered by our
    candidate-generation / blocking strategy.

Designed for low-RAM machines:
    - Does NOT build a giant Python dictionary for S2/S3.
    - Uses GNU sort for disk-backed sorting.
    - Processes one blocking strategy at a time.
    - Removes temporary files after each strategy.

Blocking strategies:
    1. country + normalized name
    2. country + compact name
    3. country + token-sorted name
    4. country + normalized address
    5. country + compact address
    6. country + address digits

Output:
    output/step15_blocking_recall_report.txt
    output/step15_candidates_raw.tsv
    output/step15_candidates.tsv
"""

import csv
import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

TMP_DIR = OUTPUT_DIR / "step15_tmp"

RAW_CANDIDATES = OUTPUT_DIR / "step15_candidates_raw.tsv"
FINAL_CANDIDATES = OUTPUT_DIR / "step15_candidates.tsv"
GT_PAIRS = OUTPUT_DIR / "step15_ground_truth_pairs.tsv"
REPORT = OUTPUT_DIR / "step15_blocking_recall_report.txt"


# ============================================================
# SETTINGS
# ============================================================

# Maximum number of S2/S3 records allowed for a blocking key.
#
# Very common keys such as:
#   "inc"
#   "llc"
#   "road"
#   ""
#
# can otherwise create millions of Cartesian-product pairs.
#
# These caps protect both runtime and disk space.
MAX_SOURCE_BLOCK = 1000
MAX_S1_BLOCK = 1000

# GNU sort memory.
# Keep this conservative because the machine has limited RAM.
SORT_MEMORY = "256M"

# Number of parallel sort threads.
SORT_THREADS = "2"


# ============================================================
# NORMALIZATION
# ============================================================

NON_ALNUM = re.compile(r"[^a-z0-9]+")
MULTISPACE = re.compile(r"\s+")


def normalize_text(value):
    if value is None:
        return ""

    value = str(value).lower().strip()

    if not value:
        return ""

    value = value.replace("&", " and ")

    # Transliteration is intentionally NOT done here.
    # We do not want to destroy useful multilingual information.
    value = NON_ALNUM.sub(" ", value)
    value = MULTISPACE.sub(" ", value)

    return value.strip()


def compact_text(value):
    if value is None:
        return ""

    value = str(value).lower().strip()

    if not value:
        return ""

    return NON_ALNUM.sub("", value)


def token_signature(value):
    value = normalize_text(value)

    if not value:
        return ""

    tokens = value.split()

    if not tokens:
        return ""

    return " ".join(sorted(set(tokens)))


def digit_signature(value):
    if value is None:
        return ""

    value = str(value)

    digits = re.findall(r"\d+", value)

    if not digits:
        return ""

    return "".join(digits)


# ============================================================
# BLOCKING KEY GENERATION
# ============================================================

def get_block_keys(row, block_type):
    """
    Return zero or more blocking keys for a record.
    """

    country = normalize_text(row.get("country", ""))

    if not country:
        return []

    name = row.get("business_name", "") or ""
    address = row.get("business_address", "") or ""

    if block_type == "name_norm":
        value = normalize_text(name)

    elif block_type == "name_compact":
        value = compact_text(name)

    elif block_type == "name_tokens":
        value = token_signature(name)

    elif block_type == "address_norm":
        value = normalize_text(address)

    elif block_type == "address_compact":
        value = compact_text(address)

    elif block_type == "address_digits":
        value = digit_signature(address)

    else:
        raise ValueError(f"Unknown block type: {block_type}")

    # Empty keys are never useful.
    if not value:
        return []

    # Extremely short generic keys are dangerous.
    if block_type in {
        "name_norm",
        "name_compact",
        "name_tokens",
    } and len(value) < 3:
        return []

    if block_type in {
        "address_norm",
        "address_compact",
        "address_digits",
    } and len(value) < 3:
        return []

    return [country + "\x1f" + value]


# ============================================================
# UTILS
# ============================================================

def check_command(command):
    if shutil.which(command) is None:
        print(f"ERROR: Required command not found: {command}")
        print("Install GNU coreutils if necessary.")
        sys.exit(1)


def run_sort(input_file, output_file):
    """
    External GNU sort.
    """

    command = [
        "sort",
        "-T", str(TMP_DIR),
        f"--buffer-size={SORT_MEMORY}",
        f"--parallel={SORT_THREADS}",
        "-t", "\t",
        "-k1,1",
        str(input_file),
        "-o", str(output_file),
    ]

    subprocess.run(command, check=True)


def run_sort_unique(input_file, output_file):
    """
    Sort and deduplicate candidate pairs.
    """

    command = [
        "sort",
        "-T", str(TMP_DIR),
        f"--buffer-size={SORT_MEMORY}",
        f"--parallel={SORT_THREADS}",
        "-t", "\t",
        "-k1,1",
        "-k2,2",
        "-u",
        str(input_file),
        "-o", str(output_file),
    ]

    subprocess.run(command, check=True)


def remove_file(path):
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


# ============================================================
# CREATE S1 BLOCK FILE
# ============================================================

def create_s1_blocks(block_type, output_file):
    """
    Creates:

        block_key <TAB> S1 <TAB> entity_id

    for Source 1.
    """

    print(f"  Reading S1: {S1_FILE.name}")

    rows = 0
    keys = 0

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f, open(
        output_file,
        "w",
        encoding="utf-8",
    ) as out:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            rows += 1

            entity_id = row.get("entity_id", "")

            if not entity_id:
                continue

            block_keys = get_block_keys(row, block_type)

            for key in block_keys:
                out.write(
                    f"{key}\tS1\t{entity_id}\n"
                )
                keys += 1

            if rows % 500_000 == 0:
                print(
                    f"    S1 rows processed: {rows:,}",
                    flush=True,
                )

    print(
        f"  S1 complete: {rows:,} rows, "
        f"{keys:,} blocking entries"
    )


# ============================================================
# CREATE SOURCE BLOCK FILE
# ============================================================

def create_source_blocks(block_type, source_file, output_file):
    """
    Creates:

        block_key <TAB> S2/S3 <TAB> entity_id

    """

    source_name = (
        "S2"
        if "source2" in source_file.name
        else "S3"
    )

    print(f"  Reading {source_name}: {source_file.name}")

    rows = 0
    keys = 0

    with open(
        source_file,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f, open(
        output_file,
        "w",
        encoding="utf-8",
    ) as out:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            rows += 1

            entity_id = row.get("entity_id", "")

            if not entity_id:
                continue

            block_keys = get_block_keys(row, block_type)

            for key in block_keys:
                out.write(
                    f"{key}\t{source_name}\t{entity_id}\n"
                )
                keys += 1

            if rows % 500_000 == 0:
                print(
                    f"    {source_name} rows processed: "
                    f"{rows:,}",
                    flush=True,
                )

    print(
        f"  {source_name} complete: {rows:,} rows, "
        f"{keys:,} blocking entries"
    )


# ============================================================
# MERGE BLOCKS
# ============================================================

def generate_candidates_from_sorted_blocks(
    s1_sorted,
    source_sorted,
    output_file,
):
    """
    Both files are sorted by block key.

    For every matching block:
        S1 records x source records

    are emitted unless the block is too large.
    """

    generated = 0
    skipped_large = 0
    blocks = 0

    with open(
        s1_sorted,
        "r",
        encoding="utf-8",
    ) as f1, open(
        source_sorted,
        "r",
        encoding="utf-8",
    ) as f2, open(
        output_file,
        "w",
        encoding="utf-8",
    ) as out:

        line1 = f1.readline()
        line2 = f2.readline()

        while line1 and line2:

            parts1 = line1.rstrip("\n").split("\t")
            parts2 = line2.rstrip("\n").split("\t")

            key1 = parts1[0]
            key2 = parts2[0]

            if key1 < key2:
                line1 = f1.readline()
                continue

            if key2 < key1:
                line2 = f2.readline()
                continue

            # ------------------------------------------------
            # Read entire S1 block
            # ------------------------------------------------

            s1_ids = []

            while line1:

                p = line1.rstrip("\n").split("\t")

                if p[0] != key1:
                    break

                if len(p) >= 3:
                    s1_ids.append(p[2])

                line1 = f1.readline()

            # ------------------------------------------------
            # Read entire source block
            # ------------------------------------------------

            source_records = []

            while line2:

                p = line2.rstrip("\n").split("\t")

                if p[0] != key2:
                    break

                if len(p) >= 3:
                    source_records.append(
                        (p[1], p[2])
                    )

                line2 = f2.readline()

            blocks += 1

            if (
                len(s1_ids) > MAX_S1_BLOCK
                or len(source_records) > MAX_SOURCE_BLOCK
            ):
                skipped_large += 1
                continue

            # ------------------------------------------------
            # Cartesian product
            # ------------------------------------------------

            for s1_id in s1_ids:

                for source, entity_id in source_records:

                    out.write(
                        f"{s1_id}\t{entity_id}\n"
                    )

                    generated += 1

    return generated, skipped_large, blocks


# ============================================================
# CREATE GROUND-TRUTH PAIRS
# ============================================================

def create_ground_truth_pairs():
    """
    Convert:

        S1 -> S2,S3,S3...

    into:

        S1    S2
        S1    S3

    This allows external sort comparison.
    """

    print("\nCreating sorted ground-truth pair file...")

    raw_gt = TMP_DIR / "ground_truth_raw.tsv"

    count = 0

    with open(
        GT_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f, open(
        raw_gt,
        "w",
        encoding="utf-8",
    ) as out:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            s1 = row.get("source1_entity_id", "")

            if not s1:
                continue

            matched = (
                row.get("matched_entity_ids", "")
                or ""
            ).strip()

            if not matched:
                continue

            for entity_id in matched.split(","):

                entity_id = entity_id.strip()

                if entity_id.startswith(("S2-", "S3-")):
                    out.write(
                        f"{s1}\t{entity_id}\n"
                    )
                    count += 1

    print(
        f"  Ground-truth pairs: {count:,}"
    )

    run_sort_unique(
        raw_gt,
        GT_PAIRS,
    )

    remove_file(raw_gt)

    return count


# ============================================================
# COMPARE CANDIDATES WITH GROUND TRUTH
# ============================================================

def calculate_recall(candidate_file):
    """
    Both files:

        S1    candidate_id

    sorted.

    Compute:
        true matches
        recovered true matches
        false candidates
        recall
    """

    print("\nCalculating blocking recall...")

    true_total = 0
    recovered = 0

    candidate_total = 0

    with open(
        GT_PAIRS,
        "r",
        encoding="utf-8",
    ) as gt, open(
        candidate_file,
        "r",
        encoding="utf-8",
    ) as cand:

        gt_line = gt.readline()
        cand_line = cand.readline()

        while gt_line:

            true_total += 1

            if not cand_line:
                gt_line = gt.readline()
                continue

            gt_pair = gt_line.rstrip("\n")

            while cand_line:

                cand_pair = cand_line.rstrip("\n")

                if cand_pair < gt_pair:
                    candidate_total += 1
                    cand_line = cand.readline()
                    continue

                if cand_pair == gt_pair:
                    recovered += 1
                    candidate_total += 1
                    cand_line = cand.readline()
                    break

                # candidate > ground truth
                break

            gt_line = gt.readline()

    # Candidate count from file is safer to calculate separately.
    candidate_total = 0

    with open(
        candidate_file,
        "r",
        encoding="utf-8",
    ) as f:

        for _ in f:
            candidate_total += 1

    recall = (
        recovered / true_total
        if true_total
        else 0.0
    )

    return (
        true_total,
        recovered,
        candidate_total,
        recall,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    start_total = time.time()

    print("=" * 72)
    print("STEP 15 — DISK-BACKED BLOCKING RECALL BENCHMARK")
    print("=" * 72)

    print()
    print(f"Project: {ROOT}")
    print(f"Temporary directory: {TMP_DIR}")
    print()
    print("RAM-safe mode:")
    print(f"  sort memory: {SORT_MEMORY}")
    print(f"  sort threads: {SORT_THREADS}")
    print(f"  max source block: {MAX_SOURCE_BLOCK}")
    print(f"  max S1 block: {MAX_S1_BLOCK}")
    print()

    # --------------------------------------------------------
    # Check required commands
    # --------------------------------------------------------

    check_command("sort")

    # --------------------------------------------------------
    # Prepare temp directory
    # --------------------------------------------------------

    if TMP_DIR.exists():

        print("Cleaning previous Step 15 temporary files...")

        for p in TMP_DIR.iterdir():

            if p.is_file():
                p.unlink()

            elif p.is_dir():
                shutil.rmtree(p)

    else:
        TMP_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    remove_file(RAW_CANDIDATES)
    remove_file(FINAL_CANDIDATES)
    remove_file(GT_PAIRS)
    remove_file(REPORT)

    # --------------------------------------------------------
    # Create ground truth
    # --------------------------------------------------------

    gt_count = create_ground_truth_pairs()

    # --------------------------------------------------------
    # Blocking strategies
    # --------------------------------------------------------

    strategies = [
        "name_norm",
        "name_compact",
        "name_tokens",
        "address_norm",
        "address_compact",
        "address_digits",
    ]

    strategy_results = []

    # --------------------------------------------------------
    # Process each blocking strategy
    # --------------------------------------------------------

    for index, block_type in enumerate(
        strategies,
        start=1,
    ):

        print()
        print("=" * 72)
        print(
            f"BLOCK {index}/{len(strategies)}: "
            f"{block_type}"
        )
        print("=" * 72)

        block_start = time.time()

        s1_raw = TMP_DIR / f"s1_{block_type}.tsv"
        source2_raw = TMP_DIR / f"s2_{block_type}.tsv"
        source3_raw = TMP_DIR / f"s3_{block_type}.tsv"

        s1_sorted = TMP_DIR / f"s1_{block_type}_sorted.tsv"
        s2_sorted = TMP_DIR / f"s2_{block_type}_sorted.tsv"
        s3_sorted = TMP_DIR / f"s3_{block_type}_sorted.tsv"

        block_candidates = (
            TMP_DIR / f"candidates_{block_type}.tsv"
        )

        # ----------------------------------------------------
        # Generate blocking files
        # ----------------------------------------------------

        create_s1_blocks(
            block_type,
            s1_raw,
        )

        create_source_blocks(
            block_type,
            S2_FILE,
            source2_raw,
        )

        create_source_blocks(
            block_type,
            S3_FILE,
            source3_raw,
        )

        # ----------------------------------------------------
        # External sorting
        # ----------------------------------------------------

        print("  Sorting S1 blocks...")
        run_sort(
            s1_raw,
            s1_sorted,
        )

        print("  Sorting S2 blocks...")
        run_sort(
            source2_raw,
            s2_sorted,
        )

        print("  Sorting S3 blocks...")
        run_sort(
            source3_raw,
            s3_sorted,
        )

        # ----------------------------------------------------
        # Generate candidates
        # ----------------------------------------------------

        print("  Matching S1 ↔ S2 blocks...")

        s2_count, s2_skipped, s2_blocks = (
            generate_candidates_from_sorted_blocks(
                s1_sorted,
                s2_sorted,
                block_candidates,
            )
        )

        # Append S3 candidates
        print("  Matching S1 ↔ S3 blocks...")

        s3_temp = (
            TMP_DIR / f"candidates_{block_type}_s3.tsv"
        )

        s3_count, s3_skipped, s3_blocks = (
            generate_candidates_from_sorted_blocks(
                s1_sorted,
                s3_sorted,
                s3_temp,
            )
        )

        # Combine
        with open(
            block_candidates,
            "a",
            encoding="utf-8",
        ) as out, open(
            s3_temp,
            "r",
            encoding="utf-8",
        ) as src:

            shutil.copyfileobj(src, out)

        remove_file(s3_temp)

        # ----------------------------------------------------
        # Remove intermediate files
        # ----------------------------------------------------

        remove_file(s1_raw)
        remove_file(source2_raw)
        remove_file(source3_raw)

        remove_file(s1_sorted)
        remove_file(s2_sorted)
        remove_file(s3_sorted)

        elapsed = time.time() - block_start

        total_generated = (
            s2_count + s3_count
        )

        skipped_blocks = (
            s2_skipped + s3_skipped
        )

        total_blocks = (
            s2_blocks + s3_blocks
        )

        print()
        print(
            f"  {block_type} generated: "
            f"{total_generated:,}"
        )

        print(
            f"  Large blocks skipped: "
            f"{skipped_blocks:,}"
        )

        print(
            f"  Blocks examined: "
            f"{total_blocks:,}"
        )

        print(
            f"  Time: {elapsed:.2f} sec"
        )

        strategy_results.append(
            (
                block_type,
                total_generated,
                skipped_blocks,
                elapsed,
            )
        )

        # ----------------------------------------------------
        # Append into master raw candidate file
        # ----------------------------------------------------

        with open(
            block_candidates,
            "r",
            encoding="utf-8",
        ) as src, open(
            RAW_CANDIDATES,
            "a",
            encoding="utf-8",
        ) as dst:

            shutil.copyfileobj(src, dst)

        remove_file(block_candidates)

    # --------------------------------------------------------
    # Deduplicate all candidate pairs
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("DEDUPLICATING ALL CANDIDATES")
    print("=" * 72)

    print(
        "Raw candidates file size:",
        f"{RAW_CANDIDATES.stat().st_size / (1024 ** 3):.2f} GB"
        if RAW_CANDIDATES.exists()
        else "0 GB",
    )

    run_sort_unique(
        RAW_CANDIDATES,
        FINAL_CANDIDATES,
    )

    remove_file(RAW_CANDIDATES)

    # --------------------------------------------------------
    # Calculate recall
    # --------------------------------------------------------

    true_total, recovered, candidate_total, recall = (
        calculate_recall(FINAL_CANDIDATES)
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    total_time = time.time() - start_total

    report_lines = [
        "=" * 72,
        "STEP 15 — BLOCKING RECALL REPORT",
        "=" * 72,
        "",
        f"Ground-truth true pairs: {true_total:,}",
        f"Unique candidate pairs:  {candidate_total:,}",
        f"Recovered true pairs:   {recovered:,}",
        "",
        f"BLOCKING RECALL:         {recall * 100:.4f}%",
        "",
        "Blocking strategies:",
        "",
    ]

    for (
        block_type,
        generated,
        skipped,
        elapsed,
    ) in strategy_results:

        report_lines.append(
            f"{block_type:20s} "
            f"generated={generated:,} "
            f"skipped={skipped:,} "
            f"time={elapsed:.1f}s"
        )

    report_lines.extend(
        [
            "",
            f"Total runtime: {total_time:.2f} sec",
            "",
            f"Candidates: {FINAL_CANDIDATES}",
            "",
            "=" * 72,
        ]
    )

    report_text = "\n".join(report_lines)

    print()
    print(report_text)

    REPORT.write_text(
        report_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Clean temporary directory
    # --------------------------------------------------------

    print()
    print("Cleaning temporary files...")

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    print()
    print("=" * 72)
    print("STEP 15 COMPLETE")
    print("=" * 72)
    print()
    print(f"Candidate file: {FINAL_CANDIDATES}")
    print(f"Report:         {REPORT}")
    print()

    # --------------------------------------------------------
    # Decision guidance
    # --------------------------------------------------------

    if recall >= 0.99:

        print(
            "BLOCKING RECALL >= 99%."
        )
        print(
            "Proceed to test-set candidate generation."
        )

    elif recall >= 0.95:

        print(
            "BLOCKING RECALL is between 95% and 99%."
        )
        print(
            "Blocking needs one more fuzzy/rare-token "
            "strategy before final test inference."
        )

    else:

        print(
            "BLOCKING RECALL is below 95%."
        )
        print(
            "Do NOT use this candidate generator for "
            "the final test submission yet."
        )


if __name__ == "__main__":
    main()