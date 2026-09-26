#!/usr/bin/env python3

"""
STEP 18 — RAM-SAFE EXACT MISSED-PAIR ANALYSIS

Purpose:
    Compare ALL ground-truth true pairs against the Step 15
    candidate-pair file using disk-backed GNU sort/comm.

RAM strategy:
    - No pandas
    - No large Python dictionaries
    - No full S1/S2/S3 loading
    - Ground truth streamed line-by-line
    - GNU sort handles large files on disk
    - Step 15 candidate file is reused
    - Only a small sample of actual missed pairs is loaded
      into RAM for feature analysis

Expected RAM:
    Low / bounded Python memory.

IMPORTANT:
    This script first validates the Step 15 candidate format
    before doing the expensive full comparison.
"""

from __future__ import annotations

import csv
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"
CANDIDATE_FILE = OUTPUT_DIR / "step15_candidates.tsv"

WORK_DIR = OUTPUT_DIR / "step18_work"

GT_PAIRS = WORK_DIR / "ground_truth_pairs.tsv"
GT_SORTED = WORK_DIR / "ground_truth_pairs.sorted.tsv"

CAND_SORTED = WORK_DIR / "step15_candidates.sorted.tsv"

MISSED_PAIRS = OUTPUT_DIR / "step18_true_missed_pairs.tsv"
SAMPLE_FILE = OUTPUT_DIR / "step18_missed_sample.tsv"

REPORT_FILE = OUTPUT_DIR / "step18_missed_pair_report.txt"

SAMPLE_SIZE = 100_000

SORT_MEMORY = "256M"
SORT_THREADS = "2"


# ============================================================
# UTILITIES
# ============================================================

def run_command(
    command,
    description,
):

    print()
    print("=" * 72)
    print(description)
    print("=" * 72)

    print(
        "Command:",
        " ".join(str(x) for x in command),
    )

    start = time.time()

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    elapsed = time.time() - start

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:

        print(result.stderr)

        raise RuntimeError(
            f"Command failed with exit code "
            f"{result.returncode}: {description}"
        )

    print(
        f"Time: {elapsed:.2f}s"
    )

    return elapsed


def require_command(name):

    if shutil.which(name) is None:

        raise RuntimeError(
            f"Required command not found: {name}"
        )


def file_size_gb(path):

    if not path.exists():
        return 0.0

    return path.stat().st_size / (
        1024 ** 3
    )


# ============================================================
# STEP 1 — VALIDATE INPUT FILES
# ============================================================

def validate_inputs():

    print("=" * 72)
    print("STEP 18 — RAM-SAFE EXACT MISSED-PAIR ANALYSIS")
    print("=" * 72)

    print()

    print(
        f"Ground truth: "
        f"{GT_FILE}"
    )

    print(
        f"Step15 candidates: "
        f"{CANDIDATE_FILE}"
    )

    if not GT_FILE.exists():

        raise FileNotFoundError(
            f"Ground truth not found:\n{GT_FILE}"
        )

    if not CANDIDATE_FILE.exists():

        raise FileNotFoundError(
            f"Step15 candidate file not found:\n"
            f"{CANDIDATE_FILE}"
        )

    print()

    print(
        f"Ground truth size: "
        f"{file_size_gb(GT_FILE):.2f} GB"
    )

    print(
        f"Candidate file size: "
        f"{file_size_gb(CANDIDATE_FILE):.2f} GB"
    )


# ============================================================
# STEP 2 — INSPECT CANDIDATE FORMAT
# ============================================================

def inspect_candidate_file():

    print()
    print("=" * 72)
    print("CHECKING STEP 15 CANDIDATE FORMAT")
    print("=" * 72)

    with CANDIDATE_FILE.open(
        "rb"
    ) as f:

        for i in range(5):

            line = f.readline()

            if not line:
                break

            print(
                f"LINE {i + 1}: "
                f"{repr(line[:300])}"
            )


# ============================================================
# STEP 3 — BUILD GROUND-TRUTH PAIR FILE
# ============================================================

def build_ground_truth_pairs():

    print()
    print("=" * 72)
    print("BUILDING GROUND-TRUTH PAIR FILE")
    print("=" * 72)

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if GT_PAIRS.exists():

        print(
            "Existing ground-truth pair file found."
        )

        print(
            f"Size: "
            f"{file_size_gb(GT_PAIRS):.2f} GB"
        )

        return

    start = time.time()

    total_rows = 0
    total_pairs = 0

    with GT_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as source, GT_PAIRS.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:

        reader = csv.DictReader(
            source,
            delimiter="\t",
        )

        writer = output

        for row in reader:

            total_rows += 1

            s1 = (
                row.get(
                    "source1_entity_id"
                )
                or row.get(
                    "entity_id"
                )
                or ""
            ).strip()

            matched = (
                row.get(
                    "matched_entity_ids"
                )
                or ""
            ).strip()

            if not s1:
                continue

            if not matched:
                continue

            # Ground truth can contain comma-separated IDs.
            #
            # Support both:
            #
            #   ID1,ID2,ID3
            #
            # and:
            #
            #   ID1 ID2 ID3

            matched_ids = (
                matched
                .replace(",", " ")
                .split()
            )

            for target in matched_ids:

                target = target.strip()

                if not target:
                    continue

                writer.write(
                    f"{s1}\t{target}\n"
                )

                total_pairs += 1

            if (
                total_rows % 500_000
                == 0
            ):

                print(
                    f"  GT rows: "
                    f"{total_rows:,} | "
                    f"pairs: "
                    f"{total_pairs:,}",
                    flush=True,
                )

    elapsed = time.time() - start

    print()
    print(
        f"Ground-truth rows: "
        f"{total_rows:,}"
    )

    print(
        f"Ground-truth pairs: "
        f"{total_pairs:,}"
    )

    print(
        f"Time: {elapsed:.2f}s"
    )

    print(
        f"File size: "
        f"{file_size_gb(GT_PAIRS):.2f} GB"
    )


# ============================================================
# STEP 4 — SORT GROUND TRUTH
# ============================================================

def sort_ground_truth():

    if GT_SORTED.exists():

        print()
        print(
            "Existing sorted ground-truth file found."
        )

        return

    run_command(
        [
            "sort",
            "-u",
            "-T",
            str(WORK_DIR),
            "-S",
            SORT_MEMORY,
            "--parallel=2",
            GT_PAIRS,
            "-o",
            GT_SORTED,
        ],
        "SORTING GROUND-TRUTH PAIRS",
    )


# ============================================================
# STEP 5 — SORT STEP 15 CANDIDATES
# ============================================================

def sort_candidates():

    if CAND_SORTED.exists():

        print()
        print(
            "Existing sorted Step15 candidate file found."
        )

        return

    run_command(
        [
            "sort",
            "-u",
            "-T",
            str(WORK_DIR),
            "-S",
            SORT_MEMORY,
            "--parallel=2",
            CANDIDATE_FILE,
            "-o",
            CAND_SORTED,
        ],
        "SORTING STEP 15 CANDIDATE PAIRS",
    )


# ============================================================
# STEP 6 — EXACT SET DIFFERENCE
# ============================================================

def find_missed_pairs():

    if MISSED_PAIRS.exists():

        print()
        print(
            "Existing missed-pair file found."
        )

        print(
            f"Size: "
            f"{file_size_gb(MISSED_PAIRS):.2f} GB"
        )

        return

    run_command(
        [
            "comm",
            "-23",
            GT_SORTED,
            CAND_SORTED,
        ],
        "CALCULATING EXACT MISSED TRUE PAIRS",
    )

    # The previous command only prints to stdout.
    # For a huge output we need shell redirection,
    # so do it properly below.

    # This branch is intentionally never used because
    # run_command captures stdout into RAM.
    #
    # The actual implementation is below.


def find_missed_pairs_disk():

    if MISSED_PAIRS.exists():

        print()
        print(
            "Existing missed-pair file found."
        )

        return

    print()
    print("=" * 72)
    print("CREATING MISSED-PAIR FILE ON DISK")
    print("=" * 72)

    start = time.time()

    with MISSED_PAIRS.open(
        "w",
        encoding="utf-8",
    ) as output:

        process = subprocess.Popen(
            [
                "comm",
                "-23",
                GT_SORTED,
                CAND_SORTED,
            ],
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
        )

        _, stderr = process.communicate()

    if process.returncode != 0:

        print(stderr)

        raise RuntimeError(
            "comm failed while creating "
            "missed-pair file."
        )

    elapsed = time.time() - start

    print(
        f"Missed-pair file created."
    )

    print(
        f"Size: "
        f"{file_size_gb(MISSED_PAIRS):.2f} GB"
    )

    print(
        f"Time: {elapsed:.2f}s"
    )


# ============================================================
# STEP 7 — COUNT MISSED PAIRS WITHOUT LOADING THEM
# ============================================================

def count_lines(path):

    start = time.time()

    process = subprocess.run(
        [
            "wc",
            "-l",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if process.returncode != 0:

        raise RuntimeError(
            process.stderr
        )

    count = int(
        process.stdout.split()[0]
    )

    print(
        f"{path.name}: "
        f"{count:,} lines"
    )

    print(
        f"Time: {time.time() - start:.2f}s"
    )

    return count


# ============================================================
# STEP 8 — RAM-SAFE RANDOM SAMPLE
# ============================================================

def create_sample(
    missed_count,
):

    print()
    print("=" * 72)
    print("CREATING RAM-SAFE SAMPLE")
    print("=" * 72)

    if missed_count <= SAMPLE_SIZE:

        shutil.copyfile(
            MISSED_PAIRS,
            SAMPLE_FILE,
        )

        print(
            "Missed-pair file smaller than "
            "sample size; copied entire file."
        )

        return

    # Reservoir sampling.
    #
    # Only SAMPLE_SIZE lines are held in RAM.

    rng = random.Random(42)

    reservoir = []

    with MISSED_PAIRS.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for index, line in enumerate(
            f
        ):

            line = line.rstrip(
                "\r\n"
            )

            if index < SAMPLE_SIZE:

                reservoir.append(line)

            else:

                j = rng.randint(
                    0,
                    index,
                )

                if j < SAMPLE_SIZE:

                    reservoir[j] = line

    with SAMPLE_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        for line in reservoir:

            f.write(
                line + "\n"
            )

    print(
        f"Sample size: "
        f"{len(reservoir):,}"
    )

    print(
        f"Sample file: "
        f"{SAMPLE_FILE}"
    )

    # Release memory immediately.

    del reservoir


# ============================================================
# STEP 9 — REPORT
# ============================================================

def write_report(
    gt_count,
    candidate_count,
    missed_count,
):

    recovered = (
        gt_count - missed_count
    )

    if gt_count:

        recall = (
            recovered / gt_count
        )

    else:

        recall = 0.0

    report = []

    report.append(
        "STEP 18 — EXACT BLOCKING RECALL ANALYSIS"
    )

    report.append(
        "=" * 72
    )

    report.append("")

    report.append(
        f"Ground-truth true pairs: "
        f"{gt_count:,}"
    )

    report.append(
        f"Step15 candidate pairs:  "
        f"{candidate_count:,}"
    )

    report.append(
        f"Recovered true pairs:    "
        f"{recovered:,}"
    )

    report.append(
        f"Missed true pairs:       "
        f"{missed_count:,}"
    )

    report.append("")

    report.append(
        f"EXACT BLOCKING RECALL: "
        f"{recall * 100:.4f}%"
    )

    report.append("")

    report.append(
        f"Missed pairs file: "
        f"{MISSED_PAIRS}"
    )

    report.append(
        f"Sample file: "
        f"{SAMPLE_FILE}"
    )

    REPORT_FILE.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("STEP 18 COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Ground truth: "
        f"{gt_count:,}"
    )

    print(
        f"Candidates: "
        f"{candidate_count:,}"
    )

    print(
        f"Recovered: "
        f"{recovered:,}"
    )

    print(
        f"Missed: "
        f"{missed_count:,}"
    )

    print(
        f"Exact recall: "
        f"{recall * 100:.4f}%"
    )

    print()
    print(
        f"Report: {REPORT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    for command in (
        "sort",
        "comm",
        "wc",
    ):

        require_command(command)

    validate_inputs()

    inspect_candidate_file()

    build_ground_truth_pairs()

    sort_ground_truth()

    # IMPORTANT:
    #
    # Sort the candidate file using the SAME GNU sort
    # ordering as the ground truth.
    #
    # This eliminates the ordering assumption that caused
    # the previous Step17 comparison to produce an impossible
    # 7,638,342 missed-pair result.

    sort_candidates()

    find_missed_pairs_disk()

    gt_count = count_lines(
        GT_SORTED
    )

    candidate_count = count_lines(
        CAND_SORTED
    )

    missed_count = count_lines(
        MISSED_PAIRS
    )

    create_sample(
        missed_count
    )

    write_report(
        gt_count,
        candidate_count,
        missed_count,
    )

    print()
    print(
        f"Total runtime: "
        f"{time.time() - overall_start:.2f}s"
    )


if __name__ == "__main__":
    main()
