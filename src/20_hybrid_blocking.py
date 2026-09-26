#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

WORK_DIR = OUTPUT_DIR / "step20_work"

FINAL_CANDIDATES = OUTPUT_DIR / "step20_candidates.tsv"
REPORT = OUTPUT_DIR / "step20_blocking_report.txt"

SORT_MEMORY = "256M"
SORT_THREADS = "2"

# IMPORTANT:
# Blocks larger than this are skipped.
MAX_BLOCK_SIZE = 1000


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

    return " ".join(
        "".join(out).split()
    )


def compact(value):

    if not value:
        return ""

    return "".join(
        ch for ch in value.lower()
        if ch.isalnum()
    )


def first_token(value):

    text = clean_text(value)

    if not text:
        return ""

    parts = text.split()

    return parts[0] if parts else ""


def name_keys(name, country):

    text = clean_text(name)

    if not text:
        return []

    tokens = [
        x for x in text.split()
        if len(x) >= 3
    ]

    keys = set()

    # --------------------------------------------------------
    # Name prefix
    # --------------------------------------------------------

    if len(text) >= 6:

        keys.add(
            f"N6|{country}|{text[:6]}"
        )

    elif len(text) >= 4:

        keys.add(
            f"N4|{country}|{text[:4]}"
        )

    # --------------------------------------------------------
    # First token prefix
    # --------------------------------------------------------

    if tokens:

        first = tokens[0]

        if len(first) >= 5:

            keys.add(
                f"NT5|{country}|{first[:5]}"
            )

        if len(first) >= 4:

            keys.add(
                f"NT4|{country}|{first[:4]}"
            )

    # --------------------------------------------------------
    # Individual token prefixes
    #
    # We use prefixes rather than complete tokens.
    # This is deliberately different from Step 16.
    # --------------------------------------------------------

    for token in tokens:

        if len(token) >= 5:

            keys.add(
                f"TK5|{country}|{token[:5]}"
            )

    return list(keys)


def address_keys(address, country):

    text = clean_text(address)

    if not text:
        return []

    tokens = [
        x for x in text.split()
        if len(x) >= 3
    ]

    keys = set()

    # --------------------------------------------------------
    # Address character prefix
    # --------------------------------------------------------

    if len(text) >= 8:

        keys.add(
            f"A8|{country}|{text[:8]}"
        )

    elif len(text) >= 5:

        keys.add(
            f"A5|{country}|{text[:5]}"
        )

    # --------------------------------------------------------
    # First address token prefix
    # --------------------------------------------------------

    if tokens:

        first = tokens[0]

        if len(first) >= 5:

            keys.add(
                f"AT5|{country}|{first[:5]}"
            )

        if len(first) >= 4:

            keys.add(
                f"AT4|{country}|{first[:4]}"
            )

    # --------------------------------------------------------
    # Numeric/address prefix
    #
    # Only first numeric component.
    # This avoids Step15's enormous full-digit blocks.
    # --------------------------------------------------------

    number = ""

    for token in text.split():

        digits = "".join(
            ch for ch in token
            if ch.isdigit()
        )

        if digits:

            number = digits[:4]
            break

    if len(number) >= 2:

        keys.add(
            f"D4|{country}|{number}"
        )

    return list(keys)


# ============================================================
# CREATE BLOCK RECORDS
# ============================================================

def create_block_file(
    source,
    input_file,
    output_file,
):

    print()
    print("=" * 72)
    print(
        f"CREATING BLOCK RECORDS: {source}"
    )
    print("=" * 72)

    start = time.time()

    rows = 0
    block_records = 0

    with input_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f, output_file.open(
        "w",
        encoding="utf-8",
    ) as out:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            rows += 1

            entity_id = (
                row.get(
                    "entity_id"
                ) or ""
            ).strip()

            if not entity_id:
                continue

            country = (
                row.get(
                    "country"
                ) or ""
            ).strip().lower()

            name = (
                row.get(
                    "business_name"
                ) or ""
            )

            address = (
                row.get(
                    "business_address"
                ) or ""
            )

            keys = set()

            keys.update(
                name_keys(
                    name,
                    country,
                )
            )

            keys.update(
                address_keys(
                    address,
                    country,
                )
            )

            for key in keys:

                out.write(
                    f"{key}\t"
                    f"{source}\t"
                    f"{entity_id}\n"
                )

                block_records += 1

            if rows % 1_000_000 == 0:

                print(
                    f"  rows={rows:,} "
                    f"block_records={block_records:,}",
                    flush=True,
                )

    elapsed = time.time() - start

    print()
    print(
        f"Rows: {rows:,}"
    )

    print(
        f"Block records: "
        f"{block_records:,}"
    )

    print(
        f"Time: {elapsed:.2f}s"
    )


# ============================================================
# SORT BLOCK FILE
# ============================================================

def sort_file(
    input_file,
    output_file,
):

    if output_file.exists():
        return

    command = [
        "sort",
        "-T",
        str(WORK_DIR),
        "-S",
        SORT_MEMORY,
        "--parallel=2",
        input_file,
        "-o",
        output_file,
    ]

    print()
    print(
        "Sorting:",
        input_file.name,
    )

    start = time.time()

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        print(result.stderr)

        raise RuntimeError(
            "sort failed"
        )

    print(
        f"Sort time: "
        f"{time.time() - start:.2f}s"
    )


# ============================================================
# GENERATE CANDIDATES FROM SORTED BLOCKS
# ============================================================

def generate_candidates(
    sorted_file,
    output_file,
):

    print()
    print("=" * 72)
    print(
        f"GENERATING CANDIDATES: "
        f"{sorted_file.name}"
    )
    print("=" * 72)

    start = time.time()

    current_key = None

    s1_ids = []
    target_ids = []

    blocks = 0
    skipped = 0
    pairs = 0

    def flush():

        nonlocal blocks
        nonlocal skipped
        nonlocal pairs

        if not current_key:
            return

        blocks += 1

        # We need both S1 and target records.

        if not s1_ids or not target_ids:
            return

        total_size = (
            len(s1_ids)
            + len(target_ids)
        )

        if total_size > MAX_BLOCK_SIZE:

            skipped += 1

            return

        for s1 in s1_ids:

            for target in target_ids:

                output_line = (
                    f"{s1}\t{target}\n"
                )

                output.write(
                    output_line
                )

                pairs += 1

    with sorted_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f, output_file.open(
        "w",
        encoding="utf-8",
    ) as output:

        for line in f:

            line = line.rstrip(
                "\r\n"
            )

            if not line:
                continue

            parts = line.split(
                "\t"
            )

            if len(parts) != 3:
                continue

            key = parts[0]
            source = parts[1]
            entity_id = parts[2]

            if (
                current_key is not None
                and key != current_key
            ):

                flush()

                s1_ids = []
                target_ids = []

            if key != current_key:

                current_key = key

            if source == "S1":

                s1_ids.append(
                    entity_id
                )

            elif (
                source == "S2"
                or source == "S3"
            ):

                target_ids.append(
                    entity_id
                )

    elapsed = time.time() - start

    print()
    print(
        f"Blocks: {blocks:,}"
    )

    print(
        f"Skipped oversized: "
        f"{skipped:,}"
    )

    print(
        f"Raw candidate pairs: "
        f"{pairs:,}"
    )

    print(
        f"Time: {elapsed:.2f}s"
    )


# ============================================================
# DEDUPLICATE CANDIDATES
# ============================================================

def deduplicate(
    raw_files,
    final_file,
):

    print()
    print("=" * 72)
    print("DEDUPLICATING HYBRID CANDIDATES")
    print("=" * 72)

    combined = WORK_DIR / (
        "all_raw_candidates.tsv"
    )

    if not combined.exists():

        print(
            "Combining candidate files..."
        )

        with combined.open(
            "wb"
        ) as out:

            for path in raw_files:

                with path.open(
                    "rb"
                ) as inp:

                    shutil.copyfileobj(
                        inp,
                        out,
                        length=1024 * 1024,
                    )

    start = time.time()

    temp_sorted = WORK_DIR / (
        "all_candidates.sorted.tsv"
    )

    command = [
        "sort",
        "-u",
        "-T",
        str(WORK_DIR),
        "-S",
        SORT_MEMORY,
        "--parallel=2",
        combined,
        "-o",
        temp_sorted,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        print(result.stderr)

        raise RuntimeError(
            "Candidate deduplication failed."
        )

    shutil.copyfile(
        temp_sorted,
        final_file,
    )

    print(
        f"Final candidates: "
        f"{final_file}"
    )

    print(
        f"Size: "
        f"{final_file.stat().st_size / (1024**3):.2f} GB"
    )

    print(
        f"Time: "
        f"{time.time() - start:.2f}s"
    )


# ============================================================
# EVALUATE AGAINST GROUND TRUTH
# ============================================================

def evaluate():

    gt_sorted = (
        OUTPUT_DIR /
        "step18_work" /
        "ground_truth_pairs.sorted.tsv"
    )

    if not gt_sorted.exists():

        raise FileNotFoundError(
            "Step18 sorted ground truth missing."
        )

    candidate_sorted = (
        WORK_DIR /
        "final_candidates.sorted.tsv"
    )

    print()
    print("=" * 72)
    print("EVALUATING BLOCKING RECALL")
    print("=" * 72)

    start = time.time()

    shutil.copyfile(
        FINAL_CANDIDATES,
        candidate_sorted,
    )

    recovered_file = (
        WORK_DIR /
        "recovered.tsv"
    )

    missed_file = (
        WORK_DIR /
        "missed.tsv"
    )

    with recovered_file.open(
        "w"
    ) as recovered:

        result = subprocess.run(
            [
                "comm",
                "-12",
                gt_sorted,
                candidate_sorted,
            ],
            stdout=recovered,
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr
        )

    with missed_file.open(
        "w"
    ) as missed:

        result = subprocess.run(
            [
                "comm",
                "-23",
                gt_sorted,
                candidate_sorted,
            ],
            stdout=missed,
            stderr=subprocess.PIPE,
            text=True,
        )

    if result.returncode != 0:

        raise RuntimeError(
            result.stderr
        )

    gt_count = int(
        subprocess.check_output(
            [
                "wc",
                "-l",
                str(gt_sorted),
            ],
            text=True,
        ).split()[0]
    )

    candidate_count = int(
        subprocess.check_output(
            [
                "wc",
                "-l",
                str(candidate_sorted),
            ],
            text=True,
        ).split()[0]
    )

    recovered_count = int(
        subprocess.check_output(
            [
                "wc",
                "-l",
                str(recovered_file),
            ],
            text=True,
        ).split()[0]
    )

    missed_count = (
        gt_count -
        recovered_count
    )

    recall = (
        recovered_count /
        gt_count
    )

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
        f"{recovered_count:,}"
    )

    print(
        f"Missed: "
        f"{missed_count:,}"
    )

    print(
        f"Blocking recall: "
        f"{recall * 100:.4f}%"
    )

    print(
        f"Evaluation time: "
        f"{time.time() - start:.2f}s"
    )

    return (
        candidate_count,
        recovered_count,
        missed_count,
        recall,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    print("=" * 72)
    print("STEP 20 — RAM-SAFE HYBRID BLOCKING")
    print("=" * 72)

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    block_files = []
    sorted_files = []
    raw_candidate_files = []

    # --------------------------------------------------------
    # BUILD ONE BLOCK FILE PER SOURCE
    # --------------------------------------------------------

    for source, input_file in sources:

        block_file = (
            WORK_DIR /
            f"{source}_blocks.tsv"
        )

        sorted_file = (
            WORK_DIR /
            f"{source}_blocks.sorted.tsv"
        )

        if not block_file.exists():

            create_block_file(
                source,
                input_file,
                block_file,
            )

        sort_file(
            block_file,
            sorted_file,
        )

        block_files.append(
            block_file
        )

        sorted_files.append(
            sorted_file
        )

    # --------------------------------------------------------
    # COMBINE SORTED BLOCK FILES
    # --------------------------------------------------------

    combined_blocks = (
        WORK_DIR /
        "all_blocks.tsv"
    )

    if not combined_blocks.exists():

        print()
        print(
            "Combining sorted block files..."
        )

        with combined_blocks.open(
            "wb"
        ) as out:

            for path in sorted_files:

                with path.open(
                    "rb"
                ) as inp:

                    shutil.copyfileobj(
                        inp,
                        out,
                        length=1024 * 1024,
                    )

    all_blocks_sorted = (
        WORK_DIR /
        "all_blocks.sorted.tsv"
    )

    sort_file(
        combined_blocks,
        all_blocks_sorted,
    )

    # --------------------------------------------------------
    # GENERATE CANDIDATES
    # --------------------------------------------------------

    raw_candidates = (
        WORK_DIR /
        "raw_candidates.tsv"
    )

    if not raw_candidates.exists():

        generate_candidates(
            all_blocks_sorted,
            raw_candidates,
        )

    raw_candidate_files.append(
        raw_candidates
    )

    # --------------------------------------------------------
    # DEDUPLICATE
    # --------------------------------------------------------

    deduplicate(
        raw_candidate_files,
        FINAL_CANDIDATES,
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    (
        candidate_count,
        recovered_count,
        missed_count,
        recall,
    ) = evaluate()

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = []

    report.append(
        "STEP 20 — HYBRID BLOCKING"
    )

    report.append("=" * 72)

    report.append(
        f"Maximum block size: "
        f"{MAX_BLOCK_SIZE}"
    )

    report.append("")

    report.append(
        f"Ground truth pairs: "
        f"7,638,365"
    )

    report.append(
        f"Candidate pairs: "
        f"{candidate_count:,}"
    )

    report.append(
        f"Recovered pairs: "
        f"{recovered_count:,}"
    )

    report.append(
        f"Missed pairs: "
        f"{missed_count:,}"
    )

    report.append("")

    report.append(
        f"Blocking recall: "
        f"{recall * 100:.4f}%"
    )

    report.append("")

    report.append(
        f"Final candidate file: "
        f"{FINAL_CANDIDATES}"
    )

    report.append("")

    report.append(
        f"Total runtime: "
        f"{time.time() - overall_start:.2f}s"
    )

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("STEP 20 COMPLETE")
    print("=" * 72)

    print(
        f"Candidates: "
        f"{candidate_count:,}"
    )

    print(
        f"Recovered: "
        f"{recovered_count:,}"
    )

    print(
        f"Recall: "
        f"{recall * 100:.4f}%"
    )

    print(
        f"Report: {REPORT}"
    )


if __name__ == "__main__":
    main()
