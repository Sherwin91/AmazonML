#!/usr/bin/env python3

import csv
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path


# ============================================================
# STEP 22
# FREQUENCY-AWARE BLOCKING BENCHMARK
#
# CPU/RAM SAFE
#
# This script:
#   - reads the 100k actual missed-pair sample
#   - tests promising blocking keys
#   - uses GNU sort with ONE CPU worker
#   - uses a 128 MB sort buffer
#   - processes one blocking key at a time
#   - deletes temporary files after each key
#   - does NOT create final candidate pairs
#
# Missed-pair format:
#
#   S1-300566736    S3-821013098
#   S1-494594626    S3-739115080
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"

MISSED_SAMPLE = OUTPUT_DIR / "step18_missed_sample.tsv"

WORK_DIR = OUTPUT_DIR / "step22_work"

REPORT_FILE = OUTPUT_DIR / "step22_frequency_block_report.txt"


# ============================================================
# RESOURCE LIMITS
# ============================================================

# Keep CPU usage low.
SORT_PARALLEL = "1"

# Keep RAM usage low.
SORT_BUFFER = "128M"

# We will test these block-size limits.
BLOCK_LIMITS = [100, 300, 500, 1000]


# ============================================================
# BLOCKING KEYS
#
# Selected from Step 21 results.
# ============================================================

BLOCKING_KEYS = [
    "first_name_token_3",
    "name_prefix_4",
    "name_compact_prefix_4",
    "name_prefix_6",
    "name_prefix_7",
    "longest_address_token_4",
    "longest_address_token_5",
    "longest_name_token_4",
    "address_digits_2",
    "name4_address4",
    "first_name_first_address",
]


# ============================================================
# NORMALIZATION
# ============================================================

NON_ALNUM = re.compile(r"[^a-z0-9]+")


def clean_text(value):
    if not value:
        return ""

    value = value.lower().strip()
    value = NON_ALNUM.sub(" ", value)
    value = re.sub(r"\s+", " ", value)

    return value


def compact(value):
    return clean_text(value).replace(" ", "")


def get_tokens(value):
    value = clean_text(value)

    if not value:
        return []

    return value.split()


def get_digits(value):
    if not value:
        return ""

    return "".join(
        character
        for character in value
        if character.isdigit()
    )


def longest_token(value, minimum_length):
    valid = [
        token
        for token in get_tokens(value)
        if len(token) >= minimum_length
    ]

    if not valid:
        return ""

    return max(
        valid,
        key=lambda token: (len(token), token)
    )


# ============================================================
# BLOCKING KEY CREATION
# ============================================================

def create_blocking_key(
    key_name,
    business_name,
    business_address
):
    name = clean_text(business_name)
    address = clean_text(business_address)

    compact_name = compact(business_name)
    compact_address = compact(business_address)

    name_tokens = get_tokens(business_name)
    address_tokens = get_tokens(business_address)

    if key_name == "first_name_token_3":

        if name_tokens and len(name_tokens[0]) >= 3:
            return name_tokens[0][:3]

        return ""

    if key_name == "name_prefix_4":

        if len(compact_name) >= 4:
            return compact_name[:4]

        return ""

    if key_name == "name_compact_prefix_4":

        if len(compact_name) >= 4:
            return compact_name[:4]

        return ""

    if key_name == "name_prefix_6":

        if len(compact_name) >= 6:
            return compact_name[:6]

        return ""

    if key_name == "name_prefix_7":

        if len(compact_name) >= 7:
            return compact_name[:7]

        return ""

    if key_name == "longest_address_token_4":

        return longest_token(
            business_address,
            4
        )

    if key_name == "longest_address_token_5":

        return longest_token(
            business_address,
            5
        )

    if key_name == "longest_name_token_4":

        return longest_token(
            business_name,
            4
        )

    if key_name == "address_digits_2":

        digits = get_digits(business_address)

        if len(digits) >= 2:
            return digits[:2]

        return ""

    if key_name == "name4_address4":

        if (
            len(compact_name) >= 4
            and len(compact_address) >= 4
        ):
            return (
                compact_name[:4]
                + "|"
                + compact_address[:4]
            )

        return ""

    if key_name == "first_name_first_address":

        if name_tokens and address_tokens:

            return (
                name_tokens[0][:4]
                + "|"
                + address_tokens[0][:4]
            )

        return ""

    return ""


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(header, possible_names):
    positions = {}

    for index, value in enumerate(header):
        positions[value.strip().lower()] = index

    for name in possible_names:

        index = positions.get(name.lower())

        if index is not None:
            return index

    return None


# ============================================================
# LOAD MISSED PAIRS
# ============================================================

def load_missed_pairs():
    """
    Reads:

        S1_ID    S2/S3_ID

    Returns:

        total_pairs

        target_to_s1

    where:

        target_to_s1[
            ("S2", "S2-123")
        ] = {"S1-456", ...}

    """

    target_to_s1 = defaultdict(set)

    total_pairs = 0

    print("Loading actual missed-pair sample...")

    with open(
        MISSED_SAMPLE,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 2:
                parts = line.split()

            if len(parts) < 2:
                continue

            s1_id = parts[0].strip()
            target_id = parts[1].strip()

            if not s1_id or not target_id:
                continue

            if target_id.startswith("S2-"):
                source = "S2"

            elif target_id.startswith("S3-"):
                source = "S3"

            else:
                continue

            target_to_s1[
                (source, target_id)
            ].add(s1_id)

            total_pairs += 1

    print(
        f"Loaded missed pairs: {total_pairs:,}"
    )

    print(
        f"Unique target records: "
        f"{len(target_to_s1):,}"
    )

    return total_pairs, target_to_s1


# ============================================================
# GENERATE BLOCK RECORDS
# ============================================================

def generate_block_records(
    source,
    input_file,
    key_name,
    output_file
):
    """
    Streams one source file.

    Writes:

        blocking_key    source    entity_id
    """

    scanned = 0
    written = 0

    with open(
        input_file,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as input_handle:

        reader = csv.reader(
            input_handle,
            delimiter="\t"
        )

        header = next(reader, None)

        if header is None:
            return 0, 0

        id_index = find_column(
            header,
            ["entity_id", "id"]
        )

        name_index = find_column(
            header,
            ["business_name", "name"]
        )

        address_index = find_column(
            header,
            ["business_address", "address"]
        )

        if id_index is None:
            raise RuntimeError(
                f"entity_id column not found in "
                f"{input_file}"
            )

        with open(
            output_file,
            "a",
            encoding="utf-8"
        ) as output_handle:

            for row in reader:

                scanned += 1

                if id_index >= len(row):
                    continue

                entity_id = row[id_index].strip()

                if not entity_id:
                    continue

                if (
                    name_index is not None
                    and name_index < len(row)
                ):
                    business_name = row[name_index]
                else:
                    business_name = ""

                if (
                    address_index is not None
                    and address_index < len(row)
                ):
                    business_address = row[address_index]
                else:
                    business_address = ""

                blocking_key = create_blocking_key(
                    key_name,
                    business_name,
                    business_address
                )

                if not blocking_key:
                    continue

                output_handle.write(
                    blocking_key
                    + "\t"
                    + source
                    + "\t"
                    + entity_id
                    + "\n"
                )

                written += 1

    return scanned, written


# ============================================================
# SORT
# ============================================================

def sort_block_file(
    input_file,
    output_file
):
    """
    GNU sort.

    One worker only.
    """

    command = [
        "sort",
        "-t",
        "\t",
        "-k",
        "1,1",
        "-k",
        "2,2",
        "-k",
        "3,3",
        "-S",
        SORT_BUFFER,
        "--parallel=1",
        "-T",
        str(WORK_DIR),
        input_file,
        "-o",
        output_file,
    ]

    subprocess.run(
        command,
        check=True
    )


# ============================================================
# ANALYZE SORTED BLOCKS
# ============================================================

def analyze_sorted_blocks(
    sorted_file,
    total_missed_pairs,
    target_to_s1
):
    """
    Analyze blocks without constructing candidate pairs.

    For every blocking key:

        n1 = number of S1 records
        n2 = number of S2 records
        n3 = number of S3 records

    Candidate count:

        n1*n2 + n1*n3

    True missed-pair recovery is checked only against
    the 100k diagnostic sample.
    """

    block_count = 0

    oversized_1000 = 0

    raw_candidate_count = 0

    candidates_by_limit = {
        limit: 0
        for limit in BLOCK_LIMITS
    }

    blocks_by_limit = {
        limit: 0
        for limit in BLOCK_LIMITS
    }

    recovered_by_limit = {
        limit: set()
        for limit in BLOCK_LIMITS
    }

    current_key = None

    s1_ids = []
    s2_ids = []
    s3_ids = []

    def process_block():

        nonlocal block_count
        nonlocal oversized_1000
        nonlocal raw_candidate_count

        if current_key is None:
            return

        block_count += 1

        n1 = len(s1_ids)
        n2 = len(s2_ids)
        n3 = len(s3_ids)

        if n1 == 0:
            return

        candidate_count = (
            n1 * n2
            + n1 * n3
        )

        raw_candidate_count += candidate_count

        largest_source_count = max(
            n1,
            n2,
            n3
        )

        if largest_source_count > 1000:
            oversized_1000 += 1

        # ----------------------------------------------------
        # Build S1 lookup only for this block.
        # ----------------------------------------------------

        s1_set = set(s1_ids)

        # ----------------------------------------------------
        # Check each block-size limit.
        # ----------------------------------------------------

        for limit in BLOCK_LIMITS:

            if largest_source_count > limit:
                continue

            blocks_by_limit[limit] += 1

            candidates_by_limit[limit] += (
                candidate_count
            )

            # ------------------------------------------------
            # S2 targets
            # ------------------------------------------------

            for target_id in s2_ids:

                sampled_s1s = target_to_s1.get(
                    ("S2", target_id)
                )

                if not sampled_s1s:
                    continue

                for s1_id in sampled_s1s:

                    if s1_id in s1_set:

                        recovered_by_limit[
                            limit
                        ].add(
                            (
                                s1_id,
                                target_id
                            )
                        )

            # ------------------------------------------------
            # S3 targets
            # ------------------------------------------------

            for target_id in s3_ids:

                sampled_s1s = target_to_s1.get(
                    ("S3", target_id)
                )

                if not sampled_s1s:
                    continue

                for s1_id in sampled_s1s:

                    if s1_id in s1_set:

                        recovered_by_limit[
                            limit
                        ].add(
                            (
                                s1_id,
                                target_id
                            )
                        )

    # --------------------------------------------------------
    # Stream sorted blocks.
    # --------------------------------------------------------

    with open(
        sorted_file,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        for line in file:

            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                continue

            blocking_key = parts[0]
            source = parts[1]
            entity_id = parts[2]

            if current_key is None:
                current_key = blocking_key

            if blocking_key != current_key:

                process_block()

                current_key = blocking_key

                s1_ids = []
                s2_ids = []
                s3_ids = []

            if source == "S1":
                s1_ids.append(entity_id)

            elif source == "S2":
                s2_ids.append(entity_id)

            elif source == "S3":
                s3_ids.append(entity_id)

        process_block()

    return {
        "block_count": block_count,
        "oversized_1000": oversized_1000,
        "raw_candidate_count": raw_candidate_count,
        "candidates_by_limit": candidates_by_limit,
        "blocks_by_limit": blocks_by_limit,
        "recovered_by_limit": recovered_by_limit,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("STEP 22 — FREQUENCY-AWARE BLOCKING BENCHMARK")
    print("=" * 72)

    start_time = time.time()

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load actual missed pairs.
    # --------------------------------------------------------

    total_missed_pairs, target_to_s1 = (
        load_missed_pairs()
    )

    if total_missed_pairs == 0:

        raise RuntimeError(
            "No missed pairs were loaded from "
            "step18_missed_sample.tsv"
        )

    # --------------------------------------------------------
    # Input files.
    # --------------------------------------------------------

    source_files = [
        ("S1", S1_FILE),
        ("S2", S2_FILE),
        ("S3", S3_FILE),
    ]

    report_lines = []

    report_lines.append(
        "STEP 22 — FREQUENCY-AWARE BLOCKING BENCHMARK"
    )

    report_lines.append(
        "=" * 72
    )

    report_lines.append(
        f"Missed pairs: {total_missed_pairs:,}"
    )

    report_lines.append(
        f"Block limits: {BLOCK_LIMITS}"
    )

    report_lines.append(
        "Sort workers: 1"
    )

    report_lines.append(
        f"Sort buffer: {SORT_BUFFER}"
    )

    report_lines.append("")

    # ========================================================
    # PROCESS ONE BLOCKING KEY AT A TIME
    # ========================================================

    for key_number, key_name in enumerate(
        BLOCKING_KEYS,
        start=1
    ):

        print()
        print("=" * 72)
        print(
            f"[{key_number}/{len(BLOCKING_KEYS)}] "
            f"{key_name}"
        )
        print("=" * 72)

        key_start = time.time()

        raw_file = (
            WORK_DIR
            / f"{key_name}.raw.tsv"
        )

        sorted_file = (
            WORK_DIR
            / f"{key_name}.sorted.tsv"
        )

        # Remove old files if present.
        if raw_file.exists():
            raw_file.unlink()

        if sorted_file.exists():
            sorted_file.unlink()

        total_scanned = 0
        total_written = 0

        # ----------------------------------------------------
        # Generate block records.
        # ----------------------------------------------------

        for source, input_file in source_files:

            print(
                f"Generating {source} records..."
            )

            scanned, written = (
                generate_block_records(
                    source,
                    input_file,
                    key_name,
                    raw_file
                )
            )

            total_scanned += scanned
            total_written += written

            print(
                f"  scanned: {scanned:,}"
            )

            print(
                f"  written: {written:,}"
            )

        # ----------------------------------------------------
        # Sort.
        # ----------------------------------------------------

        print(
            "Sorting with ONE CPU worker..."
        )

        sort_start = time.time()

        sort_block_file(
            raw_file,
            sorted_file
        )

        sort_elapsed = (
            time.time()
            - sort_start
        )

        print(
            f"Sort time: {sort_elapsed:.2f}s"
        )

        # Raw file no longer needed.
        raw_file.unlink()

        # ----------------------------------------------------
        # Analyze.
        # ----------------------------------------------------

        print(
            "Analyzing blocks..."
        )

        result = analyze_sorted_blocks(
            sorted_file,
            total_missed_pairs,
            target_to_s1
        )

        key_elapsed = (
            time.time()
            - key_start
        )

        print()
        print(
            f"Blocks: "
            f"{result['block_count']:,}"
        )

        print(
            f"Oversized >1000: "
            f"{result['oversized_1000']:,}"
        )

        print(
            f"Raw candidate estimate: "
            f"{result['raw_candidate_count']:,}"
        )

        print()
        print(
            "BLOCK LIMIT RESULTS"
        )

        report_lines.append(
            f"KEY: {key_name}"
        )

        report_lines.append(
            f"scanned={total_scanned}"
        )

        report_lines.append(
            f"written={total_written}"
        )

        report_lines.append(
            f"blocks={result['block_count']}"
        )

        report_lines.append(
            f"oversized_1000="
            f"{result['oversized_1000']}"
        )

        report_lines.append(
            f"raw_candidates="
            f"{result['raw_candidate_count']}"
        )

        for limit in BLOCK_LIMITS:

            blocks = result[
                "blocks_by_limit"
            ][limit]

            candidates = result[
                "candidates_by_limit"
            ][limit]

            recovered = len(
                result[
                    "recovered_by_limit"
                ][limit]
            )

            recall = (
                recovered
                / total_missed_pairs
                * 100.0
            )

            print(
                f"  <= {limit:4d}: "
                f"blocks={blocks:,} "
                f"candidates={candidates:,} "
                f"recovered={recovered:,} "
                f"sample_recall={recall:.2f}%"
            )

            report_lines.append(
                f"limit={limit}\t"
                f"blocks={blocks}\t"
                f"candidates={candidates}\t"
                f"recovered={recovered}\t"
                f"sample_recall={recall:.4f}"
            )

        report_lines.append(
            f"runtime={key_elapsed:.2f}s"
        )

        report_lines.append("")

        print(
            f"Key runtime: {key_elapsed:.2f}s"
        )

        # ----------------------------------------------------
        # Delete temporary sorted file immediately.
        # ----------------------------------------------------

        if sorted_file.exists():
            sorted_file.unlink()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    total_elapsed = (
        time.time()
        - start_time
    )

    report_lines.append(
        f"TOTAL RUNTIME: {total_elapsed:.2f}s"
    )

    REPORT_FILE.write_text(
        "\n".join(report_lines)
        + "\n",
        encoding="utf-8"
    )

    # Remove work directory if empty.
    try:
        WORK_DIR.rmdir()
    except OSError:
        pass

    print()
    print("=" * 72)
    print("STEP 22 COMPLETE")
    print("=" * 72)

    print(
        f"Report: {REPORT_FILE}"
    )

    print(
        f"Total runtime: "
        f"{total_elapsed:.2f}s"
    )

    print()
    print(
        "No candidate-pair file was generated."
    )


if __name__ == "__main__":
    main()