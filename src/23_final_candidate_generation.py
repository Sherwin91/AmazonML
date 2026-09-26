#!/usr/bin/env python3

"""
STEP 23 — FINAL COUNTRY-AWARE CANDIDATE GENERATION

Purpose
-------
Generate a practical, high-recall candidate set for TEST data.

Design
------
- Streams TSV files.
- Uses GNU sort on disk.
- ONE CPU worker.
- 128 MB sort buffer.
- No pandas.
- No Polars.
- No full-dataset in-memory structures.
- Processes one blocking key at a time.
- Uses country-aware compound blocking.
- Refuses oversized blocks.
- Refuses oversized blocking-key candidate sets.
- Deduplicates candidates externally.

Output
------
output/step23_unique_candidate_pairs.tsv

Format:
source1_entity_id    candidate_entity_id

This is an intermediate pair-per-line candidate file.

Step 24 will score these pairs with:
output/real_raw_match_model.joblib
"""


import csv
import re
import shutil
import subprocess
import time
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"

S1_FILE = TEST_DIR / "test_source1.tsv"
S2_FILE = TEST_DIR / "test_source2.tsv"
S3_FILE = TEST_DIR / "test_source3.tsv"

WORK_DIR = OUTPUT_DIR / "step23_work"

FINAL_PAIRS = OUTPUT_DIR / "step23_unique_candidate_pairs.tsv"

REPORT_FILE = OUTPUT_DIR / "step23_candidate_report.txt"


# ============================================================
# RESOURCE LIMITS
# ============================================================

# Your laptop has limited CPU.
SORT_PARALLEL = "1"

# Keep GNU sort conservative.
SORT_BUFFER = "128M"

# Maximum candidate pairs allowed from ONE blocking block.
MAX_CANDIDATES_PER_BLOCK = 50_000

# Protect the machine from enormous key-level outputs.
MAX_CANDIDATES_PER_KEY = 20_000_000

# Overall protection.
MAX_TOTAL_CANDIDATES = 60_000_000

# If a single source has an absurdly large block, don't hold
# that many IDs in Python memory.
MAX_RECORDS_PER_SOURCE_PER_BLOCK = 5_000


# ============================================================
# BLOCKING KEYS
#
# Ordered from selective compound keys toward broader keys.
# ============================================================

BLOCKING_KEYS = [
    "country_name4_address4",
    "country_first_name3_address3",
    "country_first_name3_first_address3",
    "country_name5_address3",
    "country_name5_digits2",
    "country_longest_name4_digits2",
    "country_first_name_first_address",
    "country_name_prefix7",
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
# COLUMN HELPERS
# ============================================================

def find_column(header, candidates):
    lookup = {
        value.strip().lower(): index
        for index, value in enumerate(header)
    }

    for candidate in candidates:
        index = lookup.get(candidate.lower())

        if index is not None:
            return index

    return None


# ============================================================
# BLOCKING KEY CREATION
# ============================================================

def make_blocking_key(
    key_name,
    country,
    business_name,
    business_address
):
    country_clean = compact(country)

    if not country_clean:
        return ""

    name_clean = clean_text(business_name)
    address_clean = clean_text(business_address)

    name_compact = compact(business_name)
    address_compact = compact(business_address)

    name_tokens = get_tokens(business_name)
    address_tokens = get_tokens(business_address)

    # --------------------------------------------------------
    # country + first 4 name chars + first 4 address chars
    # --------------------------------------------------------

    if key_name == "country_name4_address4":

        if (
            len(name_compact) >= 4
            and len(address_compact) >= 4
        ):
            return (
                country_clean
                + "|"
                + name_compact[:4]
                + "|"
                + address_compact[:4]
            )

        return ""

    # --------------------------------------------------------
    # country + first name token 3 chars + address 3 chars
    # --------------------------------------------------------

    if key_name == "country_first_name3_address3":

        if not name_tokens:
            return ""

        first_name = name_tokens[0]

        if len(first_name) < 3:
            return ""

        if len(address_compact) < 3:
            return ""

        return (
            country_clean
            + "|"
            + first_name[:3]
            + "|"
            + address_compact[:3]
        )

    # --------------------------------------------------------
    # country + first name token 3 + first address token 3
    # --------------------------------------------------------

    if key_name == "country_first_name3_first_address3":

        if not name_tokens:
            return ""

        if not address_tokens:
            return ""

        first_name = name_tokens[0]
        first_address = address_tokens[0]

        if len(first_name) < 3:
            return ""

        if len(first_address) < 3:
            return ""

        return (
            country_clean
            + "|"
            + first_name[:3]
            + "|"
            + first_address[:3]
        )

    # --------------------------------------------------------
    # country + name 5 + address 3
    # --------------------------------------------------------

    if key_name == "country_name5_address3":

        if (
            len(name_compact) >= 5
            and len(address_compact) >= 3
        ):
            return (
                country_clean
                + "|"
                + name_compact[:5]
                + "|"
                + address_compact[:3]
            )

        return ""

    # --------------------------------------------------------
    # country + name 5 + first 2 address digits
    # --------------------------------------------------------

    if key_name == "country_name5_digits2":

        digits = get_digits(business_address)

        if (
            len(name_compact) >= 5
            and len(digits) >= 2
        ):
            return (
                country_clean
                + "|"
                + name_compact[:5]
                + "|"
                + digits[:2]
            )

        return ""

    # --------------------------------------------------------
    # country + longest name token 4 + address digits 2
    # --------------------------------------------------------

    if key_name == "country_longest_name4_digits2":

        longest_name = longest_token(
            business_name,
            4
        )

        digits = get_digits(business_address)

        if (
            len(longest_name) >= 4
            and len(digits) >= 2
        ):
            return (
                country_clean
                + "|"
                + longest_name[:4]
                + "|"
                + digits[:2]
            )

        return ""

    # --------------------------------------------------------
    # country + first name token + first address token
    # --------------------------------------------------------

    if key_name == "country_first_name_first_address":

        if not name_tokens:
            return ""

        if not address_tokens:
            return ""

        return (
            country_clean
            + "|"
            + name_tokens[0][:4]
            + "|"
            + address_tokens[0][:4]
        )

    # --------------------------------------------------------
    # country + name prefix 7
    # --------------------------------------------------------

    if key_name == "country_name_prefix7":

        if len(name_compact) >= 7:

            return (
                country_clean
                + "|"
                + name_compact[:7]
            )

        return ""

    return ""


# ============================================================
# GENERATE BLOCK RECORDS
# ============================================================

def generate_block_records(
    source,
    input_file,
    key_name,
    raw_block_file
):
    """
    Generate:

        blocking_key<TAB>source<TAB>entity_id

    Streaming only.
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

        country_index = find_column(
            header,
            ["country"]
        )

        if id_index is None:
            raise RuntimeError(
                f"entity_id column not found in "
                f"{input_file}"
            )

        if country_index is None:
            raise RuntimeError(
                f"country column not found in "
                f"{input_file}"
            )

        with open(
            raw_block_file,
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

                country = row[country_index].strip()

                blocking_key = make_blocking_key(
                    key_name,
                    country,
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
# GNU SORT
# ============================================================

def sort_block_records(
    raw_file,
    sorted_file
):
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
        str(raw_file),
        "-o",
        str(sorted_file),
    ]

    subprocess.run(
        command,
        check=True
    )


def sort_candidate_pairs(
    input_file,
    output_file
):
    command = [
        "sort",
        "-t",
        "\t",
        "-k",
        "1,1",
        "-k",
        "2,2",
        "-S",
        SORT_BUFFER,
        "--parallel=1",
        "-T",
        str(WORK_DIR),
        str(input_file),
        "-o",
        str(output_file),
    ]

    subprocess.run(
        command,
        check=True
    )


# ============================================================
# PROCESS ONE BLOCK
# ============================================================

def process_block(
    s1_ids,
    s2_ids,
    s3_ids,
    candidate_output
):
    """
    Materialize candidate pairs only when the block is small.

    Returns:
        generated_count
        skipped
    """

    n1 = len(s1_ids)
    n2 = len(s2_ids)
    n3 = len(s3_ids)

    if n1 == 0:
        return 0, False

    candidate_count = (
        n1 * n2
        + n1 * n3
    )

    # --------------------------------------------------------
    # Safety condition 1:
    # excessively large source-side block
    # --------------------------------------------------------

    if (
        n1 > MAX_RECORDS_PER_SOURCE_PER_BLOCK
        or n2 > MAX_RECORDS_PER_SOURCE_PER_BLOCK
        or n3 > MAX_RECORDS_PER_SOURCE_PER_BLOCK
    ):
        return 0, True

    # --------------------------------------------------------
    # Safety condition 2:
    # excessively large Cartesian product
    # --------------------------------------------------------

    if candidate_count > MAX_CANDIDATES_PER_BLOCK:
        return 0, True

    generated = 0

    with open(
        candidate_output,
        "a",
        encoding="utf-8"
    ) as output_handle:

        for s1_id in s1_ids:

            for s2_id in s2_ids:

                output_handle.write(
                    s1_id
                    + "\t"
                    + s2_id
                    + "\n"
                )

                generated += 1

            for s3_id in s3_ids:

                output_handle.write(
                    s1_id
                    + "\t"
                    + s3_id
                    + "\n"
                )

                generated += 1

    return generated, False


# ============================================================
# ANALYZE ONE SORTED BLOCK FILE
# ============================================================

def generate_candidates_from_sorted_blocks(
    sorted_file,
    candidate_output
):
    """
    Read sorted blocking records and generate candidates.
    """

    block_count = 0
    skipped_blocks = 0
    generated_candidates = 0

    current_key = None

    s1_ids = []
    s2_ids = []
    s3_ids = []

    def finish_block():

        nonlocal block_count
        nonlocal skipped_blocks
        nonlocal generated_candidates

        if current_key is None:
            return

        block_count += 1

        generated, skipped = process_block(
            s1_ids,
            s2_ids,
            s3_ids,
            candidate_output
        )

        if skipped:
            skipped_blocks += 1
        else:
            generated_candidates += generated

    with open(
        sorted_file,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as input_handle:

        for line in input_handle:

            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                continue

            key = parts[0]
            source = parts[1]
            entity_id = parts[2]

            if current_key is None:
                current_key = key

            if key != current_key:

                finish_block()

                current_key = key

                s1_ids = []
                s2_ids = []
                s3_ids = []

            if source == "S1":
                s1_ids.append(entity_id)

            elif source == "S2":
                s2_ids.append(entity_id)

            elif source == "S3":
                s3_ids.append(entity_id)

    finish_block()

    return (
        block_count,
        skipped_blocks,
        generated_candidates
    )


# ============================================================
# MERGE SORTED CANDIDATE FILES
# ============================================================

def merge_candidate_files(
    sorted_files,
    final_output
):
    """
    Merge already-sorted pair files and deduplicate them.

    Uses GNU sort -m -u.
    """

    if not sorted_files:
        raise RuntimeError(
            "No candidate files were produced."
        )

    command = [
        "sort",
        "-m",
        "-u",
        "-t",
        "\t",
        "-k",
        "1,1",
        "-k",
        "2,2",
    ]

    command.extend(
        str(path)
        for path in sorted_files
    )

    merged_body = WORK_DIR / "merged_candidates.tsv"

    command.extend(
        [
            "-o",
            str(merged_body)
        ]
    )

    subprocess.run(
        command,
        check=True
    )

    # Add header.
    with open(
        final_output,
        "w",
        encoding="utf-8"
    ) as output_handle:

        output_handle.write(
            "source1_entity_id\tcandidate_entity_id\n"
        )

        with open(
            merged_body,
            "r",
            encoding="utf-8",
            errors="replace"
        ) as input_handle:

            for line in input_handle:
                output_handle.write(line)

    merged_body.unlink()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("STEP 23 — FINAL COUNTRY-AWARE CANDIDATE GENERATION")
    print("=" * 72)

    start_time = time.time()

    # --------------------------------------------------------
    # Check files.
    # --------------------------------------------------------

    required_files = [
        S1_FILE,
        S2_FILE,
        S3_FILE,
    ]

    for file_path in required_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Missing file:\n{file_path}"
            )

    # --------------------------------------------------------
    # Work directory.
    # --------------------------------------------------------

    if WORK_DIR.exists():

        print(
            "Removing old Step 23 work directory..."
        )

        shutil.rmtree(WORK_DIR)

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Report.
    # --------------------------------------------------------

    report_lines = []

    report_lines.append(
        "STEP 23 — FINAL COUNTRY-AWARE CANDIDATE GENERATION"
    )

    report_lines.append(
        "=" * 72
    )

    report_lines.append(
        f"MAX_CANDIDATES_PER_BLOCK="
        f"{MAX_CANDIDATES_PER_BLOCK}"
    )

    report_lines.append(
        f"MAX_CANDIDATES_PER_KEY="
        f"{MAX_CANDIDATES_PER_KEY}"
    )

    report_lines.append(
        f"MAX_TOTAL_CANDIDATES="
        f"{MAX_TOTAL_CANDIDATES}"
    )

    report_lines.append(
        f"SORT_BUFFER={SORT_BUFFER}"
    )

    report_lines.append(
        "SORT_PARALLEL=1"
    )

    report_lines.append("")

    # --------------------------------------------------------
    # Candidate files.
    # --------------------------------------------------------

    sorted_candidate_files = []

    total_generated = 0

    # --------------------------------------------------------
    # Process each blocking key.
    # --------------------------------------------------------

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

        raw_block_file = (
            WORK_DIR
            / f"{key_name}.blocks.tsv"
        )

        sorted_block_file = (
            WORK_DIR
            / f"{key_name}.blocks.sorted.tsv"
        )

        raw_candidate_file = (
            WORK_DIR
            / f"{key_name}.candidates.tsv"
        )

        sorted_candidate_file = (
            WORK_DIR
            / f"{key_name}.candidates.sorted.tsv"
        )

        # Remove stale files.
        for file_path in [
            raw_block_file,
            sorted_block_file,
            raw_candidate_file,
            sorted_candidate_file,
        ]:
            if file_path.exists():
                file_path.unlink()

        # ----------------------------------------------------
        # Generate block records.
        # ----------------------------------------------------

        total_scanned = 0
        total_written = 0

        for source, input_file in [
            ("S1", S1_FILE),
            ("S2", S2_FILE),
            ("S3", S3_FILE),
        ]:

            print(
                f"Generating {source} blocking records..."
            )

            scanned, written = (
                generate_block_records(
                    source,
                    input_file,
                    key_name,
                    raw_block_file
                )
            )

            total_scanned += scanned
            total_written += written

            print(
                f"  scanned = {scanned:,}"
            )

            print(
                f"  written = {written:,}"
            )

        # ----------------------------------------------------
        # Sort block records.
        # ----------------------------------------------------

        print(
            "Sorting with ONE CPU worker..."
        )

        sort_start = time.time()

        sort_block_records(
            raw_block_file,
            sorted_block_file
        )

        sort_elapsed = (
            time.time()
            - sort_start
        )

        print(
            f"Sort time = {sort_elapsed:.2f}s"
        )

        raw_block_file.unlink()

        # ----------------------------------------------------
        # Generate candidates.
        # ----------------------------------------------------

        print(
            "Generating candidate pairs..."
        )

        (
            block_count,
            skipped_blocks,
            generated_candidates,
        ) = generate_candidates_from_sorted_blocks(
            sorted_block_file,
            raw_candidate_file
        )

        sorted_block_file.unlink()

        print(
            f"Blocks = {block_count:,}"
        )

        print(
            f"Skipped oversized blocks = "
            f"{skipped_blocks:,}"
        )

        print(
            f"Generated candidates = "
            f"{generated_candidates:,}"
        )

        # ----------------------------------------------------
        # Reject an excessively large key.
        # ----------------------------------------------------

        if generated_candidates > MAX_CANDIDATES_PER_KEY:

            print(
                "KEY TOO LARGE — discarding this key."
            )

            report_lines.append(
                f"{key_name}\t"
                f"scanned={total_scanned}\t"
                f"written={total_written}\t"
                f"blocks={block_count}\t"
                f"skipped={skipped_blocks}\t"
                f"generated={generated_candidates}\t"
                f"ACCEPTED=NO_TOO_LARGE"
            )

            if raw_candidate_file.exists():
                raw_candidate_file.unlink()

            continue

        # ----------------------------------------------------
        # Protect total candidate volume.
        # ----------------------------------------------------

        if (
            total_generated
            + generated_candidates
            > MAX_TOTAL_CANDIDATES
        ):

            print(
                "GLOBAL CANDIDATE BUDGET WOULD BE "
                "EXCEEDED."
            )

            print(
                "Discarding this key."
            )

            report_lines.append(
                f"{key_name}\t"
                f"generated={generated_candidates}\t"
                f"ACCEPTED=NO_GLOBAL_BUDGET"
            )

            if raw_candidate_file.exists():
                raw_candidate_file.unlink()

            continue

        # ----------------------------------------------------
        # Sort this key's candidate pairs.
        # ----------------------------------------------------

        print(
            "Sorting candidate pairs..."
        )

        sort_candidate_pairs(
            raw_candidate_file,
            sorted_candidate_file
        )

        raw_candidate_file.unlink()

        sorted_candidate_files.append(
            sorted_candidate_file
        )

        total_generated += generated_candidates

        key_elapsed = (
            time.time()
            - key_start
        )

        print(
            f"Accepted key."
        )

        print(
            f"Running total = "
            f"{total_generated:,}"
        )

        print(
            f"Key runtime = "
            f"{key_elapsed:.2f}s"
        )

        report_lines.append(
            f"{key_name}\t"
            f"scanned={total_scanned}\t"
            f"written={total_written}\t"
            f"blocks={block_count}\t"
            f"skipped={skipped_blocks}\t"
            f"generated={generated_candidates}\t"
            f"running_total={total_generated}\t"
            f"ACCEPTED=YES\t"
            f"runtime={key_elapsed:.2f}"
        )

        # ----------------------------------------------------
        # Stop adding keys if budget is exhausted.
        # ----------------------------------------------------

        if total_generated >= MAX_TOTAL_CANDIDATES:

            print()
            print(
                "Global candidate budget reached."
            )

            break

    # --------------------------------------------------------
    # Merge and deduplicate.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("MERGING AND DEDUPLICATING")
    print("=" * 72)

    if not sorted_candidate_files:

        raise RuntimeError(
            "No blocking key produced an acceptable "
            "candidate file."
        )

    merge_start = time.time()

    merge_candidate_files(
        sorted_candidate_files,
        FINAL_PAIRS
    )

    merge_elapsed = (
        time.time()
        - merge_start
    )

    print(
        f"Merge time = {merge_elapsed:.2f}s"
    )

    # --------------------------------------------------------
    # Count final unique pairs.
    # --------------------------------------------------------

    final_pairs = 0

    with open(
        FINAL_PAIRS,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        next(file, None)

        for _ in file:
            final_pairs += 1

    # --------------------------------------------------------
    # Cleanup.
    # --------------------------------------------------------

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)

    total_elapsed = (
        time.time()
        - start_time
    )

    report_lines.append("")
    report_lines.append(
        f"RAW_GENERATED_BEFORE_DEDUP="
        f"{total_generated}"
    )

    report_lines.append(
        f"FINAL_UNIQUE_PAIRS="
        f"{final_pairs}"
    )

    report_lines.append(
        f"TOTAL_RUNTIME="
        f"{total_elapsed:.2f}s"
    )

    report_lines.append(
        f"FINAL_OUTPUT="
        f"{FINAL_PAIRS}"
    )

    REPORT_FILE.write_text(
        "\n".join(report_lines)
        + "\n",
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final output.
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 23 COMPLETE")
    print("=" * 72)

    print(
        f"Unique candidate pairs: "
        f"{final_pairs:,}"
    )

    print(
        f"Candidate file: "
        f"{FINAL_PAIRS}"
    )

    print(
        f"Report: "
        f"{REPORT_FILE}"
    )

    print(
        f"Total runtime: "
        f"{total_elapsed:.2f}s"
    )

    print()
    print(
        "Next stage: score these candidates "
        "with the existing Step 14 model."
    )


if __name__ == "__main__":
    main()
