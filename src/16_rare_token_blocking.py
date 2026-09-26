#!/usr/bin/env python3

"""
STEP 16 — FREQUENCY-AWARE RARE-TOKEN BLOCKING

Goal:
    Improve blocking recall beyond Step 15's 57.65% while avoiding
    the enormous candidate explosion caused by address_digits.

Design:
    1. Scan ALL train S1/S2/S3.
    2. Extract meaningful name tokens.
    3. Build country + token occurrence file.
    4. GNU-sort occurrences externally.
    5. Count frequencies externally.
    6. Keep only informative tokens.
    7. Re-scan records and emit only rare-token blocking entries.
    8. Sort blocking entries.
    9. Generate S1 -> S2/S3 candidates.
   10. Compare against ALL ground-truth pairs.
   11. Report recall.

RAM-safe:
    - No full dataset in RAM.
    - No Python dictionary containing millions of tokens.
    - Uses GNU sort with bounded memory.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR = ROOT / "output"
TMP_DIR = OUTPUT_DIR / "step16_tmp"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

REPORT_FILE = OUTPUT_DIR / "step16_rare_token_recall_report.txt"
CANDIDATE_FILE = OUTPUT_DIR / "step16_candidates.tsv"

SORT_MEMORY = "256M"
SORT_THREADS = "2"

# ------------------------------------------------------------
# Blocking parameters
# ------------------------------------------------------------

# A token occurring more than this number of records in a
# country is considered too common to be a useful blocking key.
MAX_TOKEN_FREQUENCY = 250

# Ignore tiny tokens.
MIN_TOKEN_LENGTH = 4

# Maximum number of useful tokens emitted from one record.
MAX_TOKENS_PER_RECORD = 4

# Prevent one pathological token from producing a massive
# Cartesian product.
MAX_BLOCK_SIZE = 1500

# Tokens that are generally not discriminative.
STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "restaurant",
    "restaurants",
    "hotel",
    "hotels",
    "company",
    "companies",
    "corporation",
    "corporations",
    "limited",
    "private",
    "public",
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "plc",
    "pvt",
    "co",
    "corp",
    "corporate",
    "group",
    "groups",
    "services",
    "service",
    "center",
    "centre",
    "international",
    "india",
    "indian",
    "america",
    "american",
    "usa",
    "united",
    "states",
}


# ============================================================
# HELPERS
# ============================================================

def log(msg: str = "") -> None:
    print(msg, flush=True)


def run_cmd(cmd: list[str], stdout=None, stdin=None) -> None:
    log("  $ " + " ".join(str(x) for x in cmd))

    result = subprocess.run(
        cmd,
        stdout=stdout,
        stdin=stdin,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}):\n"
            f"{' '.join(str(x) for x in cmd)}\n\n"
            f"{result.stderr}"
        )


def sort_file(
    input_file: Path,
    output_file: Path,
    keys: list[str] | None = None,
) -> None:
    """
    GNU external sort.

    keys examples:
        ["1,1"]
        ["1,1", "2,2"]
    """

    cmd = [
        "sort",
        f"--buffer-size={SORT_MEMORY}",
        f"--parallel={SORT_THREADS}",
        "--stable",
    ]

    if keys:
        for key in keys:
            cmd.extend(["-k", key])

    cmd.extend([
        str(input_file),
        "-o",
        str(output_file),
    ])

    run_cmd(cmd)


def check_tools() -> None:
    if shutil.which("sort") is None:
        raise RuntimeError("GNU sort was not found.")

    if shutil.which("uniq") is None:
        raise RuntimeError("GNU uniq was not found.")


def normalize_token(token: str) -> str:
    token = token.lower()

    # Keep unicode letters/numbers.
    token = re.sub(r"[^\w]+", "", token, flags=re.UNICODE)

    return token


def extract_tokens(text: str) -> list[str]:
    if not text:
        return []

    raw = re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    result = []

    for token in raw:
        token = normalize_token(token)

        if len(token) < MIN_TOKEN_LENGTH:
            continue

        if token in STOPWORDS:
            continue

        if token.isdigit():
            continue

        result.append(token)

    # Unique tokens.
    result = list(dict.fromkeys(result))

    return result


def select_tokens(text: str) -> list[str]:
    """
    We do not yet know token frequencies during the first scan,
    so this only performs lexical filtering.

    The second scan filters using the frequency table.
    """

    return extract_tokens(text)


# ============================================================
# TSV SCANNING
# ============================================================

def scan_token_occurrences(
    source_name: str,
    path: Path,
    output_file,
) -> int:

    count = 0
    occurrence_count = 0

    log(f"  Reading {source_name}: {path.name}")

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            count += 1

            entity_id = row.get("entity_id", "")
            country = (row.get("country") or "").strip().lower()
            name = row.get("business_name") or ""

            if not entity_id or not country:
                continue

            tokens = select_tokens(name)

            for token in tokens:
                key = f"{country}|{token}"

                output_file.write(
                    f"{key}\t{source_name}\t{entity_id}\n"
                )

                occurrence_count += 1

            if count % 500_000 == 0:
                log(f"    rows processed: {count:,}")

    log(
        f"  {source_name} complete: "
        f"{count:,} rows, "
        f"{occurrence_count:,} token occurrences"
    )

    return count


# ============================================================
# BUILD TOKEN OCCURRENCES
# ============================================================

def build_occurrences() -> tuple[Path, Path]:

    raw_file = TMP_DIR / "token_occurrences_raw.tsv"
    sorted_file = TMP_DIR / "token_occurrences_sorted.tsv"

    if raw_file.exists():
        raw_file.unlink()

    if sorted_file.exists():
        sorted_file.unlink()

    log()
    log("=" * 72)
    log("BUILDING TOKEN OCCURRENCES")
    log("=" * 72)

    start = time.time()

    with raw_file.open(
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as out:

        scan_token_occurrences("S1", S1_FILE, out)
        scan_token_occurrences("S2", S2_FILE, out)
        scan_token_occurrences("S3", S3_FILE, out)

    log()
    log("Sorting token occurrences...")

    sort_file(
        raw_file,
        sorted_file,
        keys=["1,1", "2,2", "3,3"],
    )

    elapsed = time.time() - start

    log(f"Token occurrence file: {sorted_file}")
    log(f"Time: {elapsed:.2f} sec")

    return raw_file, sorted_file


# ============================================================
# FREQUENCY TABLE
# ============================================================

def build_frequency_file(sorted_occurrences: Path) -> Path:

    frequency_file = TMP_DIR / "token_frequencies.tsv"

    if frequency_file.exists():
        frequency_file.unlink()

    log()
    log("=" * 72)
    log("CALCULATING TOKEN FREQUENCIES")
    log("=" * 72)

    start = time.time()

    current_key = None
    current_count = 0
    unique_keys = 0
    rare_keys = 0

    with sorted_occurrences.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as src, frequency_file.open(
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as out:

        for line in src:
            parts = line.rstrip("\n").split("\t", 1)

            if len(parts) != 2:
                continue

            key = parts[0]

            if current_key is None:
                current_key = key
                current_count = 1

            elif key == current_key:
                current_count += 1

            else:
                unique_keys += 1

                if current_count <= MAX_TOKEN_FREQUENCY:
                    out.write(
                        f"{current_key}\t{current_count}\n"
                    )
                    rare_keys += 1

                current_key = key
                current_count = 1

        if current_key is not None:
            unique_keys += 1

            if current_count <= MAX_TOKEN_FREQUENCY:
                out.write(
                    f"{current_key}\t{current_count}\n"
                )
                rare_keys += 1

    elapsed = time.time() - start

    log(f"Unique country-token keys: {unique_keys:,}")
    log(
        f"Informative keys <= {MAX_TOKEN_FREQUENCY}: "
        f"{rare_keys:,}"
    )
    log(f"Time: {elapsed:.2f} sec")

    return frequency_file


# ============================================================
# LOAD ONLY RARE KEYS
# ============================================================

def load_rare_keys(frequency_file: Path) -> dict[str, int]:

    """
    Important:
        We expect the rare-token table to be substantially smaller
        than the full occurrence table.

    This is the only dictionary in this script.
    """

    log()
    log("=" * 72)
    log("LOADING RARE TOKEN TABLE")
    log("=" * 72)

    start = time.time()

    rare = {}

    with frequency_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            parts = line.rstrip("\n").split("\t")

            if len(parts) != 2:
                continue

            key = parts[0]

            try:
                freq = int(parts[1])
            except ValueError:
                continue

            rare[key] = freq

    elapsed = time.time() - start

    log(f"Rare country-token keys loaded: {len(rare):,}")
    log(f"Time: {elapsed:.2f} sec")

    return rare


# ============================================================
# SECOND PASS — BUILD RARE BLOCK ENTRIES
# ============================================================

def build_block_entries(
    source_name: str,
    path: Path,
    rare_keys: dict[str, int],
    output_file,
) -> int:

    rows = 0
    emitted = 0

    log(f"  Reading {source_name}: {path.name}")

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            rows += 1

            entity_id = row.get("entity_id", "")
            country = (row.get("country") or "").strip().lower()
            name = row.get("business_name") or ""

            if not entity_id or not country:
                continue

            tokens = select_tokens(name)

            matches = []

            for token in tokens:
                key = f"{country}|{token}"

                if key in rare_keys:
                    matches.append((rare_keys[key], token))

            if not matches:
                continue

            # Prefer the rarest tokens.
            matches.sort(key=lambda x: x[0])

            used = set()

            for freq, token in matches:

                if token in used:
                    continue

                used.add(token)

                block_key = f"{country}|{token}"

                output_file.write(
                    f"{block_key}\t{source_name}\t{entity_id}\n"
                )

                emitted += 1

                if len(used) >= MAX_TOKENS_PER_RECORD:
                    break

            if rows % 500_000 == 0:
                log(f"    rows processed: {rows:,}")

    log(
        f"  {source_name} complete: "
        f"{rows:,} rows, "
        f"{emitted:,} rare-token entries"
    )

    return emitted


def build_block_entries_file(
    rare_keys: dict[str, int],
) -> Path:

    raw = TMP_DIR / "rare_block_entries_raw.tsv"
    sorted_file = TMP_DIR / "rare_block_entries_sorted.tsv"

    if raw.exists():
        raw.unlink()

    if sorted_file.exists():
        sorted_file.unlink()

    log()
    log("=" * 72)
    log("BUILDING RARE-TOKEN BLOCK ENTRIES")
    log("=" * 72)

    start = time.time()

    with raw.open(
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as out:

        build_block_entries(
            "S1",
            S1_FILE,
            rare_keys,
            out,
        )

        build_block_entries(
            "S2",
            S2_FILE,
            rare_keys,
            out,
        )

        build_block_entries(
            "S3",
            S3_FILE,
            rare_keys,
            out,
        )

    log()
    log("Sorting rare-token block entries...")

    sort_file(
        raw,
        sorted_file,
        keys=["1,1", "2,2", "3,3"],
    )

    elapsed = time.time() - start

    log(f"Sorted block entries: {sorted_file}")
    log(f"Time: {elapsed:.2f} sec")

    return sorted_file


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    sorted_entries: Path,
) -> tuple[Path, int, int]:

    candidate_raw = TMP_DIR / "rare_candidates_raw.tsv"
    candidate_sorted = TMP_DIR / "rare_candidates_sorted.tsv"

    if candidate_raw.exists():
        candidate_raw.unlink()

    if candidate_sorted.exists():
        candidate_sorted.unlink()

    log()
    log("=" * 72)
    log("GENERATING RARE-TOKEN CANDIDATES")
    log("=" * 72)

    start = time.time()

    # Current block.
    current_key = None

    s1_ids: list[str] = []
    s2_ids: list[str] = []
    s3_ids: list[str] = []

    blocks_examined = 0
    blocks_skipped = 0
    raw_candidates = 0

    def flush_block(out) -> None:
        nonlocal blocks_examined
        nonlocal blocks_skipped
        nonlocal raw_candidates

        if current_key is None:
            return

        total_targets = len(s2_ids) + len(s3_ids)

        if not s1_ids or total_targets == 0:
            return

        blocks_examined += 1

        if len(s1_ids) > MAX_BLOCK_SIZE:
            blocks_skipped += 1
            return

        if len(s2_ids) > MAX_BLOCK_SIZE:
            blocks_skipped += 1
            return

        if len(s3_ids) > MAX_BLOCK_SIZE:
            blocks_skipped += 1
            return

        for s1 in s1_ids:

            for s2 in s2_ids:
                out.write(
                    f"{s1}\t{s2}\n"
                )
                raw_candidates += 1

            for s3 in s3_ids:
                out.write(
                    f"{s1}\t{s3}\n"
                )
                raw_candidates += 1

    with sorted_entries.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as src, candidate_raw.open(
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as out:

        for line in src:

            parts = line.rstrip("\n").split("\t")

            if len(parts) != 3:
                continue

            block_key, source, entity_id = parts

            if current_key is None:
                current_key = block_key

            if block_key != current_key:

                flush_block(out)

                current_key = block_key

                s1_ids = []
                s2_ids = []
                s3_ids = []

            if source == "S1":
                s1_ids.append(entity_id)

            elif source == "S2":
                s2_ids.append(entity_id)

            elif source == "S3":
                s3_ids.append(entity_id)

        flush_block(out)

    log()
    log("Sorting candidate pairs...")

    sort_file(
        candidate_raw,
        candidate_sorted,
        keys=["1,1", "2,2"],
    )

    elapsed = time.time() - start

    log(f"Raw candidate pairs: {raw_candidates:,}")
    log(f"Blocks examined: {blocks_examined:,}")
    log(f"Blocks skipped: {blocks_skipped:,}")
    log(f"Time: {elapsed:.2f} sec")

    return candidate_sorted, raw_candidates, blocks_skipped


# ============================================================
# GROUND TRUTH
# ============================================================

def create_ground_truth_pairs() -> Path:

    gt_sorted = TMP_DIR / "ground_truth_pairs_sorted.tsv"

    if gt_sorted.exists():
        gt_sorted.unlink()

    log()
    log("=" * 72)
    log("CREATING SORTED GROUND-TRUTH PAIRS")
    log("=" * 72)

    raw = TMP_DIR / "ground_truth_pairs_raw.tsv"

    if raw.exists():
        raw.unlink()

    count = 0

    with GT_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as src, raw.open(
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as out:

        reader = csv.DictReader(src, delimiter="\t")

        for row in reader:

            s1 = row.get("source1_entity_id", "")
            matched = row.get("matched_entity_ids", "")

            if not s1:
                continue

            matched = matched.strip()

            if not matched:
                continue

            for target in matched.split(","):

                target = target.strip()

                if target:
                    out.write(
                        f"{s1}\t{target}\n"
                    )

                    count += 1

    sort_file(
        raw,
        gt_sorted,
        keys=["1,1", "2,2"],
    )

    log(f"Ground-truth pairs: {count:,}")

    return gt_sorted


# ============================================================
# RECALL
# ============================================================

def calculate_recall(
    candidate_file: Path,
    gt_file: Path,
) -> tuple[int, int, float]:

    log()
    log("=" * 72)
    log("CALCULATING BLOCKING RECALL")
    log("=" * 72)

    # Because both files are sorted by S1 + target, perform a
    # streaming merge instead of loading either file into RAM.

    recovered = 0
    gt_total = 0

    candidate_line = None
    candidate_file_handle = candidate_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
    )

    def read_candidate():
        line = candidate_file_handle.readline()

        if not line:
            return None

        return line.rstrip("\n")

    candidate_line = read_candidate()

    with gt_file.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as gt:

        for line in gt:

            gt_line = line.rstrip("\n")

            if not gt_line:
                continue

            gt_total += 1

            while (
                candidate_line is not None
                and candidate_line < gt_line
            ):
                candidate_line = read_candidate()

            if candidate_line == gt_line:
                recovered += 1

    candidate_file_handle.close()

    recall = (
        recovered / gt_total
        if gt_total
        else 0.0
    )

    return gt_total, recovered, recall


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    overall_start = time.time()

    log("=" * 72)
    log("STEP 16 — FREQUENCY-AWARE RARE-TOKEN BLOCKING")
    log("=" * 72)

    log()
    log(f"Project: {ROOT}")
    log(f"Temporary directory: {TMP_DIR}")

    log()
    log("RAM-safe configuration:")
    log(f"  sort memory: {SORT_MEMORY}")
    log(f"  sort threads: {SORT_THREADS}")
    log(f"  max token frequency: {MAX_TOKEN_FREQUENCY}")
    log(f"  max block size: {MAX_BLOCK_SIZE}")
    log(f"  min token length: {MIN_TOKEN_LENGTH}")
    log(f"  max tokens / record: {MAX_TOKENS_PER_RECORD}")

    check_tools()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fresh temp directory.
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    TMP_DIR.mkdir(parents=True)

    # --------------------------------------------------------
    # 1. Ground truth
    # --------------------------------------------------------

    gt_file = create_ground_truth_pairs()

    # --------------------------------------------------------
    # 2. Token frequency analysis
    # --------------------------------------------------------

    raw_occurrences, sorted_occurrences = build_occurrences()

    frequency_file = build_frequency_file(
        sorted_occurrences
    )

    # --------------------------------------------------------
    # 3. Load only rare keys
    # --------------------------------------------------------

    rare_keys = load_rare_keys(
        frequency_file
    )

    # --------------------------------------------------------
    # 4. Build blocks
    # --------------------------------------------------------

    sorted_entries = build_block_entries_file(
        rare_keys
    )

    # Release dictionary before candidate processing.
    del rare_keys

    # --------------------------------------------------------
    # 5. Generate candidates
    # --------------------------------------------------------

    candidate_sorted, raw_candidates, skipped_blocks = (
        generate_candidates(sorted_entries)
    )

    # --------------------------------------------------------
    # 6. Calculate recall
    # --------------------------------------------------------

    gt_total, recovered, recall = calculate_recall(
        candidate_sorted,
        gt_file,
    )

    # Copy final candidate file.
    if CANDIDATE_FILE.exists():
        CANDIDATE_FILE.unlink()

    shutil.copyfile(
        candidate_sorted,
        CANDIDATE_FILE,
    )

    elapsed = time.time() - overall_start

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = f"""
STEP 16 — FREQUENCY-AWARE RARE-TOKEN BLOCKING

Configuration
-------------
MAX_TOKEN_FREQUENCY = {MAX_TOKEN_FREQUENCY}
MAX_BLOCK_SIZE = {MAX_BLOCK_SIZE}
MIN_TOKEN_LENGTH = {MIN_TOKEN_LENGTH}
MAX_TOKENS_PER_RECORD = {MAX_TOKENS_PER_RECORD}

Ground-truth true pairs:
    {gt_total:,}

Raw candidate pairs:
    {raw_candidates:,}

Recovered true pairs:
    {recovered:,}

BLOCKING RECALL:
    {recall * 100:.4f}%

Blocks skipped:
    {skipped_blocks:,}

Total runtime:
    {elapsed:.2f} seconds
"""

    REPORT_FILE.write_text(
        report.strip() + "\n",
        encoding="utf-8",
    )

    log()
    log("=" * 72)
    log("STEP 16 — RESULT")
    log("=" * 72)

    log(f"Ground-truth true pairs: {gt_total:,}")
    log(f"Raw candidate pairs:     {raw_candidates:,}")
    log(f"Recovered true pairs:    {recovered:,}")
    log()
    log(f"BLOCKING RECALL:         {recall * 100:.4f}%")
    log()
    log(f"Blocks skipped:          {skipped_blocks:,}")
    log(f"Total runtime:           {elapsed:.2f} sec")
    log()
    log(f"Candidates: {CANDIDATE_FILE}")
    log(f"Report:     {REPORT_FILE}")

    log()
    log("=" * 72)

    if recall >= 0.95:
        log("SUCCESS: blocking recall is >= 95%.")
        log("Candidate generation is ready for the next stage.")
    elif recall >= 0.85:
        log("PROMISING: blocking recall is >= 85%.")
        log("Do not use for final submission yet.")
    else:
        log("BLOCKING RECALL IS STILL BELOW 85%.")
        log("Do not use for final test submission yet.")

    log("=" * 72)

    # Keep the final candidate file and report.
    # Remove the very large temporary files to recover disk space.
    log()
    log("Cleaning temporary files...")

    shutil.rmtree(TMP_DIR, ignore_errors=True)

    log("Temporary files removed.")
    log()
    log("STEP 16 COMPLETE.")


if __name__ == "__main__":
    main()