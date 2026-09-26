#!/usr/bin/env python3

"""
STEP 7 — EXTRACT TRAINING FEATURES

Reads:
    output/training_pairs_small.tsv

Expected pair columns:
    s1_entity_id
    matched_entity_id
    matched_source
    label

Streams:
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv

Creates:
    output/training_features.tsv

Designed for low-RAM machines.
"""

from __future__ import annotations

import csv
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset" / "train"
OUTPUT_DIR = BASE_DIR / "output"

PAIRS_FILE = OUTPUT_DIR / "training_pairs_small.tsv"
OUTPUT_FILE = OUTPUT_DIR / "training_features.tsv"

S1_FILE = DATASET_DIR / "train_source1.tsv"
S2_FILE = DATASET_DIR / "train_source2.tsv"
S3_FILE = DATASET_DIR / "train_source3.tsv"


# =============================================================================
# SETTINGS
# =============================================================================

PROGRESS_EVERY = 500_000


# =============================================================================
# RAPIDFUZZ
# =============================================================================

try:
    from rapidfuzz.fuzz import ratio, token_sort_ratio

    RAPIDFUZZ_AVAILABLE = True

except ImportError:
    RAPIDFUZZ_AVAILABLE = False


# =============================================================================
# NORMALIZATION
# =============================================================================

WHITESPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(value: str) -> str:
    """
    Lowercase text, remove punctuation, normalize whitespace.
    Unicode characters are preserved.
    """

    if value is None:
        return ""

    value = str(value).strip().lower()

    if not value:
        return ""

    value = NON_ALNUM_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value)

    return value.strip()


def compact_text(value: str) -> str:
    """
    Normalized text with spaces removed.
    """

    return normalize_text(value).replace(" ", "")


def token_sort_text(value: str) -> str:
    """
    Normalize text and alphabetically sort tokens.
    """

    normalized = normalize_text(value)

    if not normalized:
        return ""

    return " ".join(sorted(normalized.split()))


def digit_text(value: str) -> str:
    """
    Extract digits from an address.
    """

    if value is None:
        return ""

    return "".join(
        character
        for character in str(value)
        if character.isdigit()
    )


# =============================================================================
# SIMILARITY
# =============================================================================

def string_similarity(a: str, b: str) -> float:

    if not a or not b:
        return 0.0

    if a == b:
        return 100.0

    if RAPIDFUZZ_AVAILABLE:
        return float(ratio(a, b))

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio() * 100.0


def token_similarity(a: str, b: str) -> float:

    if not a or not b:
        return 0.0

    if a == b:
        return 100.0

    if RAPIDFUZZ_AVAILABLE:
        return float(token_sort_ratio(a, b))

    return SequenceMatcher(
        None,
        token_sort_text(a),
        token_sort_text(b),
    ).ratio() * 100.0


# =============================================================================
# RECORD
# =============================================================================

def prepare_record(
    business_name: str,
    business_address: str,
    country: str,
):
    """
    Store only the information needed for feature calculation.
    """

    name = business_name or ""
    address = business_address or ""
    country = country or ""

    return {
        "name": name,
        "address": address,
        "country": country.strip().lower(),

        "name_norm": normalize_text(name),
        "name_compact": compact_text(name),
        "name_tokens": token_sort_text(name),

        "address_norm": normalize_text(address),
        "address_compact": compact_text(address),
        "address_tokens": token_sort_text(address),
        "address_digits": digit_text(address),
    }


# =============================================================================
# COLUMN HELPERS
# =============================================================================

def normalized_column(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_column(fieldnames, candidates):

    normalized = {
        normalized_column(column): column
        for column in fieldnames
    }

    for candidate in candidates:

        candidate_normalized = normalized_column(candidate)

        if candidate_normalized in normalized:
            return normalized[candidate_normalized]

    return None


# =============================================================================
# LOAD TRAINING PAIRS
# =============================================================================

def load_training_pairs():

    print("=" * 80)
    print("LOADING TRAINING PAIRS")
    print("=" * 80)

    print()
    print(f"Input: {PAIRS_FILE}")

    if not PAIRS_FILE.exists():
        print()
        print("ERROR: Training-pair file does not exist.")
        print(PAIRS_FILE)
        sys.exit(1)

    pairs = []

    required_s1_ids = set()
    required_s2_ids = set()
    required_s3_ids = set()

    start = time.time()

    with PAIRS_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        if not reader.fieldnames:
            print("ERROR: File has no header.")
            sys.exit(1)

        print()
        print("Detected columns:")

        for column in reader.fieldnames:
            print(f"  {column}")

        # ---------------------------------------------------------------------
        # IMPORTANT:
        # These are the actual columns generated by Step 6.
        # ---------------------------------------------------------------------

        s1_column = find_column(
            reader.fieldnames,
            [
                "s1_entity_id",
                "s1_id",
                "source1_id",
                "source_1_id",
            ],
        )

        candidate_column = find_column(
            reader.fieldnames,
            [
                "matched_entity_id",
                "candidate_id",
                "matched_id",
                "match_id",
                "target_id",
                "entity_id_s2",
                "entity_id_s3",
            ],
        )

        source_column = find_column(
            reader.fieldnames,
            [
                "matched_source",
                "candidate_source",
                "source",
                "match_source",
                "target_source",
            ],
        )

        label_column = find_column(
            reader.fieldnames,
            [
                "label",
                "is_match",
                "match",
                "target",
                "y",
            ],
        )

        print()
        print("Column mapping:")
        print(f"  S1 ID:       {s1_column}")
        print(f"  Candidate:   {candidate_column}")
        print(f"  Source:      {source_column}")
        print(f"  Label:       {label_column}")

        if s1_column is None:
            print()
            print("ERROR: Could not find S1 entity ID column.")
            sys.exit(1)

        if candidate_column is None:
            print()
            print("ERROR: Could not find matched entity ID column.")
            sys.exit(1)

        if source_column is None:
            print()
            print("ERROR: Could not find matched source column.")
            sys.exit(1)

        if label_column is None:
            print()
            print("ERROR: Could not find label column.")
            sys.exit(1)

        # ---------------------------------------------------------------------
        # Read pairs
        # ---------------------------------------------------------------------

        for row in reader:

            s1_id = str(
                row.get(s1_column, "")
            ).strip()

            candidate_id = str(
                row.get(candidate_column, "")
            ).strip()

            source = str(
                row.get(source_column, "")
            ).strip().upper()

            raw_label = str(
                row.get(label_column, "")
            ).strip().lower()

            if not s1_id or not candidate_id:
                continue

            if source not in {"S2", "S3"}:

                print()
                print("ERROR: Invalid candidate source:")
                print(f"  S1:       {s1_id}")
                print(f"  Candidate:{candidate_id}")
                print(f"  Source:   {source}")

                sys.exit(1)

            if raw_label in {
                "1",
                "true",
                "yes",
                "y",
            }:
                label = 1

            else:
                label = 0

            pairs.append(
                (
                    s1_id,
                    candidate_id,
                    source,
                    label,
                )
            )

            required_s1_ids.add(s1_id)

            if source == "S2":
                required_s2_ids.add(candidate_id)

            else:
                required_s3_ids.add(candidate_id)

    elapsed = time.time() - start

    positive = sum(
        1
        for pair in pairs
        if pair[3] == 1
    )

    negative = len(pairs) - positive

    print()
    print("Training pairs loaded:")
    print(f"  Total:       {len(pairs):,}")
    print(f"  Positive:    {positive:,}")
    print(f"  Negative:    {negative:,}")

    print()
    print("Unique records required:")
    print(f"  S1:          {len(required_s1_ids):,}")
    print(f"  S2:          {len(required_s2_ids):,}")
    print(f"  S3:          {len(required_s3_ids):,}")

    print()
    print(f"Time:          {elapsed:.2f} sec")

    return (
        pairs,
        required_s1_ids,
        required_s2_ids,
        required_s3_ids,
    )


# =============================================================================
# STREAM SOURCE
# =============================================================================

def retrieve_records(
    filepath: Path,
    required_ids: set,
    source_name: str,
):

    print()
    print("=" * 80)
    print(f"READING {source_name}")
    print("=" * 80)

    print()
    print(f"File:       {filepath}")
    print(f"Required:   {len(required_ids):,}")

    records = {}

    if not required_ids:
        return records

    start = time.time()

    rows_seen = 0
    last_progress = 0

    with filepath.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        if not reader.fieldnames:
            print("ERROR: Source file has no header.")
            sys.exit(1)

        entity_column = find_column(
            reader.fieldnames,
            [
                "entity_id",
                "id",
            ],
        )

        name_column = find_column(
            reader.fieldnames,
            [
                "business_name",
                "name",
            ],
        )

        address_column = find_column(
            reader.fieldnames,
            [
                "business_address",
                "address",
            ],
        )

        country_column = find_column(
            reader.fieldnames,
            [
                "country",
            ],
        )

        if entity_column is None:
            print("ERROR: entity_id column not found.")
            sys.exit(1)

        if name_column is None:
            print("ERROR: business_name column not found.")
            sys.exit(1)

        if address_column is None:
            print("ERROR: business_address column not found.")
            sys.exit(1)

        if country_column is None:
            print("ERROR: country column not found.")
            sys.exit(1)

        for row in reader:

            rows_seen += 1

            entity_id = str(
                row.get(entity_column, "")
            ).strip()

            if entity_id in required_ids:

                records[entity_id] = prepare_record(
                    row.get(name_column, "") or "",
                    row.get(address_column, "") or "",
                    row.get(country_column, "") or "",
                )

            if rows_seen - last_progress >= PROGRESS_EVERY:

                elapsed = time.time() - start

                print(
                    f"  Scanned: {rows_seen:,} | "
                    f"Found: {len(records):,}/{len(required_ids):,} | "
                    f"Time: {elapsed:.1f}s"
                )

                last_progress = rows_seen

            # If every required record has been found,
            # stop immediately instead of reading the rest.
            if len(records) == len(required_ids):
                break

    elapsed = time.time() - start

    print()
    print(f"{source_name} complete:")
    print(f"  Rows scanned:  {rows_seen:,}")
    print(f"  Records found: {len(records):,}")
    print(
        f"  Records missing: "
        f"{len(required_ids) - len(records):,}"
    )
    print(f"  Time:           {elapsed:.2f} sec")

    return records


# =============================================================================
# FEATURES
# =============================================================================

FEATURE_COLUMNS = [
    "s1_entity_id",
    "matched_entity_id",
    "matched_source",
    "label",

    "same_country",

    "name_exact",
    "name_compact_exact",
    "name_similarity",
    "name_token_similarity",

    "address_exact",
    "address_compact_exact",
    "address_similarity",
    "address_token_similarity",

    "address_digits_exact",

    "name_length_diff",
    "address_length_diff",

    "name_missing",
    "address_missing",
]


def calculate_features(s1, candidate):

    s1_name_norm = s1["name_norm"]
    s1_name_compact = s1["name_compact"]
    s1_address_norm = s1["address_norm"]
    s1_address_compact = s1["address_compact"]

    candidate_name_norm = candidate["name_norm"]
    candidate_name_compact = candidate["name_compact"]
    candidate_address_norm = candidate["address_norm"]
    candidate_address_compact = candidate["address_compact"]

    # -------------------------------------------------------------------------
    # COUNTRY
    # -------------------------------------------------------------------------

    same_country = int(
        bool(s1["country"])
        and bool(candidate["country"])
        and s1["country"] == candidate["country"]
    )

    # -------------------------------------------------------------------------
    # NAME
    # -------------------------------------------------------------------------

    name_exact = int(
        bool(s1_name_norm)
        and bool(candidate_name_norm)
        and s1_name_norm == candidate_name_norm
    )

    name_compact_exact = int(
        bool(s1_name_compact)
        and bool(candidate_name_compact)
        and s1_name_compact == candidate_name_compact
    )

    name_similarity = string_similarity(
        s1_name_norm,
        candidate_name_norm,
    )

    name_token_similarity = token_similarity(
        s1_name_norm,
        candidate_name_norm,
    )

    # -------------------------------------------------------------------------
    # ADDRESS
    # -------------------------------------------------------------------------

    address_exact = int(
        bool(s1_address_norm)
        and bool(candidate_address_norm)
        and s1_address_norm == candidate_address_norm
    )

    address_compact_exact = int(
        bool(s1_address_compact)
        and bool(candidate_address_compact)
        and s1_address_compact == candidate_address_compact
    )

    address_similarity = string_similarity(
        s1_address_norm,
        candidate_address_norm,
    )

    address_token_similarity = token_similarity(
        s1_address_norm,
        candidate_address_norm,
    )

    # -------------------------------------------------------------------------
    # ADDRESS DIGITS
    # -------------------------------------------------------------------------

    address_digits_exact = int(
        bool(s1["address_digits"])
        and bool(candidate["address_digits"])
        and s1["address_digits"]
        == candidate["address_digits"]
    )

    # -------------------------------------------------------------------------
    # LENGTHS
    # -------------------------------------------------------------------------

    name_length_diff = abs(
        len(s1_name_norm)
        - len(candidate_name_norm)
    )

    address_length_diff = abs(
        len(s1_address_norm)
        - len(candidate_address_norm)
    )

    # -------------------------------------------------------------------------
    # MISSING
    # -------------------------------------------------------------------------

    name_missing = int(
        not s1_name_norm
        or not candidate_name_norm
    )

    address_missing = int(
        not s1_address_norm
        or not candidate_address_norm
    )

    return [
        same_country,

        name_exact,
        name_compact_exact,
        round(name_similarity, 4),
        round(name_token_similarity, 4),

        address_exact,
        address_compact_exact,
        round(address_similarity, 4),
        round(address_token_similarity, 4),

        address_digits_exact,

        name_length_diff,
        address_length_diff,

        name_missing,
        address_missing,
    ]


# =============================================================================
# WRITE FEATURES
# =============================================================================

def create_feature_file(
    pairs,
    s1_records,
    s2_records,
    s3_records,
):

    print()
    print("=" * 80)
    print("CALCULATING FEATURES")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = OUTPUT_FILE.with_suffix(
        ".tmp"
    )

    start = time.time()

    written = 0
    skipped_s1 = 0
    skipped_candidate = 0

    positives = 0
    negatives = 0

    with temporary_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(
            file,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow(FEATURE_COLUMNS)

        for index, (
            s1_id,
            candidate_id,
            source,
            label,
        ) in enumerate(
            pairs,
            start=1,
        ):

            s1 = s1_records.get(s1_id)

            if s1 is None:
                skipped_s1 += 1
                continue

            if source == "S2":
                candidate = s2_records.get(
                    candidate_id
                )

            else:
                candidate = s3_records.get(
                    candidate_id
                )

            if candidate is None:
                skipped_candidate += 1
                continue

            features = calculate_features(
                s1,
                candidate,
            )

            writer.writerow(
                [
                    s1_id,
                    candidate_id,
                    source,
                    label,
                    *features,
                ]
            )

            written += 1

            if label == 1:
                positives += 1
            else:
                negatives += 1

            if index % 10_000 == 0:

                elapsed = time.time() - start

                print(
                    f"  Processed: {index:,}/{len(pairs):,} | "
                    f"Written: {written:,} | "
                    f"Time: {elapsed:.1f}s"
                )

    temporary_file.replace(OUTPUT_FILE)

    elapsed = time.time() - start

    print()
    print("=" * 80)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 80)

    print()
    print(f"Output:")
    print(f"  {OUTPUT_FILE}")

    print()
    print("Rows:")
    print(f"  Written:       {written:,}")
    print(f"  Positive:      {positives:,}")
    print(f"  Negative:      {negatives:,}")

    print()
    print("Skipped:")
    print(f"  Missing S1:    {skipped_s1:,}")
    print(f"  Missing target:{skipped_candidate:,}")

    print()
    print(f"Feature runtime: {elapsed:.2f} sec")

    return written


# =============================================================================
# MAIN
# =============================================================================

def main():

    overall_start = time.time()

    print()
    print("=" * 80)
    print("STEP 7 — TRAINING FEATURE EXTRACTION")
    print("=" * 80)

    print()
    print("Machine-safe mode:")
    print("  • No SQLite candidate generation")
    print("  • Streaming TSV files")
    print("  • Only required records retained in RAM")
    print("  • 100,000 training pairs")

    print()
    print("Similarity engine:")

    if RAPIDFUZZ_AVAILABLE:
        print("  rapidfuzz: AVAILABLE")
    else:
        print("  rapidfuzz: NOT AVAILABLE")
        print("  Using difflib fallback.")

    # -------------------------------------------------------------------------
    # 1. PAIRS
    # -------------------------------------------------------------------------

    (
        pairs,
        required_s1_ids,
        required_s2_ids,
        required_s3_ids,
    ) = load_training_pairs()

    if not pairs:
        print()
        print("ERROR: No training pairs were loaded.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. S1
    # -------------------------------------------------------------------------

    s1_records = retrieve_records(
        S1_FILE,
        required_s1_ids,
        "S1",
    )

    # -------------------------------------------------------------------------
    # 3. S2
    # -------------------------------------------------------------------------

    s2_records = retrieve_records(
        S2_FILE,
        required_s2_ids,
        "S2",
    )

    # -------------------------------------------------------------------------
    # 4. S3
    # -------------------------------------------------------------------------

    s3_records = retrieve_records(
        S3_FILE,
        required_s3_ids,
        "S3",
    )

    # -------------------------------------------------------------------------
    # 5. FEATURES
    # -------------------------------------------------------------------------

    written = create_feature_file(
        pairs,
        s1_records,
        s2_records,
        s3_records,
    )

    total_runtime = time.time() - overall_start

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STEP 7 FINISHED")
    print("=" * 80)

    print()
    print(f"Feature file:")
    print(f"  {OUTPUT_FILE}")

    print()
    print(f"Rows created:")
    print(f"  {written:,}")

    print()
    print(f"Total runtime:")
    print(f"  {total_runtime:.2f} sec")

    print()
    print("NEXT:")
    print("  STEP 8 — train and validate the matching model")
    print()


if __name__ == "__main__":
    main()