#!/usr/bin/env python3

"""
STEP 6 — FAST TRAINING SAMPLE

Creates:
    output/training_pairs_small.tsv

50,000 real positive pairs
50,000 real negative pairs

Ground truth format:

source1_entity_id    matched_entity_ids

S1-123    S2-111,S2-222,S3-333

Negative pairs are created ONLY from actual S2/S3 records.
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
OUTPUT_DIR = BASE_DIR / "output"

GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"
SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
SOURCE3 = TRAIN_DIR / "train_source3.tsv"

OUTPUT_FILE = OUTPUT_DIR / "training_pairs_small.tsv"


# =============================================================================
# SETTINGS
# =============================================================================

TARGET_POSITIVES = 50_000
TARGET_NEGATIVES = 50_000

RANDOM_SEED = 42

# Number of actual IDs retained per source/country.
POOL_SIZE = 100_000

rng = random.Random(RANDOM_SEED)


# =============================================================================
# HELPERS
# =============================================================================

def reservoir_add(
    reservoir,
    value,
    maximum,
    seen_count,
):
    """
    Keep a random sample of actual IDs without loading
    the complete source file.
    """

    if len(reservoir) < maximum:
        reservoir.append(value)
        return

    position = rng.randrange(seen_count)

    if position < maximum:
        reservoir[position] = value


# =============================================================================
# STEP 1
# SAMPLE REAL POSITIVE PAIRS FROM GROUND TRUTH
# =============================================================================

def sample_positive_pairs():

    print("=" * 80)
    print("STEP 6 — FAST TRAINING SAMPLE")
    print("=" * 80)

    print()
    print("Reading ground truth:")
    print(GROUND_TRUTH)

    start = time.time()

    positive_pairs = []

    total_s1_rows = 0
    total_true_pairs = 0

    with GROUND_TRUTH.open(
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
            raise RuntimeError(
                "Ground-truth file has no header."
            )

        print()
        print("Detected columns:")
        print(reader.fieldnames)

        if "source1_entity_id" not in reader.fieldnames:
            raise RuntimeError(
                "Expected column 'source1_entity_id' "
                "was not found."
            )

        if "matched_entity_ids" not in reader.fieldnames:
            raise RuntimeError(
                "Expected column 'matched_entity_ids' "
                "was not found."
            )

        for row in reader:

            total_s1_rows += 1

            s1_id = str(
                row["source1_entity_id"]
            ).strip()

            matched_ids_raw = str(
                row["matched_entity_ids"] or ""
            ).strip()

            if not s1_id:
                continue

            if not matched_ids_raw:
                continue

            matched_ids = [
                x.strip()
                for x in matched_ids_raw.split(",")
                if x.strip()
            ]

            for matched_id in matched_ids:

                if not (
                    matched_id.startswith("S2-")
                    or matched_id.startswith("S3-")
                ):
                    continue

                if matched_id.startswith("S2-"):
                    source = "S2"
                else:
                    source = "S3"

                total_true_pairs += 1

                pair = (
                    s1_id,
                    matched_id,
                    source,
                    1,
                )

                # Reservoir sample from ALL real positive pairs.
                if len(positive_pairs) < TARGET_POSITIVES:

                    positive_pairs.append(pair)

                else:

                    position = rng.randrange(
                        total_true_pairs
                    )

                    if position < TARGET_POSITIVES:
                        positive_pairs[position] = pair

    elapsed = time.time() - start

    print()
    print(f"S1 ground-truth rows: {total_s1_rows:,}")
    print(f"True positive pairs:  {total_true_pairs:,}")
    print(f"Positive pairs kept:  {len(positive_pairs):,}")
    print(f"Time:                 {elapsed:.2f} sec")

    if len(positive_pairs) < TARGET_POSITIVES:
        raise RuntimeError(
            f"Only {len(positive_pairs):,} positive pairs found."
        )

    return positive_pairs


# =============================================================================
# STEP 2
# GET COUNTRY FOR SAMPLED S1 RECORDS
# =============================================================================

def load_sampled_s1_countries(positive_pairs):

    print()
    print("=" * 80)
    print("LOADING S1 COUNTRIES")
    print("=" * 80)

    required_s1 = {
        pair[0]
        for pair in positive_pairs
    }

    countries = {}

    start = time.time()

    with SOURCE1.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        if "entity_id" not in reader.fieldnames:
            raise RuntimeError(
                "Source 1 does not contain entity_id."
            )

        if "country" not in reader.fieldnames:
            raise RuntimeError(
                "Source 1 does not contain country."
            )

        for row in reader:

            entity_id = str(
                row["entity_id"]
            ).strip()

            if entity_id not in required_s1:
                continue

            country = str(
                row["country"] or ""
            ).strip().lower()

            countries[entity_id] = country

            if len(countries) == len(required_s1):
                break

    elapsed = time.time() - start

    print()
    print(f"S1 IDs required: {len(required_s1):,}")
    print(f"S1 countries found: {len(countries):,}")
    print(
        f"S1 countries missing: "
        f"{len(required_s1) - len(countries):,}"
    )
    print(f"Time: {elapsed:.2f} sec")

    if len(countries) != len(required_s1):
        raise RuntimeError(
            "Some sampled S1 IDs were not found in Source 1."
        )

    return countries


# =============================================================================
# STEP 3
# BUILD REAL S2/S3 ID POOLS
# =============================================================================

def build_source_pools(filepath, source_name):

    print()
    print("=" * 80)
    print(f"BUILDING REAL {source_name} ID POOLS")
    print("=" * 80)

    print()
    print(filepath)

    pools = {}

    seen_counts = {}

    rows_scanned = 0

    start = time.time()

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

        if "entity_id" not in reader.fieldnames:
            raise RuntimeError(
                f"{source_name} missing entity_id."
            )

        if "country" not in reader.fieldnames:
            raise RuntimeError(
                f"{source_name} missing country."
            )

        for row in reader:

            rows_scanned += 1

            entity_id = str(
                row["entity_id"] or ""
            ).strip()

            if not entity_id:
                continue

            country = str(
                row["country"] or ""
            ).strip().lower()

            if not country:
                country = "__unknown__"

            if country not in pools:
                pools[country] = []

            if country not in seen_counts:
                seen_counts[country] = 0

            seen_counts[country] += 1

            reservoir_add(
                pools[country],
                entity_id,
                POOL_SIZE,
                seen_counts[country],
            )

            if rows_scanned % 1_000_000 == 0:

                elapsed = time.time() - start

                print(
                    f"  scanned={rows_scanned:,} "
                    f"time={elapsed:.1f}s"
                )

    elapsed = time.time() - start

    print()
    print(f"{source_name} rows scanned: {rows_scanned:,}")

    for country in sorted(pools):

        print(
            f"  {country}: "
            f"{len(pools[country]):,} real IDs"
        )

    print(f"Time: {elapsed:.2f} sec")

    return pools


# =============================================================================
# STEP 4
# BUILD COMPLETE TRUE-MATCH LOOKUP FOR SAMPLED S1s
# =============================================================================

def build_true_match_lookup(
    positive_pairs,
):

    """
    Positive sampling alone does NOT tell us all true matches
    for a sampled S1.

    Example:

        S1-A -> S2-1,S2-2,S3-5,S3-9

    If S2-2 wasn't selected among our 50k positive samples,
    it is still NOT a valid negative.

    Therefore we scan ground truth again and build the complete
    true-match set for the sampled S1 entities.
    """

    print()
    print("=" * 80)
    print("BUILDING TRUE-MATCH LOOKUP")
    print("=" * 80)

    sampled_s1 = {
        pair[0]
        for pair in positive_pairs
    }

    true_matches = {}

    start = time.time()

    with GROUND_TRUTH.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        for row in reader:

            s1_id = str(
                row["source1_entity_id"]
            ).strip()

            if s1_id not in sampled_s1:
                continue

            raw_ids = str(
                row["matched_entity_ids"] or ""
            ).strip()

            if not raw_ids:
                continue

            key = s1_id

            if key not in true_matches:
                true_matches[key] = {
                    "S2": set(),
                    "S3": set(),
                }

            for matched_id in raw_ids.split(","):

                matched_id = matched_id.strip()

                if matched_id.startswith("S2-"):
                    true_matches[key]["S2"].add(
                        matched_id
                    )

                elif matched_id.startswith("S3-"):
                    true_matches[key]["S3"].add(
                        matched_id
                    )

    elapsed = time.time() - start

    total_matches = sum(
        len(v["S2"]) + len(v["S3"])
        for v in true_matches.values()
    )

    print()
    print(f"Sampled S1 entities: {len(sampled_s1):,}")
    print(f"S1s with true matches: {len(true_matches):,}")
    print(f"True matches loaded: {total_matches:,}")
    print(f"Time: {elapsed:.2f} sec")

    return true_matches


# =============================================================================
# STEP 5
# CREATE VALID NEGATIVES
# =============================================================================

def create_negative_pairs(
    positive_pairs,
    s1_countries,
    true_matches,
    s2_pools,
    s3_pools,
):

    print()
    print("=" * 80)
    print("CREATING REAL NEGATIVE PAIRS")
    print("=" * 80)

    negatives = []

    used_pairs = set()

    attempts = 0

    start = time.time()

    positive_index = 0

    while len(negatives) < TARGET_NEGATIVES:

        positive = positive_pairs[
            positive_index
        ]

        positive_index += 1

        if positive_index >= len(positive_pairs):
            positive_index = 0

        s1_id = positive[0]

        # Alternate S2/S3 to get both source types.
        if len(negatives) % 2 == 0:
            source = "S2"
            pools = s2_pools
        else:
            source = "S3"
            pools = s3_pools

        country = s1_countries[s1_id]

        # Prefer same-country candidate.
        pool = pools.get(country)

        if not pool:

            # Fallback to another available country.
            available_pools = [
                value
                for value in pools.values()
                if value
            ]

            if not available_pools:
                raise RuntimeError(
                    f"No real {source} IDs available."
                )

            pool = rng.choice(
                available_pools
            )

        candidate_id = rng.choice(pool)

        attempts += 1

        # ---------------------------------------------------------------------
        # Make absolutely sure this candidate is not a true match.
        # ---------------------------------------------------------------------

        if candidate_id in true_matches.get(
            s1_id,
            {"S2": set(), "S3": set()},
        )[source]:
            continue

        pair_key = (
            s1_id,
            candidate_id,
            source,
        )

        # Avoid exact duplicate pairs.
        if pair_key in used_pairs:
            continue

        used_pairs.add(pair_key)

        negatives.append(
            (
                s1_id,
                candidate_id,
                source,
                0,
            )
        )

        if len(negatives) % 10_000 == 0:

            elapsed = time.time() - start

            print(
                f"  Negatives: "
                f"{len(negatives):,}/"
                f"{TARGET_NEGATIVES:,} | "
                f"Attempts: {attempts:,} | "
                f"Time: {elapsed:.2f}s"
            )

    elapsed = time.time() - start

    print()
    print(f"Negative pairs: {len(negatives):,}")
    print(f"Attempts:       {attempts:,}")
    print(f"Time:           {elapsed:.2f} sec")

    return negatives


# =============================================================================
# STEP 6
# WRITE
# =============================================================================

def write_training_pairs(
    positives,
    negatives,
):

    print()
    print("=" * 80)
    print("WRITING TRAINING PAIRS")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs = positives + negatives

    rng.shuffle(pairs)

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer = csv.writer(
            file,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow(
            [
                "s1_entity_id",
                "matched_entity_id",
                "matched_source",
                "label",
            ]
        )

        writer.writerows(pairs)

    print()
    print(f"Output: {OUTPUT_FILE}")
    print(f"Positive: {len(positives):,}")
    print(f"Negative: {len(negatives):,}")
    print(f"Total: {len(pairs):,}")


# =============================================================================
# STEP 7
# VALIDATE
# =============================================================================

def validate_training_pairs():

    print()
    print("=" * 80)
    print("VALIDATING OUTPUT")
    print("=" * 80)

    rows = 0
    positives = 0
    negatives = 0

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file,
            delimiter="\t",
        )

        expected_columns = {
            "s1_entity_id",
            "matched_entity_id",
            "matched_source",
            "label",
        }

        if set(reader.fieldnames or []) != expected_columns:
            raise RuntimeError(
                f"Unexpected output columns: "
                f"{reader.fieldnames}"
            )

        for row in reader:

            rows += 1

            source = row["matched_source"]
            label = row["label"]

            if source not in {"S2", "S3"}:
                raise RuntimeError(
                    f"Invalid source: {source}"
                )

            if label == "1":
                positives += 1

            elif label == "0":
                negatives += 1

            else:
                raise RuntimeError(
                    f"Invalid label: {label}"
                )

    print()
    print(f"Rows:       {rows:,}")
    print(f"Positive:   {positives:,}")
    print(f"Negative:   {negatives:,}")

    if positives != TARGET_POSITIVES:
        raise RuntimeError(
            f"Expected {TARGET_POSITIVES:,} positives."
        )

    if negatives != TARGET_NEGATIVES:
        raise RuntimeError(
            f"Expected {TARGET_NEGATIVES:,} negatives."
        )

    if rows != TARGET_POSITIVES + TARGET_NEGATIVES:
        raise RuntimeError(
            "Unexpected total row count."
        )

    print()
    print("VALIDATION PASSED.")


# =============================================================================
# MAIN
# =============================================================================

def main():

    overall_start = time.time()

    # 1. Sample real positives.
    positives = sample_positive_pairs()

    # 2. Find countries of sampled S1 entities.
    s1_countries = load_sampled_s1_countries(
        positives
    )

    # 3. Build real candidate pools.
    s2_pools = build_source_pools(
        SOURCE2,
        "S2",
    )

    s3_pools = build_source_pools(
        SOURCE3,
        "S3",
    )

    # 4. Load ALL known true matches for sampled S1s.
    true_matches = build_true_match_lookup(
        positives
    )

    # 5. Generate genuine negatives.
    negatives = create_negative_pairs(
        positives,
        s1_countries,
        true_matches,
        s2_pools,
        s3_pools,
    )

    # 6. Write.
    write_training_pairs(
        positives,
        negatives,
    )

    # 7. Validate.
    validate_training_pairs()

    total_time = time.time() - overall_start

    print()
    print("=" * 80)
    print("STEP 6 COMPLETE")
    print("=" * 80)

    print()
    print(f"Total runtime: {total_time:.2f} sec")

    print()
    print("NEXT COMMAND:")
    print()
    print("python src/07_extract_features.py")
    print()


if __name__ == "__main__":
    main()