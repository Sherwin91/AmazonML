from pathlib import Path
import random
import re
import unicodedata
from difflib import SequenceMatcher

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"

GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"
SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
SOURCE3 = TRAIN_DIR / "train_source3.tsv"


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 50_000
SAMPLE_S1_COUNT = 30
RANDOM_SEED = 42


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Conservative Unicode-aware normalization.

    Important:
    - preserves non-Latin scripts
    - converts case consistently
    - removes punctuation
    - collapses whitespace
    """
    if value is None:
        return ""

    value = str(value)

    if not value or value.lower() == "nan":
        return ""

    value = unicodedata.normalize("NFKC", value)
    value = value.casefold()

    # Keep Unicode letters/numbers and whitespace.
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    value = re.sub(r"\s+", " ", value).strip()

    return value


def similarity(a, b):
    """
    Similarity percentage using SequenceMatcher.
    """
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return None

    return round(
        SequenceMatcher(None, a, b).ratio() * 100,
        1,
    )


def exact_normalized(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    return bool(a and b and a == b)


# ============================================================
# STEP 1
# GET SAMPLE S1 IDS AND THEIR GROUND-TRUTH MATCHES
# ============================================================

def get_sample_matches():

    print("\n" + "=" * 80)
    print("STEP 1 — SAMPLING GROUND-TRUTH MATCHES")
    print("=" * 80)

    rows = []

    for chunk in pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):
        rows.extend(
            chunk[
                ["source1_entity_id", "matched_entity_ids"]
            ].to_dict("records")
        )

    print(f"Loaded ground-truth metadata rows: {len(rows):,}")

    random.seed(RANDOM_SEED)

    sample = random.sample(
        rows,
        min(SAMPLE_S1_COUNT, len(rows)),
    )

    sample_s1_ids = set()
    matched_ids = set()

    mapping = {}

    for row in sample:

        s1_id = row["source1_entity_id"]

        sample_s1_ids.add(s1_id)

        raw_matches = row["matched_entity_ids"].strip()

        if raw_matches:
            ids = [
                x.strip()
                for x in raw_matches.split(",")
                if x.strip()
            ]
        else:
            ids = []

        mapping[s1_id] = ids

        for entity_id in ids:
            matched_ids.add(entity_id)

    print(f"Sampled S1 entities: {len(sample_s1_ids)}")
    print(f"Unique matched S2/S3 records: {len(matched_ids)}")

    print("\nSampled S1 entities:")
    for s1_id, ids in mapping.items():
        print(
            f"  {s1_id}: "
            f"{len(ids)} matches"
        )

    return sample_s1_ids, matched_ids, mapping


# ============================================================
# STEP 2
# STREAM SOURCE FILE AND RETRIEVE REQUIRED RECORDS
# ============================================================

def retrieve_records(filepath, wanted_ids):

    found = {}

    print("\n" + "-" * 80)
    print(f"Scanning: {filepath.name}")
    print("-" * 80)

    processed = 0

    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=False,
    ):

        processed += len(chunk)

        mask = chunk["entity_id"].isin(wanted_ids)

        selected = chunk.loc[mask]

        for row in selected.itertuples(index=False):

            record = {
                "entity_id": getattr(row, "entity_id"),
                "business_name": getattr(
                    row,
                    "business_name",
                    "",
                ),
                "business_address": getattr(
                    row,
                    "business_address",
                    "",
                ),
                "country": getattr(
                    row,
                    "country",
                    "",
                ),
            }

            found[record["entity_id"]] = record

        del chunk

        if processed % 1_000_000 == 0:
            print(
                f"  Processed {processed:,} rows..."
            )

        # Stop early if every requested ID has been found.
        if len(found) == len(wanted_ids):
            print(
                f"  Found all {len(wanted_ids)} requested records."
            )
            break

    print(
        f"  Retrieved: {len(found):,} / "
        f"{len(wanted_ids):,}"
    )

    return found


# ============================================================
# STEP 3
# PRINT COMPARISONS
# ============================================================

def print_comparison(
    s1,
    other,
    source_name,
):

    name_similarity = similarity(
        s1["business_name"],
        other["business_name"],
    )

    address_similarity = similarity(
        s1["business_address"],
        other["business_address"],
    )

    name_exact = exact_normalized(
        s1["business_name"],
        other["business_name"],
    )

    address_exact = exact_normalized(
        s1["business_address"],
        other["business_address"],
    )

    print("\n" + "·" * 75)

    print(
        f"{source_name}: "
        f"{other['entity_id']}"
    )

    print(
        f"Country: "
        f"{s1['country']} → {other['country']}"
    )

    print(
        f"\nS1 NAME:\n"
        f"  {s1['business_name']}"
    )

    print(
        f"{source_name} NAME:\n"
        f"  {other['business_name']}"
    )

    print(
        f"Name similarity: "
        f"{name_similarity}%"
    )

    print(
        f"Normalized name exact: "
        f"{name_exact}"
    )

    print(
        f"\nS1 ADDRESS:\n"
        f"  {s1['business_address']}"
    )

    print(
        f"{source_name} ADDRESS:\n"
        f"  {other['business_address']}"
    )

    if address_similarity is None:
        print("Address similarity: N/A")
    else:
        print(
            f"Address similarity: "
            f"{address_similarity}%"
        )

    print(
        f"Normalized address exact: "
        f"{address_exact}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("REAL MATCH SAMPLER")
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Ground truth
    # --------------------------------------------------------

    sample_s1_ids, matched_ids, mapping = (
        get_sample_matches()
    )

    # --------------------------------------------------------
    # 2. Retrieve S1
    # --------------------------------------------------------

    s1_records = retrieve_records(
        SOURCE1,
        sample_s1_ids,
    )

    # --------------------------------------------------------
    # 3. Split requested IDs by source
    # --------------------------------------------------------

    s2_ids = {
        entity_id
        for entity_id in matched_ids
        if entity_id.startswith("S2-")
    }

    s3_ids = {
        entity_id
        for entity_id in matched_ids
        if entity_id.startswith("S3-")
    }

    print("\n" + "=" * 80)
    print("MATCHED RECORD COUNTS")
    print("=" * 80)

    print(f"S2 records needed: {len(s2_ids)}")
    print(f"S3 records needed: {len(s3_ids)}")

    # --------------------------------------------------------
    # 4. Retrieve S2
    # --------------------------------------------------------

    s2_records = retrieve_records(
        SOURCE2,
        s2_ids,
    )

    # --------------------------------------------------------
    # 5. Retrieve S3
    # --------------------------------------------------------

    s3_records = retrieve_records(
        SOURCE3,
        s3_ids,
    )

    # --------------------------------------------------------
    # 6. Compare
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("REAL POSITIVE MATCH EXAMPLES")
    print("=" * 80)

    total_comparisons = 0

    for s1_id, matched in mapping.items():

        s1 = s1_records.get(s1_id)

        if s1 is None:
            print(
                f"\nWARNING: S1 record not found: "
                f"{s1_id}"
            )
            continue

        print("\n")
        print("#" * 80)
        print(
            f"S1 ENTITY: {s1_id}"
        )
        print("#" * 80)

        print(
            f"\nS1 NAME:\n"
            f"  {s1['business_name']}"
        )

        print(
            f"S1 ADDRESS:\n"
            f"  {s1['business_address']}"
        )

        print(
            f"S1 COUNTRY:\n"
            f"  {s1['country']}"
        )

        print(
            f"\nTRUE MATCH COUNT: "
            f"{len(matched)}"
        )

        for entity_id in matched:

            if entity_id.startswith("S2-"):
                other = s2_records.get(entity_id)
                source_name = "S2"

            elif entity_id.startswith("S3-"):
                other = s3_records.get(entity_id)
                source_name = "S3"

            else:
                continue

            if other is None:
                print(
                    f"\nWARNING: "
                    f"{entity_id} not found."
                )
                continue

            print_comparison(
                s1,
                other,
                source_name,
            )

            total_comparisons += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("SAMPLING COMPLETE")
    print("=" * 80)

    print(
        f"S1 sampled: {len(sample_s1_ids)}"
    )

    print(
        f"S2 records retrieved: "
        f"{len(s2_records)}"
    )

    print(
        f"S3 records retrieved: "
        f"{len(s3_records)}"
    )

    print(
        f"Positive comparisons displayed: "
        f"{total_comparisons}"
    )


if __name__ == "__main__":
    main()