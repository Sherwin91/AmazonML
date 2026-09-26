from pathlib import Path
from collections import Counter

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"

GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

CHUNK_SIZE = 50_000


def analyze_chunk(chunk, stats):
    for value in chunk["matched_entity_ids"].fillna(""):
        value = str(value).strip()

        # Empty match set
        if not value:
            stats["zero_match_s1"] += 1
            continue

        ids = [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

        total_matches = len(ids)

        stats["total_matches"] += total_matches

        # Number of matches for this S1
        stats["matches_per_s1"][total_matches] += 1

        # Source distribution
        s2_count = 0
        s3_count = 0

        for entity_id in ids:
            if entity_id.startswith("S2-"):
                s2_count += 1
                stats["total_s2_matches"] += 1

            elif entity_id.startswith("S3-"):
                s3_count += 1
                stats["total_s3_matches"] += 1

            else:
                stats["unexpected_ids"] += 1

        # Track whether this S1 has S2, S3, or both
        if s2_count > 0 and s3_count > 0:
            stats["s2_and_s3"] += 1
        elif s2_count > 0:
            stats["s2_only"] += 1
        elif s3_count > 0:
            stats["s3_only"] += 1

        stats["s2_matches_per_s1"][s2_count] += 1
        stats["s3_matches_per_s1"][s3_count] += 1


def print_distribution(title, counter):
    print(f"\n{title}")
    print("-" * 70)

    for key in sorted(counter):
        print(f"{key:>10}: {counter[key]:,}")


def main():
    print("=" * 80)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 80)

    print(f"File: {GROUND_TRUTH}")
    print(f"Chunk size: {CHUNK_SIZE:,}")

    stats = {
        "rows": 0,
        "zero_match_s1": 0,
        "total_matches": 0,
        "total_s2_matches": 0,
        "total_s3_matches": 0,
        "s2_only": 0,
        "s3_only": 0,
        "s2_and_s3": 0,
        "unexpected_ids": 0,
        "matches_per_s1": Counter(),
        "s2_matches_per_s1": Counter(),
        "s3_matches_per_s1": Counter(),
    }

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            GROUND_TRUTH,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ),
        start=1,
    ):
        stats["rows"] += len(chunk)

        analyze_chunk(chunk, stats)

        if chunk_number % 10 == 0:
            print(
                f"Processed {stats['rows']:,} rows...",
                flush=True,
            )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nTotal S1 rows: {stats['rows']:,}")

    print(
        f"Zero-match S1 entities: "
        f"{stats['zero_match_s1']:,} "
        f"({stats['zero_match_s1'] / stats['rows'] * 100:.2f}%)"
    )

    nonzero = stats["rows"] - stats["zero_match_s1"]

    print(
        f"Matched S1 entities: "
        f"{nonzero:,} "
        f"({nonzero / stats['rows'] * 100:.2f}%)"
    )

    print(f"\nTotal ground-truth matches: {stats['total_matches']:,}")

    print(f"S2 matches: {stats['total_s2_matches']:,}")
    print(f"S3 matches: {stats['total_s3_matches']:,}")

    print(
        f"\nS1 with S2 only: "
        f"{stats['s2_only']:,}"
    )

    print(
        f"S1 with S3 only: "
        f"{stats['s3_only']:,}"
    )

    print(
        f"S1 with both S2 and S3: "
        f"{stats['s2_and_s3']:,}"
    )

    print(
        f"\nUnexpected entity IDs: "
        f"{stats['unexpected_ids']:,}"
    )

    print_distribution(
        "TOTAL MATCHES PER S1",
        stats["matches_per_s1"],
    )

    print_distribution(
        "S2 MATCHES PER S1",
        stats["s2_matches_per_s1"],
    )

    print_distribution(
        "S3 MATCHES PER S1",
        stats["s3_matches_per_s1"],
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()