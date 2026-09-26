#!/usr/bin/env python3

"""
BUSINESS ENTITY RESOLUTION
STEP 5 — FAST MEMORY-SAFE CANDIDATE GENERATION

Uses the SQLite database created by 04_build_indexes.py.

Training candidates are generated from:
    S1 -> S2
    S1 -> S3

Blocking rules:
    1. Exact normalized name
    2. Exact compact name
    3. Exact token signature
    4. Exact normalized address
    5. Exact compact address
    6. Exact address digits
    7. Controlled name prefix
    8. Controlled address prefix

No external data is used.

The output is a UNIQUE candidate-pair file.  A pair is retained
if at least one blocking rule discovers it.
"""

from pathlib import Path
import sqlite3
import csv
import time
import sys


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

DB_PATH = OUTPUT_DIR / "entity_indexes.sqlite"
OUTPUT_PATH = OUTPUT_DIR / "fast_training_candidates.tsv"


# ============================================================================
# CONFIGURATION
# ============================================================================

S1_BATCH = 5_000

# Prefix length.
PREFIX_LENGTH = 4

# Do not expand a prefix block if it is too common.
MAX_PREFIX_BLOCK = 200

# Maximum candidates retained for one S1.
MAX_CANDIDATES_PER_S1 = 500

# Maximum candidates returned by any one block.
MAX_PER_BLOCK = 150


# ============================================================================
# DATABASE
# ============================================================================

def open_database():
    if not DB_PATH.exists():
        print("ERROR: SQLite database not found:")
        print(DB_PATH)
        print()
        print("Run:")
        print("python src/04_build_indexes.py")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-20000")

    return conn


# ============================================================================
# PREFIX RANGE
# ============================================================================

def next_prefix(prefix):
    """
    Return the smallest string strictly greater than every string
    beginning with prefix.

    Example:
        abcd -> abce
    """

    if not prefix:
        return None

    chars = list(prefix)

    for i in range(len(chars) - 1, -1, -1):
        value = ord(chars[i])

        if value < 0x10FFFF:
            chars[i] = chr(value + 1)
            return "".join(chars[: i + 1])

    return None


# ============================================================================
# BLOCK QUERY HELPERS
# ============================================================================

def add_rows(candidate_map, rows, block_name):
    """
    Add rows to candidate_map.

    candidate_map:
        (entity_id, source) -> first block that discovered it
    """

    added = 0

    for entity_id, source in rows:
        key = (entity_id, source)

        if key not in candidate_map:
            candidate_map[key] = block_name
            added += 1

            if added >= MAX_PER_BLOCK:
                break

    return added


def exact_block(
    conn,
    field,
    value,
    country,
    candidate_map,
    block_name,
):
    if not value:
        return 0

    sql = f"""
        SELECT entity_id, source
        FROM records
        WHERE country = ?
          AND {field} = ?
          AND source IN ('S2', 'S3')
        LIMIT ?
    """

    rows = conn.execute(
        sql,
        (country, value, MAX_PER_BLOCK),
    )

    return add_rows(
        candidate_map,
        rows,
        block_name,
    )


def prefix_block(
    conn,
    field,
    value,
    country,
    candidate_map,
    block_name,
):
    if not value:
        return 0

    compact = value.replace(" ", "")

    if len(compact) < PREFIX_LENGTH:
        return 0

    prefix = compact[:PREFIX_LENGTH]
    upper = next_prefix(prefix)

    if upper is None:
        return 0

    # First determine whether the block is small enough.
    count_sql = f"""
        SELECT COUNT(*)
        FROM records
        WHERE country = ?
          AND {field} >= ?
          AND {field} < ?
          AND source IN ('S2', 'S3')
    """

    count = conn.execute(
        count_sql,
        (country, prefix, upper),
    ).fetchone()[0]

    if count == 0 or count > MAX_PREFIX_BLOCK:
        return 0

    rows = conn.execute(
        f"""
            SELECT entity_id, source
            FROM records
            WHERE country = ?
              AND {field} >= ?
              AND {field} < ?
              AND source IN ('S2', 'S3')
            LIMIT ?
        """,
        (
            country,
            prefix,
            upper,
            MAX_PER_BLOCK,
        ),
    )

    return add_rows(
        candidate_map,
        rows,
        block_name,
    )


# ============================================================================
# PROCESS ONE S1
# ============================================================================

def generate_for_s1(
    conn,
    s1_row,
):
    (
        s1_id,
        country,
        name_norm,
        name_compact,
        name_tokens,
        address_norm,
        address_compact,
        address_digits,
    ) = s1_row

    candidates = {}

    # ------------------------------------------------------------------------
    # 1. Exact normalized name
    # ------------------------------------------------------------------------

    exact_block(
        conn,
        "name_norm",
        name_norm,
        country,
        candidates,
        "exact_name",
    )

    # ------------------------------------------------------------------------
    # 2. Exact compact name
    # ------------------------------------------------------------------------

    if len(candidates) < MAX_CANDIDATES_PER_S1:
        exact_block(
            conn,
            "name_compact",
            name_compact,
            country,
            candidates,
            "exact_compact_name",
        )

    # ------------------------------------------------------------------------
    # 3. Exact token signature
    #
    # This handles word-order differences such as:
    #
    #     ABC Dental Services
    #     Services ABC Dental
    # ------------------------------------------------------------------------

    if len(candidates) < MAX_CANDIDATES_PER_S1:
        exact_block(
            conn,
            "name_tokens",
            name_tokens,
            country,
            candidates,
            "name_tokens",
        )

    # ------------------------------------------------------------------------
    # 4. Exact normalized address
    # ------------------------------------------------------------------------

    if len(candidates) < MAX_CANDIDATES_PER_S1:
        exact_block(
            conn,
            "address_norm",
            address_norm,
            country,
            candidates,
            "exact_address",
        )

    # ------------------------------------------------------------------------
    # 5. Exact compact address
    # ------------------------------------------------------------------------

    if len(candidates) < MAX_CANDIDATES_PER_S1:
        exact_block(
            conn,
            "address_compact",
            address_compact,
            country,
            candidates,
            "exact_compact_address",
        )

    # ------------------------------------------------------------------------
    # 6. Address digits
    #
    # Useful when names are substantially different but the street/house/
    # postal-number structure is shared.
    # ------------------------------------------------------------------------

    if len(candidates) < MAX_CANDIDATES_PER_S1:
        exact_block(
            conn,
            "address_digits",
            address_digits,
            country,
            candidates,
            "address_digits",
        )

    # ------------------------------------------------------------------------
    # 7. Controlled name prefix
    #
    # Helps with:
    #     Advanced Circle Group
    #     Advanced Circle LLC
    #     United Bny Clinic
    #     United Bank Clinic
    #
    # but skips extremely common prefixes.
    # ------------------------------------------------------------------------

    if len(candidates) < MAX_CANDIDATES_PER_S1:
        prefix_block(
            conn,
            "name_norm",
            name_norm,
            country,
            candidates,
            "name_prefix",
        )

    # ------------------------------------------------------------------------
    # 8. Controlled address prefix
    # ------------------------------------------------------------------------

    if len(candidates) < MAX_CANDIDATES_PER_S1:
        prefix_block(
            conn,
            "address_norm",
            address_norm,
            country,
            candidates,
            "address_prefix",
        )

    # ------------------------------------------------------------------------
    # Hard safety limit
    # ------------------------------------------------------------------------

    if len(candidates) > MAX_CANDIDATES_PER_S1:
        candidates = dict(
            list(candidates.items())[:MAX_CANDIDATES_PER_S1]
        )

    return candidates


# ============================================================================
# GENERATE
# ============================================================================

def generate_candidates(conn):

    print()
    print("=" * 80)
    print("STEP 5 — FAST TRAINING CANDIDATE GENERATION")
    print("=" * 80)

    print()
    print("Database:")
    print(DB_PATH)

    print()
    print("Output:")
    print(OUTPUT_PATH)

    print()
    print("Configuration:")
    print(f"  S1 batch size:             {S1_BATCH:,}")
    print(f"  Prefix length:             {PREFIX_LENGTH}")
    print(f"  Maximum prefix block:      {MAX_PREFIX_BLOCK:,}")
    print(f"  Maximum candidates / S1:   {MAX_CANDIDATES_PER_S1:,}")

    if OUTPUT_PATH.exists():
        print()
        print("Removing previous output...")
        OUTPUT_PATH.unlink()

    # ------------------------------------------------------------------------
    # Stream S1 records.
    # ------------------------------------------------------------------------

    cursor = conn.execute(
        """
        SELECT
            entity_id,
            country,
            name_norm,
            name_compact,
            name_tokens,
            address_norm,
            address_compact,
            address_digits
        FROM records
        WHERE source = 'S1'
        ORDER BY rowid
        """
    )

    total_s1 = 0
    total_candidates = 0
    no_candidate_s1 = 0

    block_stats = {
        "exact_name": 0,
        "exact_compact_name": 0,
        "name_tokens": 0,
        "exact_address": 0,
        "exact_compact_address": 0,
        "address_digits": 0,
        "name_prefix": 0,
        "address_prefix": 0,
    }

    start = time.time()

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:

        writer = csv.writer(
            output_file,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow(
            [
                "s1_entity_id",
                "matched_entity_id",
                "matched_source",
                "block_type",
            ]
        )

        while True:

            rows = cursor.fetchmany(S1_BATCH)

            if not rows:
                break

            for row in rows:

                s1_id = row[0]

                candidates = generate_for_s1(
                    conn,
                    row,
                )

                total_s1 += 1

                if not candidates:
                    no_candidate_s1 += 1

                for (
                    matched_id,
                    matched_source,
                ), block_type in candidates.items():

                    writer.writerow(
                        [
                            s1_id,
                            matched_id,
                            matched_source,
                            block_type,
                        ]
                    )

                    total_candidates += 1
                    block_stats[block_type] += 1

                # ------------------------------------------------------------
                # Progress every 5,000 S1.
                # ------------------------------------------------------------

                if total_s1 % 5_000 == 0:

                    elapsed = time.time() - start
                    rate = total_s1 / max(elapsed, 0.001)

                    avg = total_candidates / total_s1

                    print(
                        f"  S1: {total_s1:>10,} "
                        f"| candidates: {total_candidates:>12,} "
                        f"| avg/S1: {avg:>7.2f} "
                        f"| rate: {rate:>7.1f}/sec",
                        flush=True,
                    )

    elapsed = time.time() - start

    return (
        total_s1,
        total_candidates,
        no_candidate_s1,
        block_stats,
        elapsed,
    )


# ============================================================================
# OUTPUT VALIDATION
# ============================================================================

def validate_output():

    print()
    print("=" * 80)
    print("VALIDATING CANDIDATE FILE")
    print("=" * 80)

    if not OUTPUT_PATH.exists():
        print("ERROR: output file was not created.")
        sys.exit(1)

    row_count = 0
    s1_ids = set()
    source_errors = 0

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        expected = {
            "s1_entity_id",
            "matched_entity_id",
            "matched_source",
            "block_type",
        }

        if set(reader.fieldnames or []) != expected:
            print("ERROR: incorrect columns.")
            print("Found:", reader.fieldnames)
            sys.exit(1)

        for row in reader:

            row_count += 1
            s1_ids.add(row["s1_entity_id"])

            if row["matched_source"] not in {"S2", "S3"}:
                source_errors += 1

    print(f"Candidate rows:             {row_count:,}")
    print(f"S1 entities represented:    {len(s1_ids):,}")
    print(f"Invalid source rows:        {source_errors:,}")

    if source_errors:
        print("ERROR: invalid candidate source detected.")
        sys.exit(1)

    print("Candidate file validation passed.")


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 80)
    print("BUSINESS ENTITY RESOLUTION")
    print("FAST PIPELINE")
    print("=" * 80)

    conn = open_database()

    try:

        (
            total_s1,
            total_candidates,
            no_candidate_s1,
            block_stats,
            elapsed,
        ) = generate_candidates(conn)

        validate_output()

        print()
        print("=" * 80)
        print("STEP 5 COMPLETE")
        print("=" * 80)

        print()
        print(f"S1 processed:             {total_s1:,}")
        print(f"Candidate pairs:          {total_candidates:,}")
        print(f"S1 with no candidates:    {no_candidate_s1:,}")

        if total_s1:
            print(
                f"Average candidates/S1:   "
                f"{total_candidates / total_s1:.2f}"
            )

        print()
        print("Candidates discovered by blocking rule:")
        print("-" * 60)

        for name, count in block_stats.items():
            print(f"{name:28s}: {count:,}")

        print()
        print(f"Runtime: {elapsed / 60:.2f} minutes")

        print()
        print("Output:")
        print(OUTPUT_PATH)

        print()
        print("Next:")
        print("  STEP 6 — Train matching features/model")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
