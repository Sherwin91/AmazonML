#!/usr/bin/env python3

"""
BUSINESS ENTITY RESOLUTION
STEP 5 — FAST SET-BASED CANDIDATE GENERATION

Uses the existing entity_indexes.sqlite.

IMPORTANT:
    This version does NOT perform one SQLite query per S1 row.

Instead, SQLite performs set-based joins between S1 and S2/S3.
This is dramatically faster than the previous Python-loop approach.

Blocking rules:
    1. exact normalized name
    2. exact compact name
    3. exact token signature
    4. exact normalized address
    5. exact compact address
    6. exact address digits

Very common blocking keys are ignored to prevent huge candidate explosions.

Output:
    output/fast_training_candidates.tsv
"""

from pathlib import Path
import sqlite3
import csv
import time
import sys


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

DB_PATH = OUTPUT_DIR / "entity_indexes.sqlite"
OUTPUT_PATH = OUTPUT_DIR / "fast_training_candidates.tsv"

# Keys occurring more than this many times are too broad.
MAX_BLOCK_FREQUENCY = 200


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
    conn.execute("PRAGMA cache_size=-16000")
    conn.execute("PRAGMA mmap_size=1073741824")

    return conn


def create_frequency_indexes(conn):
    """
    Create temporary frequency tables.

    These are TEMP tables, so they do not permanently enlarge
    entity_indexes.sqlite.
    """

    print()
    print("=" * 80)
    print("PREPARING BLOCK FREQUENCIES")
    print("=" * 80)

    blocking_fields = [
        "name_norm",
        "name_compact",
        "name_tokens",
        "address_norm",
        "address_compact",
        "address_digits",
    ]

    for field in blocking_fields:

        print(f"  Preparing {field} ...", flush=True)

        table_name = f"freq_{field}"

        conn.execute(f"DROP TABLE IF EXISTS temp.{table_name}")

        conn.execute(
            f"""
            CREATE TEMP TABLE {table_name} AS
            SELECT
                {field} AS block_value,
                source,
                COUNT(*) AS freq
            FROM records
            WHERE source IN ('S1', 'S2', 'S3')
              AND {field} IS NOT NULL
              AND {field} <> ''
            GROUP BY {field}, source
            """
        )

        conn.execute(
            f"""
            CREATE INDEX idx_{table_name}
            ON {table_name}(block_value, source)
            """
        )

        conn.commit()

    print()
    print("Frequency tables ready.")


def run_block(
    conn,
    field,
    block_name,
    output_file,
):
    """
    Run one set-based blocking join.

    Only keys whose frequency is <= MAX_BLOCK_FREQUENCY
    are allowed.
    """

    print()
    print("-" * 80)
    print(f"BLOCK: {block_name}")
    print(f"FIELD: {field}")
    print("-" * 80)

    freq_s1 = f"freq_{field}"

    sql = f"""
        SELECT
            s1.entity_id,
            m.entity_id,
            m.source
        FROM records AS s1
        JOIN records AS m
          ON m.source IN ('S2', 'S3')
         AND m.country = s1.country
         AND m.{field} = s1.{field}
        JOIN {freq_s1} AS f1
          ON f1.block_value = s1.{field}
         AND f1.source = 'S1'
        JOIN {freq_s1} AS f2
          ON f2.block_value = m.{field}
         AND f2.source = m.source
        WHERE s1.source = 'S1'
          AND s1.{field} IS NOT NULL
          AND s1.{field} <> ''
          AND f1.freq <= ?
          AND f2.freq <= ?
    """

    start = time.time()
    rows = 0

    cursor = conn.execute(
        sql,
        (
            MAX_BLOCK_FREQUENCY,
            MAX_BLOCK_FREQUENCY,
        ),
    )

    writer = csv.writer(
        output_file,
        delimiter="\t",
        lineterminator="\n",
    )

    for s1_id, matched_id, matched_source in cursor:

        writer.writerow(
            [
                s1_id,
                matched_id,
                matched_source,
                block_name,
            ]
        )

        rows += 1

        if rows % 500_000 == 0:
            elapsed = time.time() - start
            print(
                f"    {rows:,} candidates "
                f"| {elapsed / 60:.2f} min",
                flush=True,
            )

    elapsed = time.time() - start

    print(
        f"  Completed: {rows:,} candidates "
        f"in {elapsed / 60:.2f} min",
        flush=True,
    )

    return rows


def deduplicate_candidates():
    """
    Deduplicate the TSV using SQLite.

    The same pair can be discovered by multiple blocking rules.
    We keep the first blocking rule encountered.
    """

    print()
    print("=" * 80)
    print("DEDUPLICATING CANDIDATES")
    print("=" * 80)

    temp_db = OUTPUT_DIR / "candidate_dedup.sqlite"

    if temp_db.exists():
        temp_db.unlink()

    conn = sqlite3.connect(str(temp_db))

    conn.execute(
        """
        CREATE TABLE candidates (
            s1_entity_id TEXT NOT NULL,
            matched_entity_id TEXT NOT NULL,
            matched_source TEXT NOT NULL,
            block_type TEXT NOT NULL,
            PRIMARY KEY (
                s1_entity_id,
                matched_entity_id,
                matched_source
            )
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_candidates_s1
        ON candidates(s1_entity_id)
        """
    )

    conn.commit()

    start = time.time()
    inserted = 0

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as input_file:

        reader = csv.DictReader(
            input_file,
            delimiter="\t",
        )

        batch = []

        for row in reader:

            batch.append(
                (
                    row["s1_entity_id"],
                    row["matched_entity_id"],
                    row["matched_source"],
                    row["block_type"],
                )
            )

            if len(batch) >= 50_000:

                conn.executemany(
                    """
                    INSERT OR IGNORE INTO candidates
                    (
                        s1_entity_id,
                        matched_entity_id,
                        matched_source,
                        block_type
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    batch,
                )

                inserted += len(batch)
                batch.clear()

        if batch:
            conn.executemany(
                """
                INSERT OR IGNORE INTO candidates
                (
                    s1_entity_id,
                    matched_entity_id,
                    matched_source,
                    block_type
                )
                VALUES (?, ?, ?, ?)
                """,
                batch,
            )

    conn.commit()

    final_count = conn.execute(
        "SELECT COUNT(*) FROM candidates"
    ).fetchone()[0]

    elapsed = time.time() - start

    print(f"Raw candidate rows:       {inserted:,}")
    print(f"Unique candidate pairs:   {final_count:,}")
    print(f"Deduplication time:       {elapsed / 60:.2f} min")

    final_path = OUTPUT_DIR / "fast_training_candidates_unique.tsv"

    if final_path.exists():
        final_path.unlink()

    with final_path.open(
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

        cursor = conn.execute(
            """
            SELECT
                s1_entity_id,
                matched_entity_id,
                matched_source,
                block_type
            FROM candidates
            ORDER BY s1_entity_id
            """
        )

        for row in cursor:
            writer.writerow(row)

    conn.close()

    # Keep the SQLite dedup database because it is useful for
    # later processing and is much smaller than entity_indexes.sqlite.
    print()
    print("Final candidate file:")
    print(final_path)

    return final_path, final_count


def inspect_final_candidates(path):
    print()
    print("=" * 80)
    print("FINAL CANDIDATE SUMMARY")
    print("=" * 80)

    total = 0
    s1_ids = set()

    # This set should remain manageable because it contains only
    # 2.2M string IDs. We only use it for summary statistics.
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            total += 1
            s1_ids.add(row["s1_entity_id"])

    print(f"Candidate pairs:          {total:,}")
    print(f"S1 entities represented:  {len(s1_ids):,}")

    if len(s1_ids):
        print(
            f"Average candidates/S1:   "
            f"{total / len(s1_ids):.2f}"
        )


def main():

    print("=" * 80)
    print("BUSINESS ENTITY RESOLUTION")
    print("STEP 5 — SET-BASED FAST CANDIDATE GENERATION")
    print("=" * 80)

    print()
    print("Database:")
    print(DB_PATH)

    print()
    print("Maximum blocking frequency:")
    print(f"  {MAX_BLOCK_FREQUENCY}")

    print()
    print("Output:")
    print(OUTPUT_PATH)

    if OUTPUT_PATH.exists():
        print()
        print("Removing previous candidate file...")
        OUTPUT_PATH.unlink()

    conn = open_database()

    try:

        create_frequency_indexes(conn)

        block_definitions = [
            ("name_norm", "exact_name"),
            ("name_compact", "exact_compact_name"),
            ("name_tokens", "name_tokens"),
            ("address_norm", "exact_address"),
            ("address_compact", "exact_compact_address"),
            ("address_digits", "address_digits"),
        ]

        start = time.time()
        total_raw = 0

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

            for field, block_name in block_definitions:

                count = run_block(
                    conn,
                    field,
                    block_name,
                    output_file,
                )

                total_raw += count

        elapsed = time.time() - start

        print()
        print("=" * 80)
        print("RAW CANDIDATE GENERATION COMPLETE")
        print("=" * 80)

        print(f"Raw candidate rows: {total_raw:,}")
        print(f"Runtime:             {elapsed / 60:.2f} min")

    finally:
        conn.close()

    final_path, final_count = deduplicate_candidates()

    inspect_final_candidates(final_path)

    print()
    print("=" * 80)
    print("STEP 5 COMPLETE")
    print("=" * 80)

    print()
    print("Use this file for the next stage:")
    print(final_path)

    print()
    print("The original SQLite index was NOT modified.")


if __name__ == "__main__":
    main()
