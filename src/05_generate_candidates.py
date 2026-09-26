#!/usr/bin/env python3

"""
BUSINESS ENTITY RESOLUTION
STEP 5 — MEMORY-SAFE CANDIDATE GENERATION

Purpose
-------
Generate candidate S1 -> S2/S3 pairs from the training data.

Important challenge requirements:
- S1 is the reference entity.
- Each S1 may have zero, one, or many matches.
- Candidates must only connect S1 -> S2/S3.
- Country is used as a blocking signal, not hard-coded to US/India.
- No external data is used.
- Processing is streaming / SQLite-backed because RAM is limited.

Blocking strategies
-------------------
1. Exact normalized business name + country
2. Exact compact business name + country
3. Exact normalized address + country
4. Exact compact address + country
5. Name prefix + country
6. Address prefix + country
7. Name first-token + country, but only when the block is small

The last two strategies are capped to prevent huge candidate explosions.

Output
------
output/training_candidate_pairs.tsv

Columns:
    s1_entity_id
    matched_entity_id
    matched_source
    block_type

The same pair can be found by multiple blocks, but is written only once.
"""

from pathlib import Path
import sqlite3
import csv
import time
import sys


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

DB_PATH = OUTPUT_DIR / "entity_indexes.sqlite"
OUTPUT_PATH = OUTPUT_DIR / "training_candidate_pairs.tsv"

CHUNK_SIZE = 10_000

# Candidate explosion protection.
#
# If a blocking value occurs more often than this, we do NOT expand the
# complete block.
MAX_BLOCK_SIZE = 500

# For prefix blocks we use a shorter limit because prefixes can be common.
MAX_PREFIX_BLOCK_SIZE = 300

# Maximum number of candidates retained for one S1 from each block type.
MAX_CANDIDATES_PER_BLOCK = 200

# Global safety cap per S1.
#
# This is deliberately generous because true matches can be multiple.
MAX_TOTAL_CANDIDATES_PER_S1 = 1_000

# Prefix length for blocking.
PREFIX_LENGTH = 4

# Minimum token length for first-token blocking.
MIN_TOKEN_LENGTH = 4


# =============================================================================
# DATABASE
# =============================================================================

def connect_database():
    if not DB_PATH.exists():
        print("ERROR: SQLite database does not exist:")
        print(DB_PATH)
        print()
        print("Run:")
        print("  python src/04_build_indexes.py")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-30000")

    return conn


# =============================================================================
# TABLES
# =============================================================================

def create_block_tables(conn):
    """
    Create compact helper tables.

    We deliberately create only lightweight blocking information.

    This avoids building a gigantic token-to-record table.
    """

    print()
    print("=" * 80)
    print("CREATING BLOCKING HELPER TABLES")
    print("=" * 80)

    conn.execute("DROP TABLE IF EXISTS blocking_keys")

    conn.execute(
        """
        CREATE TABLE blocking_keys (
            entity_id TEXT NOT NULL,
            source TEXT NOT NULL,
            country TEXT NOT NULL,

            name_prefix TEXT,
            address_prefix TEXT,
            first_name_token TEXT
        )
        """
    )

    print("Creating blocking-key rows from existing SQLite records...")

    cursor = conn.execute(
        """
        SELECT
            entity_id,
            source,
            country,
            name_norm,
            address_norm
        FROM records
        """
    )

    insert_sql = """
        INSERT INTO blocking_keys (
            entity_id,
            source,
            country,
            name_prefix,
            address_prefix,
            first_name_token
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """

    buffer = []
    processed = 0
    start = time.time()

    while True:
        rows = cursor.fetchmany(CHUNK_SIZE)

        if not rows:
            break

        for entity_id, source, country, name_norm, address_norm in rows:

            name_prefix = ""
            address_prefix = ""
            first_name_token = ""

            if name_norm:
                compact_name = name_norm.replace(" ", "")

                if compact_name:
                    name_prefix = compact_name[:PREFIX_LENGTH]

                tokens = name_norm.split()

                # Choose the first reasonably informative token.
                for token in tokens:
                    compact_token = token.replace(" ", "")

                    if len(compact_token) >= MIN_TOKEN_LENGTH:
                        first_name_token = compact_token
                        break

            if address_norm:
                compact_address = address_norm.replace(" ", "")

                if compact_address:
                    address_prefix = compact_address[:PREFIX_LENGTH]

            buffer.append(
                (
                    entity_id,
                    source,
                    country,
                    name_prefix,
                    address_prefix,
                    first_name_token,
                )
            )

            processed += 1

        if len(buffer) >= 10_000:
            conn.executemany(insert_sql, buffer)
            conn.commit()
            buffer.clear()

        if processed % 500_000 == 0:
            elapsed = time.time() - start
            rate = processed / max(elapsed, 1)

            print(
                f"  helper rows: {processed:,} "
                f"| {rate:,.0f} rows/sec"
            )

    if buffer:
        conn.executemany(insert_sql, buffer)
        conn.commit()

    print(f"Blocking helper rows: {processed:,}")

    print()
    print("Creating blocking indexes...")

    conn.execute(
        """
        CREATE INDEX idx_block_name_prefix
        ON blocking_keys(country, name_prefix, source)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_block_address_prefix
        ON blocking_keys(country, address_prefix, source)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_block_first_token
        ON blocking_keys(country, first_name_token, source)
        """
    )

    conn.execute(
        """
        CREATE INDEX idx_block_entity
        ON blocking_keys(entity_id, source)
        """
    )

    conn.commit()

    print("Blocking indexes created.")


# =============================================================================
# BLOCK FREQUENCY CACHE
# =============================================================================

class BlockFrequencyCache:
    """
    Small in-memory cache.

    We never load all blocking frequencies into RAM.
    """

    def __init__(self, conn):
        self.conn = conn
        self.cache = {}

    def get_name_prefix_count(self, country, key):
        cache_key = ("name_prefix", country, key)

        if cache_key in self.cache:
            return self.cache[cache_key]

        value = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM blocking_keys
            WHERE country = ?
              AND name_prefix = ?
            """,
            (country, key),
        ).fetchone()[0]

        self.cache[cache_key] = value
        return value

    def get_address_prefix_count(self, country, key):
        cache_key = ("address_prefix", country, key)

        if cache_key in self.cache:
            return self.cache[cache_key]

        value = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM blocking_keys
            WHERE country = ?
              AND address_prefix = ?
            """,
            (country, key),
        ).fetchone()[0]

        self.cache[cache_key] = value
        return value

    def get_first_token_count(self, country, key):
        cache_key = ("first_token", country, key)

        if cache_key in self.cache:
            return self.cache[cache_key]

        value = self.conn.execute(
            """
            SELECT COUNT(*)
            FROM blocking_keys
            WHERE country = ?
              AND first_name_token = ?
            """,
            (country, key),
        ).fetchone()[0]

        self.cache[cache_key] = value
        return value


# =============================================================================
# CANDIDATE QUERIES
# =============================================================================

def add_exact_name_candidates(conn, country, name_norm, candidate_set):
    if not name_norm:
        return 0

    rows = conn.execute(
        """
        SELECT entity_id, source
        FROM records
        WHERE country = ?
          AND name_norm = ?
          AND source IN ('S2', 'S3')
        """,
        (country, name_norm),
    ).fetchmany(MAX_CANDIDATES_PER_BLOCK)

    added = 0

    for entity_id, source in rows:
        key = (entity_id, source)

        if key not in candidate_set:
            candidate_set[key] = "exact_name"
            added += 1

    return added


def add_exact_compact_name_candidates(conn, country, name_compact, candidate_set):
    if not name_compact:
        return 0

    rows = conn.execute(
        """
        SELECT entity_id, source
        FROM records
        WHERE country = ?
          AND name_compact = ?
          AND source IN ('S2', 'S3')
        """,
        (country, name_compact),
    ).fetchmany(MAX_CANDIDATES_PER_BLOCK)

    added = 0

    for entity_id, source in rows:
        key = (entity_id, source)

        if key not in candidate_set:
            candidate_set[key] = "exact_compact_name"
            added += 1

    return added


def add_exact_address_candidates(conn, country, address_norm, candidate_set):
    if not address_norm:
        return 0

    rows = conn.execute(
        """
        SELECT entity_id, source
        FROM records
        WHERE country = ?
          AND address_norm = ?
          AND source IN ('S2', 'S3')
        """,
        (country, address_norm),
    ).fetchmany(MAX_CANDIDATES_PER_BLOCK)

    added = 0

    for entity_id, source in rows:
        key = (entity_id, source)

        if key not in candidate_set:
            candidate_set[key] = "exact_address"
            added += 1

    return added


def add_exact_compact_address_candidates(
    conn,
    country,
    address_compact,
    candidate_set,
):
    if not address_compact:
        return 0

    rows = conn.execute(
        """
        SELECT entity_id, source
        FROM records
        WHERE country = ?
          AND address_compact = ?
          AND source IN ('S2', 'S3')
        """,
        (country, address_compact),
    ).fetchmany(MAX_CANDIDATES_PER_BLOCK)

    added = 0

    for entity_id, source in rows:
        key = (entity_id, source)

        if key not in candidate_set:
            candidate_set[key] = "exact_compact_address"
            added += 1

    return added


def add_name_prefix_candidates(
    conn,
    frequency_cache,
    country,
    name_norm,
    candidate_set,
):
    if not name_norm:
        return 0

    compact_name = name_norm.replace(" ", "")

    if len(compact_name) < PREFIX_LENGTH:
        return 0

    prefix = compact_name[:PREFIX_LENGTH]

    count = frequency_cache.get_name_prefix_count(country, prefix)

    if count == 0 or count > MAX_PREFIX_BLOCK_SIZE:
        return 0

    rows = conn.execute(
        """
        SELECT entity_id, source
        FROM blocking_keys
        WHERE country = ?
          AND name_prefix = ?
          AND source IN ('S2', 'S3')
        LIMIT ?
        """,
        (country, prefix, MAX_CANDIDATES_PER_BLOCK),
    )

    added = 0

    for entity_id, source in rows:
        key = (entity_id, source)

        if key not in candidate_set:
            candidate_set[key] = "name_prefix"
            added += 1

    return added


def add_address_prefix_candidates(
    conn,
    frequency_cache,
    country,
    address_norm,
    candidate_set,
):
    if not address_norm:
        return 0

    compact_address = address_norm.replace(" ", "")

    if len(compact_address) < PREFIX_LENGTH:
        return 0

    prefix = compact_address[:PREFIX_LENGTH]

    count = frequency_cache.get_address_prefix_count(
        country,
        prefix,
    )

    if count == 0 or count > MAX_PREFIX_BLOCK_SIZE:
        return 0

    rows = conn.execute(
        """
        SELECT entity_id, source
        FROM blocking_keys
        WHERE country = ?
          AND address_prefix = ?
          AND source IN ('S2', 'S3')
        LIMIT ?
        """,
        (country, prefix, MAX_CANDIDATES_PER_BLOCK),
    )

    added = 0

    for entity_id, source in rows:
        key = (entity_id, source)

        if key not in candidate_set:
            candidate_set[key] = "address_prefix"
            added += 1

    return added


def add_first_token_candidates(
    conn,
    frequency_cache,
    country,
    name_norm,
    candidate_set,
):
    if not name_norm:
        return 0

    token = ""

    for part in name_norm.split():
        compact = part.replace(" ", "")

        if len(compact) >= MIN_TOKEN_LENGTH:
            token = compact
            break

    if not token:
        return 0

    count = frequency_cache.get_first_token_count(
        country,
        token,
    )

    if count == 0 or count > MAX_BLOCK_SIZE:
        return 0

    rows = conn.execute(
        """
        SELECT entity_id, source
        FROM blocking_keys
        WHERE country = ?
          AND first_name_token = ?
          AND source IN ('S2', 'S3')
        LIMIT ?
        """,
        (country, token, MAX_CANDIDATES_PER_BLOCK),
    )

    added = 0

    for entity_id, source in rows:
        key = (entity_id, source)

        if key not in candidate_set:
            candidate_set[key] = "first_name_token"
            added += 1

    return added


# =============================================================================
# GENERATE CANDIDATES
# =============================================================================

def generate_candidates(conn):
    print()
    print("=" * 80)
    print("GENERATING TRAINING CANDIDATES")
    print("=" * 80)

    print()
    print(f"Output:")
    print(f"  {OUTPUT_PATH}")

    print()
    print("Safety limits:")
    print(f"  Maximum block size:       {MAX_BLOCK_SIZE:,}")
    print(f"  Maximum prefix block:     {MAX_PREFIX_BLOCK_SIZE:,}")
    print(f"  Max candidates / block:   {MAX_CANDIDATES_PER_BLOCK:,}")
    print(f"  Max candidates / S1:      {MAX_TOTAL_CANDIDATES_PER_S1:,}")

    # -------------------------------------------------------------------------
    # Remove old output.
    # -------------------------------------------------------------------------

    if OUTPUT_PATH.exists():
        print()
        print("Removing previous candidate file...")
        OUTPUT_PATH.unlink()

    frequency_cache = BlockFrequencyCache(conn)

    # -------------------------------------------------------------------------
    # S1 streaming cursor.
    # -------------------------------------------------------------------------

    cursor = conn.execute(
        """
        SELECT
            entity_id,
            country,
            name_norm,
            name_compact,
            address_norm,
            address_compact
        FROM records
        WHERE source = 'S1'
        ORDER BY rowid
        """
    )

    total_s1 = 0
    total_candidates = 0

    block_counts = {
        "exact_name": 0,
        "exact_compact_name": 0,
        "exact_address": 0,
        "exact_compact_address": 0,
        "name_prefix": 0,
        "address_prefix": 0,
        "first_name_token": 0,
    }

    s1_with_candidates = 0
    s1_without_candidates = 0

    start_time = time.time()

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
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

            rows = cursor.fetchmany(CHUNK_SIZE)

            if not rows:
                break

            for (
                s1_id,
                country,
                name_norm,
                name_compact,
                address_norm,
                address_compact,
            ) in rows:

                total_s1 += 1

                candidate_set = {}

                # -------------------------------------------------------------
                # 1. Exact normalized name
                # -------------------------------------------------------------

                added = add_exact_name_candidates(
                    conn,
                    country,
                    name_norm,
                    candidate_set,
                )

                block_counts["exact_name"] += added

                # -------------------------------------------------------------
                # 2. Exact compact name
                # -------------------------------------------------------------

                if len(candidate_set) < MAX_TOTAL_CANDIDATES_PER_S1:

                    added = add_exact_compact_name_candidates(
                        conn,
                        country,
                        name_compact,
                        candidate_set,
                    )

                    block_counts["exact_compact_name"] += added

                # -------------------------------------------------------------
                # 3. Exact normalized address
                # -------------------------------------------------------------

                if len(candidate_set) < MAX_TOTAL_CANDIDATES_PER_S1:

                    added = add_exact_address_candidates(
                        conn,
                        country,
                        address_norm,
                        candidate_set,
                    )

                    block_counts["exact_address"] += added

                # -------------------------------------------------------------
                # 4. Exact compact address
                # -------------------------------------------------------------

                if len(candidate_set) < MAX_TOTAL_CANDIDATES_PER_S1:

                    added = add_exact_compact_address_candidates(
                        conn,
                        country,
                        address_compact,
                        candidate_set,
                    )

                    block_counts["exact_compact_address"] += added

                # -------------------------------------------------------------
                # 5. Name prefix
                # -------------------------------------------------------------

                if len(candidate_set) < MAX_TOTAL_CANDIDATES_PER_S1:

                    added = add_name_prefix_candidates(
                        conn,
                        frequency_cache,
                        country,
                        name_norm,
                        candidate_set,
                    )

                    block_counts["name_prefix"] += added

                # -------------------------------------------------------------
                # 6. Address prefix
                # -------------------------------------------------------------

                if len(candidate_set) < MAX_TOTAL_CANDIDATES_PER_S1:

                    added = add_address_prefix_candidates(
                        conn,
                        frequency_cache,
                        country,
                        address_norm,
                        candidate_set,
                    )

                    block_counts["address_prefix"] += added

                # -------------------------------------------------------------
                # 7. First informative name token
                # -------------------------------------------------------------

                if len(candidate_set) < MAX_TOTAL_CANDIDATES_PER_S1:

                    added = add_first_token_candidates(
                        conn,
                        frequency_cache,
                        country,
                        name_norm,
                        candidate_set,
                    )

                    block_counts["first_name_token"] += added

                # -------------------------------------------------------------
                # Final safety cap
                # -------------------------------------------------------------

                if len(candidate_set) > MAX_TOTAL_CANDIDATES_PER_S1:

                    candidate_items = list(
                        candidate_set.items()
                    )[:MAX_TOTAL_CANDIDATES_PER_S1]

                    candidate_set = dict(candidate_items)

                # -------------------------------------------------------------
                # Write candidates immediately.
                #
                # Nothing is accumulated across S1 records.
                # -------------------------------------------------------------

                if candidate_set:
                    s1_with_candidates += 1

                    for (matched_id, matched_source), block_type in (
                        candidate_set.items()
                    ):
                        writer.writerow(
                            [
                                s1_id,
                                matched_id,
                                matched_source,
                                block_type,
                            ]
                        )

                        total_candidates += 1

                else:
                    s1_without_candidates += 1

                # -------------------------------------------------------------
                # Progress
                # -------------------------------------------------------------

                if total_s1 % 10_000 == 0:

                    elapsed = time.time() - start_time
                    rate = total_s1 / max(elapsed, 1)

                    print(
                        f"  S1 processed: {total_s1:,} "
                        f"| candidates: {total_candidates:,} "
                        f"| rate: {rate:,.0f} S1/sec",
                        flush=True,
                    )

    elapsed = time.time() - start_time

    return (
        total_s1,
        total_candidates,
        s1_with_candidates,
        s1_without_candidates,
        block_counts,
        elapsed,
    )


# =============================================================================
# VALIDATE OUTPUT
# =============================================================================

def validate_output():
    print()
    print("=" * 80)
    print("VALIDATING CANDIDATE OUTPUT")
    print("=" * 80)

    if not OUTPUT_PATH.exists():
        print("ERROR: candidate file was not created.")
        sys.exit(1)

    rows = 0
    unique_pairs = set()

    # IMPORTANT:
    # We do NOT load the whole file.
    #
    # unique_pairs is used only for a bounded sample of validation.
    # The candidate file itself can be very large.

    sample_limit = 100_000

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        expected_columns = {
            "s1_entity_id",
            "matched_entity_id",
            "matched_source",
            "block_type",
        }

        if set(reader.fieldnames or []) != expected_columns:
            print("ERROR: unexpected output columns.")
            print("Found:", reader.fieldnames)
            sys.exit(1)

        for row in reader:

            rows += 1

            if rows <= sample_limit:
                pair = (
                    row["s1_entity_id"],
                    row["matched_entity_id"],
                    row["matched_source"],
                )

                unique_pairs.add(pair)

            if row["matched_source"] not in {"S2", "S3"}:
                print(
                    "ERROR: invalid matched source:",
                    row["matched_source"],
                )
                sys.exit(1)

    print(f"Candidate rows: {rows:,}")

    if rows:
        print(
            f"First {min(rows, sample_limit):,} sampled pairs "
            f"are unique: {len(unique_pairs):,}"
        )

    print("Output validation passed.")


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("BUSINESS ENTITY RESOLUTION")
    print("STEP 5 — MEMORY-SAFE CANDIDATE GENERATION")
    print("=" * 80)

    print()
    print("Database:")
    print(f"  {DB_PATH}")

    print()
    print("IMPORTANT:")
    print("This step uses TRAIN S1/S2/S3 data.")
    print("The purpose is to create the candidate pool for model training.")
    print()

    conn = connect_database()

    try:

        # ---------------------------------------------------------------------
        # Create helper blocking table.
        # ---------------------------------------------------------------------

        create_block_tables(conn)

        # ---------------------------------------------------------------------
        # Generate candidates.
        # ---------------------------------------------------------------------

        (
            total_s1,
            total_candidates,
            s1_with_candidates,
            s1_without_candidates,
            block_counts,
            elapsed,
        ) = generate_candidates(conn)

        # ---------------------------------------------------------------------
        # Validate.
        # ---------------------------------------------------------------------

        validate_output()

        # ---------------------------------------------------------------------
        # Summary.
        # ---------------------------------------------------------------------

        print()
        print("=" * 80)
        print("STEP 5 COMPLETE")
        print("=" * 80)

        print()
        print(f"S1 processed:              {total_s1:,}")
        print(f"S1 with candidates:        {s1_with_candidates:,}")
        print(f"S1 without candidates:     {s1_without_candidates:,}")
        print(f"Candidate rows:            {total_candidates:,}")

        if total_s1:
            avg = total_candidates / total_s1
        else:
            avg = 0

        print(f"Average candidates / S1:   {avg:.2f}")

        print()
        print("Candidates added by block:")
        print("-" * 50)

        for block_type, count in block_counts.items():
            print(f"{block_type:28s}: {count:,}")

        print()
        print(f"Runtime: {elapsed / 60:.2f} minutes")

        print()
        print("Candidate file:")
        print(f"  {OUTPUT_PATH}")

        print()
        print("Next step:")
        print("  Build training labels + matching features.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()