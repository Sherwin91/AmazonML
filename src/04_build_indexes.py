from pathlib import Path
import re
import sqlite3
import unicodedata

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"
OUTPUT_DIR = BASE_DIR / "output"
DB_PATH = OUTPUT_DIR / "entity_indexes.sqlite"

SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
SOURCE3 = TRAIN_DIR / "train_source3.tsv"

CHUNK_SIZE = 50_000


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Conservative Unicode-aware normalization.

    Important:
    - preserves non-Latin scripts
    - case-folds
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

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(r"\s+", " ", value).strip()

    return value


def compact_text(value):
    """
    Normalized text with spaces removed.

    Useful for:
        ABC Private Limited
        ABCPrivateLimited
    """
    return normalize_text(value).replace(" ", "")


def token_signature(value):
    """
    Sorted unique normalized tokens.

    Example:

        'United Clinic Bny'
        -> 'bny clinic united'

    Helps with reordered business names.
    """
    normalized = normalize_text(value)

    if not normalized:
        return ""

    tokens = normalized.split()

    return " ".join(sorted(set(tokens)))


def digit_signature(value):
    """
    Extracts all numeric sequences.

    Example:

        '1374 A, Town Of Richmond'
        -> '1374'

        '19705-19707 Unit 9'
        -> '19705 19707 9'
    """
    if value is None:
        return ""

    value = str(value)

    numbers = re.findall(r"\d+", value)

    return " ".join(numbers)


# ============================================================
# DATABASE
# ============================================================

def create_database():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        print(f"Removing existing database: {DB_PATH}")
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)

    connection.execute("PRAGMA journal_mode=WAL;")
    connection.execute("PRAGMA synchronous=NORMAL;")
    connection.execute("PRAGMA temp_store=FILE;")
    connection.execute("PRAGMA cache_size=-50000;")

    connection.execute(
        """
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            business_name TEXT,
            business_address TEXT,
            country TEXT,
            name_norm TEXT,
            name_compact TEXT,
            name_tokens TEXT,
            address_norm TEXT,
            address_compact TEXT,
            address_tokens TEXT,
            address_digits TEXT
        )
        """
    )

    connection.commit()

    return connection


# ============================================================
# INDEXES
# ============================================================

def create_indexes(connection):
    print("\nCreating SQLite indexes...")

    indexes = [
        """
        CREATE INDEX idx_records_source
        ON records(source)
        """,

        """
        CREATE INDEX idx_records_country
        ON records(country)
        """,

        """
        CREATE INDEX idx_records_source_country
        ON records(source, country)
        """,

        """
        CREATE INDEX idx_records_name_norm
        ON records(name_norm)
        """,

        """
        CREATE INDEX idx_records_name_compact
        ON records(name_compact)
        """,

        """
        CREATE INDEX idx_records_name_tokens
        ON records(name_tokens)
        """,

        """
        CREATE INDEX idx_records_address_norm
        ON records(address_norm)
        """,

        """
        CREATE INDEX idx_records_address_compact
        ON records(address_compact)
        """,

        """
        CREATE INDEX idx_records_address_digits
        ON records(address_digits)
        """,
    ]

    for sql in indexes:
        connection.execute(sql)

    connection.commit()

    print("Indexes created.")


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(connection, filepath, source_name):
    print("\n" + "=" * 80)
    print(f"BUILDING INDEX: {source_name}")
    print("=" * 80)

    total_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            filepath,
            sep="\t",
            dtype=str,
            chunksize=CHUNK_SIZE,
            keep_default_na=False,
        ),
        start=1,
    ):
        # ----------------------------------------------------
        # Ensure required columns exist
        # ----------------------------------------------------

        required_columns = [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]

        for column in required_columns:
            if column not in chunk.columns:
                raise RuntimeError(
                    f"Missing column '{column}' in {filepath}"
                )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        chunk["name_norm"] = chunk["business_name"].map(
            normalize_text
        )

        chunk["name_compact"] = chunk["business_name"].map(
            compact_text
        )

        chunk["name_tokens"] = chunk["business_name"].map(
            token_signature
        )

        chunk["address_norm"] = chunk["business_address"].map(
            normalize_text
        )

        chunk["address_compact"] = chunk["business_address"].map(
            compact_text
        )

        chunk["address_tokens"] = chunk["business_address"].map(
            token_signature
        )

        chunk["address_digits"] = chunk["business_address"].map(
            digit_signature
        )

        chunk["source"] = source_name

        # ----------------------------------------------------
        # Select columns
        # ----------------------------------------------------

        output_columns = [
            "entity_id",
            "source",
            "business_name",
            "business_address",
            "country",
            "name_norm",
            "name_compact",
            "name_tokens",
            "address_norm",
            "address_compact",
            "address_tokens",
            "address_digits",
        ]

        chunk = chunk[output_columns]

        # ----------------------------------------------------
        # Insert into SQLite
        # ----------------------------------------------------

        chunk.to_sql(
            "records",
            connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=2_000,
        )

        total_rows += len(chunk)

        if chunk_number % 10 == 0:
            print(
                f"  chunks: {chunk_number:,} | "
                f"rows: {total_rows:,}"
            )

    print(f"\n{source_name} complete.")
    print(f"Rows indexed: {total_rows:,}")

    return total_rows


# ============================================================
# DATABASE VALIDATION
# ============================================================

def validate_database(connection):
    print("\n" + "=" * 80)
    print("DATABASE VALIDATION")
    print("=" * 80)

    total = connection.execute(
        "SELECT COUNT(*) FROM records"
    ).fetchone()[0]

    print(f"Total records: {total:,}")

    print("\nRecords by source:")

    rows = connection.execute(
        """
        SELECT source, COUNT(*)
        FROM records
        GROUP BY source
        ORDER BY source
        """
    ).fetchall()

    for source, count in rows:
        print(f"  {source}: {count:,}")

    print("\nRecords by source/country:")

    rows = connection.execute(
        """
        SELECT source, country, COUNT(*)
        FROM records
        GROUP BY source, country
        ORDER BY source, country
        """
    ).fetchall()

    for source, country, count in rows:
        print(
            f"  {source:8s} | "
            f"{country:10s} | "
            f"{count:,}"
        )

    print("\nMissing normalized values:")

    rows = connection.execute(
        """
        SELECT
            SUM(CASE WHEN name_norm = '' THEN 1 ELSE 0 END),
            SUM(CASE WHEN address_norm = '' THEN 1 ELSE 0 END)
        FROM records
        """
    ).fetchone()

    print(f"  Empty normalized names: {rows[0]:,}")
    print(f"  Empty normalized addresses: {rows[1]:,}")

    print("\nSample indexed records:")

    rows = connection.execute(
        """
        SELECT
            entity_id,
            source,
            business_name,
            country,
            name_norm,
            name_tokens,
            address_digits
        FROM records
        ORDER BY RANDOM()
        LIMIT 10
        """
    ).fetchall()

    for row in rows:
        print("-" * 60)
        print(f"ID:       {row[0]}")
        print(f"Source:   {row[1]}")
        print(f"Name:     {row[2]}")
        print(f"Country:  {row[3]}")
        print(f"NameNorm: {row[4]}")
        print(f"Tokens:   {row[5]}")
        print(f"Digits:   {row[6]}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("BUSINESS ENTITY RESOLUTION")
    print("STEP 4 — MEMORY-SAFE SQLITE INDEX BUILD")
    print("=" * 80)

    print(f"\nDatabase:")
    print(f"  {DB_PATH}")

    connection = create_database()

    try:
        s1_count = process_source(
            connection,
            SOURCE1,
            "S1",
        )

        s2_count = process_source(
            connection,
            SOURCE2,
            "S2",
        )

        s3_count = process_source(
            connection,
            SOURCE3,
            "S3",
        )

        connection.commit()

        print("\n" + "=" * 80)
        print("ALL SOURCE FILES INDEXED")
        print("=" * 80)

        print(f"S1: {s1_count:,}")
        print(f"S2: {s2_count:,}")
        print(f"S3: {s3_count:,}")

        create_indexes(connection)

        connection.commit()

        validate_database(connection)

    finally:
        connection.close()

    print("\n" + "=" * 80)
    print("STEP 4 COMPLETE")
    print("=" * 80)

    print(f"\nSQLite database created:")
    print(f"  {DB_PATH}")

    if DB_PATH.exists():
        size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        print(f"Database size: {size_mb:.1f} MB")

    print("\nNext step:")
    print("  Build candidate generation from the SQLite indexes.")


if __name__ == "__main__":
    main()