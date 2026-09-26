from pathlib import Path
import gc
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"

FILES = {
    "TRAIN SOURCE 1": TRAIN_DIR / "train_source1.tsv",
    "TRAIN SOURCE 2": TRAIN_DIR / "train_source2.tsv",
    "TRAIN SOURCE 3": TRAIN_DIR / "train_source3.tsv",
}

CHUNK_SIZE = 50_000


def inspect_file(label, filepath):
    print("\n" + "=" * 80)
    print(label)
    print("=" * 80)
    print(f"File: {filepath}")

    if not filepath.exists():
        print("FILE NOT FOUND")
        return

    total_rows = 0

    missing_counts = {}
    country_counts = {}

    name_count = 0
    name_total_length = 0
    name_min_length = None
    name_max_length = 0

    address_count = 0
    address_total_length = 0
    address_min_length = None
    address_max_length = 0

    first_rows = None

    chunk_number = 0

    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE,
        keep_default_na=True,
        low_memory=True,
    ):
        chunk_number += 1

        if first_rows is None:
            first_rows = chunk.head(5).copy()

        rows = len(chunk)
        total_rows += rows

        # ---------------------------------------------------------
        # Missing values
        # ---------------------------------------------------------
        for column in chunk.columns:
            count = int(chunk[column].isna().sum())
            missing_counts[column] = missing_counts.get(column, 0) + count

        # ---------------------------------------------------------
        # Country distribution
        # ---------------------------------------------------------
        if "country" in chunk.columns:
            counts = chunk["country"].value_counts(dropna=False)

            for country, count in counts.items():
                key = "<MISSING>" if pd.isna(country) else str(country)
                country_counts[key] = country_counts.get(key, 0) + int(count)

        # ---------------------------------------------------------
        # Business name statistics
        # ---------------------------------------------------------
        if "business_name" in chunk.columns:
            names = chunk["business_name"].dropna().astype(str)
            lengths = names.str.len()

            name_count += len(lengths)

            if len(lengths) > 0:
                name_total_length += int(lengths.sum())

                current_min = int(lengths.min())
                current_max = int(lengths.max())

                if name_min_length is None:
                    name_min_length = current_min
                else:
                    name_min_length = min(name_min_length, current_min)

                name_max_length = max(name_max_length, current_max)

        # ---------------------------------------------------------
        # Business address statistics
        # ---------------------------------------------------------
        if "business_address" in chunk.columns:
            addresses = chunk["business_address"].dropna().astype(str)
            lengths = addresses.str.len()

            address_count += len(lengths)

            if len(lengths) > 0:
                address_total_length += int(lengths.sum())

                current_min = int(lengths.min())
                current_max = int(lengths.max())

                if address_min_length is None:
                    address_min_length = current_min
                else:
                    address_min_length = min(address_min_length, current_min)

                address_max_length = max(address_max_length, current_max)

        del chunk

        if chunk_number % 10 == 0:
            print(
                f"Processed {total_rows:,} rows...",
                flush=True
            )

        gc.collect()

    # -------------------------------------------------------------
    # Results
    # -------------------------------------------------------------
    print("\nRESULTS")
    print("-" * 80)

    print(f"Rows: {total_rows:,}")

    print("\nColumns:")
    for column in missing_counts:
        print(f"  - {column}")

    print("\nMissing values:")
    for column, count in missing_counts.items():
        percentage = (count / total_rows) * 100 if total_rows else 0
        print(
            f"  {column}: {count:,} "
            f"({percentage:.2f}%)"
        )

    print("\nCountry distribution:")
    for country, count in sorted(
        country_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        percentage = (count / total_rows) * 100 if total_rows else 0
        print(
            f"  {country}: {count:,} "
            f"({percentage:.2f}%)"
        )

    print("\nBusiness name:")
    if name_count:
        print(f"  Non-missing: {name_count:,}")
        print(
            f"  Mean length: "
            f"{name_total_length / name_count:.2f}"
        )
        print(f"  Min length: {name_min_length}")
        print(f"  Max length: {name_max_length}")

    print("\nBusiness address:")
    if address_count:
        print(f"  Non-missing: {address_count:,}")
        print(
            f"  Mean length: "
            f"{address_total_length / address_count:.2f}"
        )
        print(f"  Min length: {address_min_length}")
        print(f"  Max length: {address_max_length}")

    print("\nFirst 5 rows:")
    if first_rows is not None:
        print(first_rows.to_string(index=False))

    print("\nFinished:", label)


def main():
    print("=" * 80)
    print("MEMORY-SAFE DATASET INSPECTION")
    print("=" * 80)
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print(f"Train directory: {TRAIN_DIR}")

    for label, filepath in FILES.items():
        inspect_file(label, filepath)
        gc.collect()

    print("\n" + "=" * 80)
    print("ALL TRAIN FILES INSPECTED")
    print("=" * 80)


if __name__ == "__main__":
    main()