#!/usr/bin/env python3

import csv
import os
import re
import time
import unicodedata
from collections import defaultdict

from rapidfuzz import fuzz


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAIRS_FILE = os.path.join(
    BASE,
    "output",
    "training_pairs_small.tsv"
)

OUTPUT_FILE = os.path.join(
    BASE,
    "output",
    "real_raw_training_features.tsv"
)

S1_FILE = os.path.join(
    BASE,
    "dataset",
    "train",
    "train_source1.tsv"
)

S2_FILE = os.path.join(
    BASE,
    "dataset",
    "train",
    "train_source2.tsv"
)

S3_FILE = os.path.join(
    BASE,
    "dataset",
    "train",
    "train_source3.tsv"
)


# ---------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------

def normalize_text(value):
    if value is None:
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKD",
        value
    )

    value = "".join(
        ch for ch in value
        if not unicodedata.combining(ch)
    )

    value = value.lower()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


def compact_text(value):
    value = normalize_text(value)

    return re.sub(
        r"\s+",
        "",
        value
    )


def digit_string(value):
    if value is None:
        return ""

    return "".join(
        ch for ch in str(value)
        if ch.isdigit()
    )


def tokens(value):
    value = normalize_text(value)

    if not value:
        return []

    return value.split()


def token_jaccard(a, b):
    ta = set(tokens(a))
    tb = set(tokens(b))

    if not ta and not tb:
        return 1.0

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def token_overlap(a, b):
    ta = set(tokens(a))
    tb = set(tokens(b))

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / min(
        len(ta),
        len(tb)
    )


def digit_overlap(a, b):
    da = set(digit_string(a))
    db = set(digit_string(b))

    if not da or not db:
        return 0.0

    return len(da & db) / len(da | db)


def safe_ratio(a, b):
    if not a or not b:
        return 0.0

    return len(a) / len(b)


def prefix_similarity(a, b, n=5):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    n = min(n, len(a), len(b))

    if n == 0:
        return 0.0

    return 1.0 if a[:n] == b[:n] else 0.0


def suffix_similarity(a, b, n=5):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    n = min(n, len(a), len(b))

    if n == 0:
        return 0.0

    return 1.0 if a[-n:] == b[-n:] else 0.0


def char_jaccard(a, b, n=3):
    a = compact_text(a)
    b = compact_text(b)

    if len(a) < n or len(b) < n:
        return 0.0

    ga = {
        a[i:i+n]
        for i in range(len(a) - n + 1)
    }

    gb = {
        b[i:i+n]
        for i in range(len(b) - n + 1)
    }

    if not ga or not gb:
        return 0.0

    return len(ga & gb) / len(ga | gb)


# ---------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------

def make_features(
    s1,
    other,
    source
):

    n1 = normalize_text(s1["business_name"])
    n2 = normalize_text(other["business_name"])

    c1 = compact_text(s1["business_name"])
    c2 = compact_text(other["business_name"])

    a1 = normalize_text(s1["business_address"])
    a2 = normalize_text(other["business_address"])

    ac1 = compact_text(s1["business_address"])
    ac2 = compact_text(other["business_address"])

    d1 = digit_string(s1["business_address"])
    d2 = digit_string(other["business_address"])

    country1 = normalize_text(s1["country"])
    country2 = normalize_text(other["country"])

    name_missing = int(not n1 or not n2)
    address_missing = int(not a1 or not a2)

    name_ratio = (
        fuzz.ratio(n1, n2) / 100.0
        if n1 and n2 else 0.0
    )

    name_token_sort = (
        fuzz.token_sort_ratio(n1, n2) / 100.0
        if n1 and n2 else 0.0
    )

    name_token_set = (
        fuzz.token_set_ratio(n1, n2) / 100.0
        if n1 and n2 else 0.0
    )

    name_wratio = (
        fuzz.WRatio(n1, n2) / 100.0
        if n1 and n2 else 0.0
    )

    name_partial = (
        fuzz.partial_ratio(n1, n2) / 100.0
        if n1 and n2 else 0.0
    )

    address_ratio = (
        fuzz.ratio(a1, a2) / 100.0
        if a1 and a2 else 0.0
    )

    address_token_sort = (
        fuzz.token_sort_ratio(a1, a2) / 100.0
        if a1 and a2 else 0.0
    )

    address_token_set = (
        fuzz.token_set_ratio(a1, a2) / 100.0
        if a1 and a2 else 0.0
    )

    address_wratio = (
        fuzz.WRatio(a1, a2) / 100.0
        if a1 and a2 else 0.0
    )

    address_partial = (
        fuzz.partial_ratio(a1, a2) / 100.0
        if a1 and a2 else 0.0
    )

    name_exact = int(
        bool(n1) and n1 == n2
    )

    name_compact_exact = int(
        bool(c1) and c1 == c2
    )

    address_exact = int(
        bool(a1) and a1 == a2
    )

    address_compact_exact = int(
        bool(ac1) and ac1 == ac2
    )

    address_digits_exact = int(
        bool(d1) and d1 == d2
    )

    same_country = int(
        country1 == country2
    )

    name_length_diff = abs(
        len(n1) - len(n2)
    )

    address_length_diff = abs(
        len(a1) - len(a2)
    )

    name_length_ratio = (
        min(len(n1), len(n2))
        /
        max(len(n1), len(n2))
        if n1 and n2 else 0.0
    )

    address_length_ratio = (
        min(len(a1), len(a2))
        /
        max(len(a1), len(a2))
        if a1 and a2 else 0.0
    )

    name_prefix = prefix_similarity(
        n1,
        n2
    )

    name_suffix = suffix_similarity(
        n1,
        n2
    )

    address_prefix = prefix_similarity(
        a1,
        a2
    )

    address_suffix = suffix_similarity(
        a1,
        a2
    )

    name_jaccard = token_jaccard(
        n1,
        n2
    )

    name_overlap = token_overlap(
        n1,
        n2
    )

    address_jaccard = token_jaccard(
        a1,
        a2
    )

    address_overlap = token_overlap(
        a1,
        a2
    )

    name_char_jaccard = char_jaccard(
        n1,
        n2,
        3
    )

    address_char_jaccard = char_jaccard(
        a1,
        a2,
        3
    )

    address_digit_overlap = digit_overlap(
        a1,
        a2
    )

    # Strong independent signals
    name_strong = int(
        name_ratio >= 0.90
        or name_token_set >= 0.95
        or name_exact
        or name_compact_exact
    )

    address_strong = int(
        address_ratio >= 0.90
        or address_token_set >= 0.95
        or address_exact
        or address_compact_exact
        or address_digits_exact
    )

    # Agreement between independent evidence
    name_address_agreement = (
        name_ratio * address_ratio
    )

    combined_similarity = (
        0.55 * name_wratio
        +
        0.45 * address_wratio
    )

    # Useful interactions
    exact_name_and_country = (
        name_exact * same_country
    )

    exact_address_and_country = (
        address_exact * same_country
    )

    strong_name_and_address = (
        name_strong * address_strong
    )

    name_address_max = max(
        name_wratio,
        address_wratio
    )

    name_address_min = min(
        name_wratio,
        address_wratio
    )

    return [
        same_country,
        name_exact,
        name_compact_exact,
        name_ratio,
        name_token_sort,
        name_token_set,
        name_wratio,
        name_partial,

        address_exact,
        address_compact_exact,
        address_digits_exact,
        address_ratio,
        address_token_sort,
        address_token_set,
        address_wratio,
        address_partial,

        name_jaccard,
        name_overlap,
        address_jaccard,
        address_overlap,

        name_char_jaccard,
        address_char_jaccard,

        name_length_diff,
        address_length_diff,
        name_length_ratio,
        address_length_ratio,

        name_prefix,
        name_suffix,
        address_prefix,
        address_suffix,

        address_digit_overlap,

        name_missing,
        address_missing,

        name_strong,
        address_strong,

        name_address_agreement,
        combined_similarity,

        exact_name_and_country,
        exact_address_and_country,
        strong_name_and_address,

        name_address_max,
        name_address_min,

        source == "S2",
        source == "S3",
    ]


FEATURE_NAMES = [
    "same_country",
    "name_exact",
    "name_compact_exact",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_wratio",
    "name_partial",

    "address_exact",
    "address_compact_exact",
    "address_digits_exact",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "address_wratio",
    "address_partial",

    "name_token_jaccard",
    "name_token_overlap",
    "address_token_jaccard",
    "address_token_overlap",

    "name_char_jaccard",
    "address_char_jaccard",

    "name_length_diff",
    "address_length_diff",
    "name_length_ratio",
    "address_length_ratio",

    "name_prefix",
    "name_suffix",
    "address_prefix",
    "address_suffix",

    "address_digit_overlap",

    "name_missing",
    "address_missing",

    "name_strong",
    "address_strong",

    "name_address_agreement",
    "combined_similarity",

    "exact_name_and_country",
    "exact_address_and_country",
    "strong_name_and_address",

    "name_address_max",
    "name_address_min",

    "is_source_s2",
    "is_source_s3",
]


# ---------------------------------------------------------------------
# LOAD PAIRS
# ---------------------------------------------------------------------

print("=" * 80)
print("STEP 13 — REAL RAW-STRING FEATURE EXTRACTION")
print("=" * 80)

start = time.time()

print("\nLoading training pairs...")

pairs = []

required_s1 = set()
required_s2 = set()
required_s3 = set()

with open(
    PAIRS_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.DictReader(
        f,
        delimiter="\t"
    )

    for row in reader:

        s1_id = row["s1_entity_id"]
        matched_id = row["matched_entity_id"]
        source = row["matched_source"]

        pairs.append(
            (
                s1_id,
                matched_id,
                source,
                int(row["label"])
            )
        )

        required_s1.add(s1_id)

        if source == "S2":
            required_s2.add(matched_id)
        else:
            required_s3.add(matched_id)

print(f"Pairs: {len(pairs):,}")
print(f"Required S1: {len(required_s1):,}")
print(f"Required S2: {len(required_s2):,}")
print(f"Required S3: {len(required_s3):,}")


# ---------------------------------------------------------------------
# STREAM RECORDS
# ---------------------------------------------------------------------

def load_required(
    filename,
    required_ids,
    label
):

    records = {}

    scanned = 0

    print(
        f"\nReading {label}: {filename}"
    )

    with open(
        filename,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:

            scanned += 1

            entity_id = row["entity_id"]

            if entity_id in required_ids:

                records[entity_id] = {
                    "entity_id": entity_id,
                    "business_name": row.get(
                        "business_name",
                        ""
                    ) or "",
                    "business_address": row.get(
                        "business_address",
                        ""
                    ) or "",
                    "country": row.get(
                        "country",
                        ""
                    ) or "",
                }

    print(
        f"Scanned: {scanned:,}"
    )

    print(
        f"Loaded: {len(records):,}"
    )

    missing = len(required_ids - records.keys())

    print(
        f"Missing: {missing:,}"
    )

    if missing:
        raise RuntimeError(
            f"{label}: {missing} required IDs missing"
        )

    return records


s1_records = load_required(
    S1_FILE,
    required_s1,
    "S1"
)

s2_records = load_required(
    S2_FILE,
    required_s2,
    "S2"
)

s3_records = load_required(
    S3_FILE,
    required_s3,
    "S3"
)


# ---------------------------------------------------------------------
# GENERATE FEATURES
# ---------------------------------------------------------------------

print("\nGenerating REAL raw-string features...")

header = [
    "s1_entity_id",
    "matched_entity_id",
    "matched_source",
    "label",
] + FEATURE_NAMES

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as out:

    writer = csv.writer(
        out,
        delimiter="\t"
    )

    writer.writerow(header)

    for i, (
        s1_id,
        matched_id,
        source,
        label
    ) in enumerate(pairs, 1):

        s1 = s1_records[s1_id]

        if source == "S2":
            other = s2_records[matched_id]
        else:
            other = s3_records[matched_id]

        features = make_features(
            s1,
            other,
            source
        )

        writer.writerow(
            [
                s1_id,
                matched_id,
                source,
                label,
            ]
            + features
        )

        if i % 10000 == 0:
            print(
                f"  {i:,} / {len(pairs):,}"
            )


elapsed = time.time() - start

print("\n" + "=" * 80)
print("STEP 13 COMPLETE")
print("=" * 80)

print(f"\nRows: {len(pairs):,}")
print(f"Features: {len(FEATURE_NAMES)}")

print(
    f"\nOutput:\n{OUTPUT_FILE}"
)

print(
    f"\nRuntime: {elapsed:.2f}s"
)

print("\nIMPORTANT:")
print(
    "These features are computed directly from the original "
    "business names and addresses."
)