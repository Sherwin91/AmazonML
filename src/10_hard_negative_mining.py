from __future__ import annotations

import csv
import os
import re
import time
from collections import defaultdict, Counter

from rapidfuzz.fuzz import ratio


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_FEATURES = os.path.join(BASE, "output", "training_features.tsv")
S1_FILE = os.path.join(BASE, "dataset", "train", "train_source1.tsv")
S2_FILE = os.path.join(BASE, "dataset", "train", "train_source2.tsv")
S3_FILE = os.path.join(BASE, "dataset", "train", "train_source3.tsv")
GT_FILE = os.path.join(BASE, "dataset", "train", "train_ground_truth.tsv")

OUT_FILE = os.path.join(BASE, "output", "hard_negative_pairs.tsv")
META_FILE = os.path.join(BASE, "output", "hard_negative_metadata.txt")

MAX_NEGATIVES_PER_S1 = 4
MAX_CANDIDATES_PER_S1 = 25


# ============================================================
# NORMALIZATION
# ============================================================

def norm(text):
    if not text:
        return ""

    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def compact(text):
    if not text:
        return ""

    return re.sub(r"\W", "", text, flags=re.UNICODE)


def get_digits(text):
    if not text:
        return ""

    return "".join(re.findall(r"\d", text))


# ============================================================
# LOAD REQUIRED S1 IDS
# ============================================================

def load_required_ids():

    print("=" * 80)
    print("STEP 10 — RAM-SAFE HARD NEGATIVE MINING")
    print("=" * 80)

    print()
    print("Configuration:")
    print(f"  S1 entities:              sampled training entities")
    print(f"  Max negatives / S1:       {MAX_NEGATIVES_PER_S1}")
    print(f"  Max candidates / S1:      {MAX_CANDIDATES_PER_S1}")
    print("  Full S2/S3 index:          NO")
    print("  Full S2/S3 records RAM:    NO")
    print("  SQLite candidate joins:    NO")
    print("  RapidFuzz:                 AVAILABLE")

    print()
    print("=" * 80)
    print("LOADING S1 IDS")
    print("=" * 80)

    ids = set()

    with open(
        TRAIN_FEATURES,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:
            ids.add(row["s1_entity_id"])

    print(f"Required S1 entities: {len(ids):,}")

    return ids


# ============================================================
# LOAD S1 RECORDS
# ============================================================

def load_s1(required):

    print()
    print("=" * 80)
    print("LOADING REQUIRED S1 RECORDS")
    print("=" * 80)

    start = time.time()

    records = {}

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        scanned = 0

        for row in reader:

            scanned += 1

            sid = row["entity_id"]

            if sid not in required:
                continue

            name = norm(
                row.get("business_name", "") or ""
            )

            address = norm(
                row.get("business_address", "") or ""
            )

            records[sid] = {
                "country": (
                    row.get("country", "") or ""
                ).lower(),

                "name": name,
                "name_compact": compact(name),

                "address": address,
                "address_compact": compact(address),

                "digits": get_digits(address),
            }

    print(f"S1 rows scanned:   {scanned:,}")
    print(f"S1 records loaded: {len(records):,}")
    print(f"Time: {time.time() - start:.2f}s")

    return records


# ============================================================
# GROUND TRUTH
# ============================================================

def load_truth(required):

    print()
    print("=" * 80)
    print("LOADING GROUND TRUTH")
    print("=" * 80)

    start = time.time()

    truth = defaultdict(set)

    with open(
        GT_FILE,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        scanned = 0

        for row in reader:

            scanned += 1

            sid = row["source1_entity_id"]

            if sid not in required:
                continue

            values = row["matched_entity_ids"].strip()

            if not values:
                continue

            for target in values.split(","):

                target = target.strip()

                if target:
                    truth[sid].add(target)

    print(f"Ground-truth rows scanned: {scanned:,}")
    print(f"S1s with true matches:     {len(truth):,}")
    print(
        f"True match IDs loaded:      "
        f"{sum(len(v) for v in truth.values()):,}"
    )
    print(f"Time: {time.time() - start:.2f}s")

    return truth


# ============================================================
# BUILD ONLY S1 INDEX
# ============================================================

def build_s1_index(records):

    print()
    print("=" * 80)
    print("BUILDING SMALL S1-ONLY BLOCK INDEX")
    print("=" * 80)

    start = time.time()

    index = defaultdict(list)

    for sid, r in records.items():

        country = r["country"]

        # ----------------------------------------------------
        # Exact compact name
        # ----------------------------------------------------

        if r["name_compact"]:

            key = (
                "N",
                country,
                r["name_compact"],
            )

            index[key].append(sid)

        # ----------------------------------------------------
        # Name prefix
        # ----------------------------------------------------

        if len(r["name_compact"]) >= 4:

            key = (
                "NP",
                country,
                r["name_compact"][:4],
            )

            index[key].append(sid)

        # ----------------------------------------------------
        # Exact compact address
        # ----------------------------------------------------

        if r["address_compact"]:

            key = (
                "A",
                country,
                r["address_compact"],
            )

            index[key].append(sid)

        # ----------------------------------------------------
        # Address prefix
        # ----------------------------------------------------

        if len(r["address_compact"]) >= 6:

            key = (
                "AP",
                country,
                r["address_compact"][:6],
            )

            index[key].append(sid)

        # ----------------------------------------------------
        # Address digits
        # ----------------------------------------------------

        if len(r["digits"]) >= 3:

            key = (
                "D",
                country,
                r["digits"],
            )

            index[key].append(sid)

    print(f"S1 block keys: {len(index):,}")
    print(f"Time: {time.time() - start:.2f}s")

    return index


# ============================================================
# GENERATE S1 CANDIDATES FROM ONE SOURCE ROW
# ============================================================

def get_candidate_s1s(row, index):

    country = (
        row.get("country", "") or ""
    ).lower()

    name = norm(
        row.get("business_name", "") or ""
    )

    address = norm(
        row.get("business_address", "") or ""
    )

    nc = compact(name)
    ac = compact(address)
    dg = get_digits(address)

    candidates = set()

    # Exact name
    if nc:

        candidates.update(
            index.get(
                ("N", country, nc),
                []
            )
        )

    # Name prefix
    if len(nc) >= 4:

        candidates.update(
            index.get(
                ("NP", country, nc[:4]),
                []
            )
        )

    # Exact address
    if ac:

        candidates.update(
            index.get(
                ("A", country, ac),
                []
            )
        )

    # Address prefix
    if len(ac) >= 6:

        candidates.update(
            index.get(
                ("AP", country, ac[:6]),
                []
            )
        )

    # Address digits
    if len(dg) >= 3:

        candidates.update(
            index.get(
                ("D", country, dg),
                []
            )
        )

    return candidates, name, nc, address, ac, dg


# ============================================================
# SCORE
# ============================================================

def score_pair(
    s1,
    target_name,
    target_nc,
    target_address,
    target_ac,
    target_digits,
):

    ns = (
        ratio(
            s1["name_compact"],
            target_nc,
        )
        if s1["name_compact"] and target_nc
        else 0.0
    )

    ads = (
        ratio(
            s1["address_compact"],
            target_ac,
        )
        if s1["address_compact"] and target_ac
        else 0.0
    )

    exact_name = (
        bool(s1["name_compact"])
        and s1["name_compact"] == target_nc
    )

    exact_address = (
        bool(s1["address_compact"])
        and s1["address_compact"] == target_ac
    )

    exact_digits = (
        bool(s1["digits"])
        and s1["digits"] == target_digits
    )

    if exact_name:

        score = 100.0

    elif exact_address:

        score = 98.0

    else:

        score = (
            0.60 * ns
            + 0.30 * ads
            + 0.10 * (
                100.0
                if exact_digits
                else 0.0
            )
        )

    return score, ns, ads


# ============================================================
# MINE SOURCE
# ============================================================

def mine_source(
    source_name,
    source_file,
    s1_records,
    s1_index,
    truth,
):

    print()
    print("=" * 80)
    print(f"STREAMING {source_name}")
    print("=" * 80)

    start = time.time()

    # Candidate list accumulated only for S1 IDs.
    candidates = defaultdict(list)

    rows_scanned = 0
    block_hits = 0

    with open(
        source_file,
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            rows_scanned += 1

            target_id = row["entity_id"]

            found_s1s, name, nc, address, ac, dg = (
                get_candidate_s1s(
                    row,
                    s1_index,
                )
            )

            if not found_s1s:
                continue

            block_hits += len(found_s1s)

            # We immediately calculate similarity.
            for sid in found_s1s:

                # Limit candidate accumulation.
                if len(candidates[sid]) >= (
                    MAX_CANDIDATES_PER_S1
                ):
                    continue

                # Never add known true matches.
                if target_id in truth.get(
                    sid,
                    set(),
                ):
                    continue

                s1 = s1_records[sid]

                score, ns, ads = score_pair(
                    s1,
                    name,
                    nc,
                    address,
                    ac,
                    dg,
                )

                # Hard-negative criteria.
                hard = (
                    ns >= 70
                    or ads >= 60
                    or (
                        ns >= 82
                        and ads >= 40
                    )
                    or (
                        ads >= 82
                        and ns >= 40
                    )
                )

                if not hard:
                    continue

                candidates[sid].append(
                    (
                        score,
                        ns,
                        ads,
                        target_id,
                    )
                )

    print(f"Rows scanned:        {rows_scanned:,}")
    print(f"Block hits:          {block_hits:,}")
    print(f"S1s with candidates: {len(candidates):,}")
    print(
        f"Time: {time.time() - start:.2f}s"
    )

    # --------------------------------------------------------
    # Select strongest negatives
    # --------------------------------------------------------

    results = []

    for sid, values in candidates.items():

        values.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        used = set()

        for score, ns, ads, target_id in values:

            if target_id in used:
                continue

            used.add(target_id)

            results.append(
                {
                    "s1_entity_id": sid,
                    "matched_entity_id": target_id,
                    "matched_source": source_name,
                    "hard_score": round(
                        score,
                        4,
                    ),
                    "name_similarity": round(
                        ns,
                        4,
                    ),
                    "address_similarity": round(
                        ads,
                        4,
                    ),
                    "label": 0,
                }
            )

            if len(used) >= MAX_NEGATIVES_PER_S1:
                break

    print(
        f"Hard negatives retained: "
        f"{len(results):,}"
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    required = load_required_ids()

    s1_records = load_s1(required)

    if len(s1_records) != len(required):

        raise RuntimeError(
            "Some required S1 records were not found."
        )

    truth = load_truth(required)

    s1_index = build_s1_index(
        s1_records
    )

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    s2_results = mine_source(
        "S2",
        S2_FILE,
        s1_records,
        s1_index,
        truth,
    )

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    s3_results = mine_source(
        "S3",
        S3_FILE,
        s1_records,
        s1_index,
        truth,
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("COMBINING RESULTS")
    print("=" * 80)

    all_results = (
        s2_results
        + s3_results
    )

    grouped = defaultdict(list)

    for row in all_results:

        grouped[
            row["s1_entity_id"]
        ].append(row)

    final_results = []

    for sid, rows in grouped.items():

        rows.sort(
            key=lambda x: x["hard_score"],
            reverse=True,
        )

        # Maximum 4 across S2 + S3.
        final_results.extend(
            rows[
                :MAX_NEGATIVES_PER_S1
            ]
        )

    # --------------------------------------------------------
    # Safety validation
    # --------------------------------------------------------

    seen = set()

    for row in final_results:

        pair = (
            row["s1_entity_id"],
            row["matched_entity_id"],
        )

        if pair in seen:

            raise RuntimeError(
                f"Duplicate pair: {pair}"
            )

        seen.add(pair)

        if row["matched_entity_id"] in truth.get(
            row["s1_entity_id"],
            set(),
        ):

            raise RuntimeError(
                f"GROUND TRUTH LEAK: {pair}"
            )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    with open(
        OUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow(
            [
                "s1_entity_id",
                "matched_entity_id",
                "matched_source",
                "hard_score",
                "name_similarity",
                "address_similarity",
                "label",
            ]
        )

        for row in final_results:

            writer.writerow(
                [
                    row["s1_entity_id"],
                    row["matched_entity_id"],
                    row["matched_source"],
                    row["hard_score"],
                    row["name_similarity"],
                    row["address_similarity"],
                    row["label"],
                ]
            )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    source_counts = Counter(
        row["matched_source"]
        for row in final_results
    )

    if final_results:

        scores = [
            row["hard_score"]
            for row in final_results
        ]

        names = [
            row["name_similarity"]
            for row in final_results
        ]

        addresses = [
            row["address_similarity"]
            for row in final_results
        ]

        score_mean = sum(scores) / len(scores)
        name_mean = sum(names) / len(names)
        address_mean = sum(addresses) / len(addresses)

    else:

        score_mean = 0
        name_mean = 0
        address_mean = 0

    elapsed = (
        time.time()
        - overall_start
    )

    metadata = f"""
STEP 10 — RAM-SAFE HARD NEGATIVE MINING

Required S1 entities: {len(required):,}
S1 records loaded: {len(s1_records):,}

S2 hard negatives: {source_counts.get("S2", 0):,}
S3 hard negatives: {source_counts.get("S3", 0):,}

Final hard negatives: {len(final_results):,}
S1s represented: {len(grouped):,}

Mean hard score: {score_mean:.4f}
Mean name similarity: {name_mean:.4f}
Mean address similarity: {address_mean:.4f}

Ground-truth leakage: NONE
Duplicate pairs: NONE

Total runtime: {elapsed:.2f}s
"""

    with open(
        META_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(metadata)

    print()
    print("=" * 80)
    print("STEP 10 COMPLETE")
    print("=" * 80)

    print(metadata)

    print(f"Output: {OUT_FILE}")
    print(f"Metadata: {META_FILE}")

    print()
    print(
        "NEXT: STEP 11 — HARD-NEGATIVE FEATURE EXTRACTION"
    )


if __name__ == "__main__":
    main()