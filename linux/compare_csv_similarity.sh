#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <csv_file_a> <csv_file_b> [output_csv]"
  exit 1
fi

CSV_A="$1"
CSV_B="$2"
OUT_CSV="${3:-comparison_similarity.csv}"
KEY_COL="rel_path"

if [[ ! -f "$CSV_A" ]]; then
  echo "Error: file not found: $CSV_A" >&2
  exit 2
fi

if [[ ! -f "$CSV_B" ]]; then
  echo "Error: file not found: $CSV_B" >&2
  exit 2
fi

python3 - "$CSV_A" "$CSV_B" "$KEY_COL" "$OUT_CSV" <<'PY'
import csv
import re
import sys
from pathlib import Path


def read_csv(path: str, key_col: str):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV or missing header")
        if key_col not in reader.fieldnames:
            raise ValueError(f"{path}: missing required key column '{key_col}'")

        rows = {}
        duplicate_keys = []
        for i, row in enumerate(reader, start=2):
            key = (row.get(key_col) or "").strip()
            if not key:
                continue
            if key in rows:
                duplicate_keys.append((key, i))
            rows[key] = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        return reader.fieldnames, rows, duplicate_keys


def pct(n, d):
    return 0.0 if d == 0 else (100.0 * n / d)


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def normalize_rel_path(p: str) -> str:
    # Normalize common run-specific prefixes and timestamps in rel_path so
    # two runs can still be compared by logical scenario.
    s = (p or "").strip()
    s = re.sub(r"^outputs\.wwb_\d{8}_\d{6}/", "", s)
    # Replace full timestamps and bare dates anywhere in the path
    s = re.sub(r"\d{8}_\d{6}", "<RUN_TS>", s)
    s = re.sub(r"\d{8}", "<DATE>", s)
    return s


def build_normalized_map(rows, key_col):
    out = {}
    collisions = []
    for k, row in rows.items():
        nk = normalize_rel_path(row.get(key_col, k))
        if nk in out:
            collisions.append(nk)
        out[nk] = row
    return out, collisions


def main():
    a_path, b_path, key_col, out_csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    a_cols, a_rows, a_dups = read_csv(a_path, key_col)
    b_cols, b_rows, b_dups = read_csv(b_path, key_col)

    a_keys = set(a_rows)
    b_keys = set(b_rows)
    common_keys = sorted(a_keys & b_keys)
    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)
    union_keys = a_keys | b_keys

    shared_cols = [c for c in a_cols if c in b_cols and c != key_col and c and c.strip()]

    print("=== CSV Similarity Report ===")
    print(f"A: {a_path}")
    print(f"B: {b_path}")
    print(f"Key column: {key_col}")
    print()

    print("--- Key Coverage ---")
    print(f"Rows in A: {len(a_keys)}")
    print(f"Rows in B: {len(b_keys)}")
    print(f"Common keys: {len(common_keys)}")
    print(f"Only in A: {len(only_a)}")
    print(f"Only in B: {len(only_b)}")
    key_jaccard = 0.0 if not union_keys else len(common_keys) / len(union_keys)
    print(f"Key-set Jaccard similarity: {key_jaccard:.6f} ({key_jaccard*100:.2f}%)")
    print()

    if a_dups:
        print(f"Warning: duplicate keys in A (showing first 5): {a_dups[:5]}")
    if b_dups:
        print(f"Warning: duplicate keys in B (showing first 5): {b_dups[:5]}")
    if a_dups or b_dups:
        print()

    if not shared_cols:
        print("No shared non-key columns between files.")
        return

    # If exact key match has no overlap, try normalized rel_path mapping.
    use_normalized = False
    if len(common_keys) == 0:
        a_norm_rows, a_collisions = build_normalized_map(a_rows, key_col)
        b_norm_rows, b_collisions = build_normalized_map(b_rows, key_col)
        norm_common = sorted(set(a_norm_rows) & set(b_norm_rows))
        if norm_common:
            use_normalized = True
            print("Info: no exact rel_path overlap; using normalized rel_path fallback.")
            print("Normalization rules: strip leading outputs.wwb_<timestamp>/ and replace /<timestamp>/ with /<RUN_TS>/")
            if a_collisions:
                print(f"Warning: normalized-key collisions in A (count={len(a_collisions)}), last row kept")
            if b_collisions:
                print(f"Warning: normalized-key collisions in B (count={len(b_collisions)}), last row kept")
            print(f"Common normalized keys: {len(norm_common)}")
            print()
            a_rows = a_norm_rows
            b_rows = b_norm_rows
            common_keys = norm_common

    print("--- Column Similarity (on common keys) ---")
    print(f"Shared columns: {len(shared_cols)}")

    total_cells = len(common_keys) * len(shared_cols)
    exact_cells = 0
    fully_equal_rows = 0

    for col in shared_cols:
        exact = 0
        numeric_pairs = 0
        numeric_sim_sum = 0.0

        for k in common_keys:
            av = a_rows[k].get(col, "")
            bv = b_rows[k].get(col, "")

            if av == bv:
                exact += 1

            af = to_float(av)
            bf = to_float(bv)
            if af is not None and bf is not None:
                denom = max(abs(af), abs(bf), 1e-12)
                sim = 1.0 - abs(af - bf) / denom
                if sim < 0.0:
                    sim = 0.0
                numeric_sim_sum += sim
                numeric_pairs += 1

        exact_cells += exact
        exact_ratio = 0.0 if not common_keys else exact / len(common_keys)

        if numeric_pairs > 0:
            num_sim = numeric_sim_sum / numeric_pairs
            print(f"{col}: exact={exact}/{len(common_keys)} ({exact_ratio*100:.2f}%), numeric_similarity={num_sim*100:.2f}% over {numeric_pairs} numeric rows")
        else:
            print(f"{col}: exact={exact}/{len(common_keys)} ({exact_ratio*100:.2f}%)")

    # Row-level equality across all shared columns
    for k in common_keys:
        if all(a_rows[k].get(c, "") == b_rows[k].get(c, "") for c in shared_cols):
            fully_equal_rows += 1

    print()
    print("--- Overall Similarity ---")
    if total_cells > 0:
        print(f"Cell exact-match similarity: {exact_cells}/{total_cells} ({pct(exact_cells, total_cells):.2f}%)")
    else:
        print("Cell exact-match similarity: N/A (no comparable cells)")
    print(f"Row exact-match similarity (across shared columns): {fully_equal_rows}/{len(common_keys)} ({pct(fully_equal_rows, len(common_keys)):.2f}%)")

    if only_a:
        print()
        print(f"Sample keys only in A (up to 10): {only_a[:10]}")
    if only_b:
        print(f"Sample keys only in B (up to 10): {only_b[:10]}")

    # Output per-row similarity CSV using normalized rel_path as key
    sim_col = "similarity"
    has_sim_col = sim_col in shared_cols
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if has_sim_col:
            writer.writerow([
                "rel_path_norm",
                "similarity_a",
                "similarity_b",
                "abs_diff",
                "rel_diff_pct",
            ])
            for k in common_keys:
                av = a_rows[k].get(sim_col, "")
                bv = b_rows[k].get(sim_col, "")
                af = to_float(av)
                bf = to_float(bv)
                if af is not None and bf is not None:
                    abs_diff = abs(af - bf)
                    denom = max(abs(af), abs(bf), 1e-12)
                    rel_diff_pct = 100.0 * abs_diff / denom
                else:
                    abs_diff = ""
                    rel_diff_pct = ""
                writer.writerow([k, av, bv, abs_diff, rel_diff_pct])
        else:
            writer.writerow(["rel_path_norm"])
            for k in common_keys:
                writer.writerow([k])

    print()
    print(f"Wrote per-row similarity CSV: {out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(3)
PY
