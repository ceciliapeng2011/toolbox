
#!/bin/sh
# POSIX shell wrapper + gawk core.
# Summarize all metrics.csv files under a ROOT directory (multiple models),
# and produce a second CSV with failure comparisons of sparse-xattention vs dense.
#
# Structure expected:
#   <ROOT>/<model_name>/<run_ts>/<ov_variant>/<cb_variant>/metrics.csv
#
# Usage:
#   sh summarize_metrics.sh [ROOT_DIR] [SUMMARY_CSV] [FAILURES_CSV]
# Defaults:
#   ROOT_DIR="."
#   SUMMARY_CSV="metrics_summary.csv"
#   FAILURES_CSV="metrics_failures.csv"
#
# Notes:
# - Requires gawk (for FPAT, ENDFILE, associative arrays).
# - CSV parsing is quote-aware; handles BOM and CRLF.
# - Context columns in summary: model_name, run_ts, ov_variant, cb_variant, rel_path
# - Failures compare sparse-xattention similarity to dense baseline (same group):
#   Failure if: sparse_similarity <= dense_similarity - 0.1
#   Dense matching: prefer same mtoks; else highest-sim dense in group.

set -eu

ROOT_DIR="${1:-.}"
OUT_SUMMARY="${2:-metrics_summary.csv}"
OUT_FAILURES="${3:-metrics_failures.csv}"

# Ensure gawk is available
if ! command -v gawk >/dev/null 2>&1; then
  echo "Error: gawk is required but not found. Please install gawk and retry." >&2
  exit 1
fi

# Build a sorted list of metrics.csv (relative paths), inside ROOT_DIR
TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

(
  cd "$ROOT_DIR"
  find . -type f -name 'metrics.csv' -print | sed 's|^\./||' | LC_ALL=C sort > "$TMP_LIST"
)

if [ ! -s "$TMP_LIST" ]; then
  echo "No metrics.csv files found under: $ROOT_DIR" >&2
  exit 0
fi

(
  cd "$ROOT_DIR"
  files=$(cat "$TMP_LIST")
  [ -z "$files" ] && { echo "No metrics.csv files found under: $ROOT_DIR" >&2; exit 0; }

  gawk -v FAIL="$OUT_FAILURES" '
  BEGIN {
    FPAT = "([^,]*)|(\"[^\"]*\")"  # quote-aware CSV split
    OFS  = ","

    # Fixed context columns for summary
    ctx_cols[1] = "model_name"
    ctx_cols[2] = "run_ts"
    ctx_cols[3] = "ov_variant"
    ctx_cols[4] = "cb_variant"
    ctx_cols[5] = "rel_path"

    file_idx = 0
    header_count = 0
  }

  # Normalize CRLF per record
  { sub(/\r$/, "", $0) }

  # Header (union) buildup
  FNR == 1 {
    sub(/^\xEF\xBB\xBF/, "", $1)  # strip BOM in first column if present

    delete file_hdr
    for (i = 1; i <= NF; i++) {
      file_hdr[i] = $i
      if (!( ($i) in seen_header)) {
        seen_header[$i] = 1
        header_count++
        headers[header_count] = $i
      }
    }

    # reset last row buffer
    last_nf = 0
    delete last_vals
  }

  # Track last data row
  {
    last_nf = NF
    for (i = 1; i <= NF; i++) {
      val = $i
      if (val ~ /^".*"$/) { sub(/^"/,"",val); sub(/"$/,"",val); gsub(/""/,"\"",val) }
      last_vals[i] = val
    }
  }

  ENDFILE {
    file_idx++

    # Path parts: <model>/<run_ts>/<ov>/<cb>/metrics.csv
    n = split(FILENAME, p, "/")
    model = (n >= 1 ? p[1] : "")
    runts = (n >= 2 ? p[2] : "")
    ov    = (n >= 3 ? p[3] : "")
    cb    = (n >= 4 ? p[4] : "")
    rel   = FILENAME

    # Persist context
    ctx_model[file_idx] = model
    ctx_runts[file_idx] = runts
    ctx_ov[file_idx]    = ov
    ctx_cb[file_idx]    = cb
    ctx_rel[file_idx]   = rel

    # Build per-file key->value map from header and last row
    for (i = 1; i <= last_nf; i++) {
      key = file_hdr[i]
      vals[file_idx, key] = last_vals[i]
    }

    # Extract similarity (if present)
    sim_present[file_idx] = ((file_idx, "similarity") in vals) && (length(vals[file_idx, "similarity"]) > 0)
    sim[file_idx] = (sim_present[file_idx] ? vals[file_idx, "similarity"] + 0.0 : 0.0)  # numeric

    # Extract mtoks from cb_variant (e.g., mtoks-4096)
    mtoks[file_idx] = ""
    if (cb ~ /mtoks-([0-9]+)/) { match(cb, /mtoks-([0-9]+)/, m); mtoks[file_idx] = m[1] }

    # Type flags
    is_dense[file_idx]  = (cb ~ /^cb_dense/)
    is_sparse[file_idx] = (cb ~ /^cb_sparse[-_]xattention/)

    # Group key for comparison
    g = model "|" runts "|" ov
    groups[g] = 1

    # Track file indices per group and type
    if (is_dense[file_idx])  { d_count[g]++; d_list[g, d_count[g]]   = file_idx }
    if (is_sparse[file_idx]) { s_count[g]++; s_list[g, s_count[g]]   = file_idx }
  }

  # CSV escape for safe output
  function csv_escape(s,   needs) {
    needs = (s ~ /[",\n]/)
    gsub(/"/, "\"\"", s)
    if (needs) return "\"" s "\""
    return s
  }

  END {
    # --------- Summary Output (to stdout) ---------
    # Header
    out = ""
    for (i = 1; i <= 5; i++) { out = out ((i==1) ? "" : OFS) ctx_cols[i] }
    for (h = 1; h <= header_count; h++) { out = out OFS headers[h] }
    print out

    # Rows
    for (fi = 1; fi <= file_idx; fi++) {
      row = ""
      row = row csv_escape(ctx_model[fi])
      row = row OFS csv_escape(ctx_runts[fi])
      row = row OFS csv_escape(ctx_ov[fi])
      row = row OFS csv_escape(ctx_cb[fi])
      row = row OFS csv_escape(ctx_rel[fi])

      for (h = 1; h <= header_count; h++) {
        key = headers[h]
        v = ((fi, key) in vals) ? vals[fi, key] : ""
        row = row OFS csv_escape(v)
      }
      print row
    }

    # --------- Failures Output (to FAIL) ---------
    # Header for failures
    fail_hdr = "model_name,run_ts,ov_variant,sparse_cb_variant,dense_cb_variant,sparse_rel_path,dense_rel_path,match_strategy,sparse_similarity,dense_similarity,delta"
    print fail_hdr > FAIL

    # Iterate groups
    for (g in groups) {
      nd = d_count[g] + 0
      ns = s_count[g] + 0

      if (ns == 0 || nd == 0) continue  # need both sides to compare

      # Pre-compute best dense by similarity
      best_dense_idx = 0
      best_dense_sim = -1e99
      for (di = 1; di <= nd; di++) {
        dfi = d_list[g, di]
        if (!sim_present[dfi]) continue
        if (sim[dfi] > best_dense_sim) {
          best_dense_sim = sim[dfi]
          best_dense_idx = dfi
        }
      }
      if (best_dense_idx == 0) continue  # no dense with valid similarity

      # Compare each sparse variant
      for (si = 1; si <= ns; si++) {
        sfi = s_list[g, si]
        if (!sim_present[sfi]) continue

        # Try mtoks match first
        want_mtoks = mtoks[sfi]
        chosen_dfi = 0
        strategy   = "mtoks"
        if (length(want_mtoks) > 0) {
          for (di = 1; di <= nd; di++) {
            dfi = d_list[g, di]
            if (!sim_present[dfi]) continue
            if (mtoks[dfi] == want_mtoks) { chosen_dfi = dfi; break }
          }
        }

        # Fallback: best dense by similarity
        if (chosen_dfi == 0) {
          chosen_dfi = best_dense_idx
          strategy   = "max_dense"
        }

        # Compute delta and check failure: sparse <= dense - 0.1
        delta = sim[sfi] - sim[chosen_dfi]
        if (delta <= -0.1) {
          fail_row = ""
          fail_row = fail_row csv_escape(ctx_model[sfi])
          fail_row = fail_row OFS csv_escape(ctx_runts[sfi])
          fail_row = fail_row OFS csv_escape(ctx_ov[sfi])
          fail_row = fail_row OFS csv_escape(ctx_cb[sfi])
          fail_row = fail_row OFS csv_escape(ctx_cb[chosen_dfi])
          fail_row = fail_row OFS csv_escape(ctx_rel[sfi])
          fail_row = fail_row OFS csv_escape(ctx_rel[chosen_dfi])
          fail_row = fail_row OFS csv_escape(strategy)
          fail_row = fail_row OFS sim[sfi]
          fail_row = fail_row OFS sim[chosen_dfi]
          fail_row = fail_row OFS delta
          print fail_row >> FAIL
        }
      }
    }
  }
  ' $files > "$OUT_SUMMARY"
)

echo "✅ Summary written to: $OUT_SUMMARY"
echo "✅ Failures written to: $OUT_FAILURES"
