
#!/bin/sh
# POSIX shell wrapper + gawk core.
# Scan <ROOT>/<model>/<run_ts>/<ov_variant>/<cb_variant>/metrics_per_question.csv
# and keep only rows whose "similarity" is smaller than a threshold.
#
# Usage:
#   sh filter_per_question_by_similarity.sh [ROOT_DIR] [THRESHOLD] [OUTPUT_CSV]
#
# Defaults:
#   ROOT_DIR="."
#   THRESHOLD=0.1
#   OUTPUT_CSV="metrics_per_question_below_threshold.csv"
#
# Notes:
# - Requires gawk (for FPAT, robust CSV parsing).
# - CSV parsing is quote-aware; handles BOM and CRLF.
# - Output header:
#   model_name,run_ts,ov_variant,cb_variant,rel_path,similarity,prompts,source_model,optimized_model

set -eu

ROOT_DIR="${1:-.}"
THRESHOLD="${2:-0.1}"
OUT_CSV="${3:-metrics_per_question_below_threshold.csv}"

# Ensure gawk is available
if ! command -v gawk >/dev/null 2>&1; then
  echo "Error: gawk is required but not found. Please install gawk and retry." >&2
  exit 1
fi

# Build a sorted list of metrics_per_question.csv (relative paths), inside ROOT_DIR
TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

(
  cd "$ROOT_DIR"
  find . -type f -name 'metrics_per_question.csv' -print | sed 's|^\./||' | LC_ALL=C sort > "$TMP_LIST"
)

if [ ! -s "$TMP_LIST" ]; then
  echo "No metrics_per_question.csv files found under: $ROOT_DIR" >&2
  exit 0
fi

(
  cd "$ROOT_DIR"
  files=$(cat "$TMP_LIST")
  [ -z "$files" ] && { echo "No metrics_per_question.csv files found under: $ROOT_DIR" >&2; exit 0; }

  gawk -v OUT="$OUT_CSV" -v THRESH="$THRESHOLD" '
  BEGIN {
    # Quote-aware CSV parsing; treat commas inside quotes correctly
    FPAT = "([^,]*)|(\"[^\"]*\")"
    OFS  = ","

    # Write header once (truncate file)
    hdr = "model_name,run_ts,ov_variant,cb_variant,rel_path,similarity,prompts,source_model,optimized_model"
    print hdr > OUT
  }

  # Normalize CRLF per record
  { sub(/\r$/, "", $0) }

  # Helper: dequote a CSV field (remove surrounding quotes, unescape "")
  function dequote(s) {
    if (s ~ /^".*"$/) {
      sub(/^"/, "", s)
      sub(/"$/, "", s)
      gsub(/""/, "\"", s)
    }
    return s
  }

  # Helper: escape for CSV output (quote if contains comma, quote, or newline)
  function csv_escape(s,   needs) {
    needs = (s ~ /[",\n]/)
    gsub(/"/, "\"\"", s)
    if (needs) return "\"" s "\""
    return s
  }

  # Track per-file column indexes
  FNR == 1 {
    # Strip UTF-8 BOM in first header cell if any
    sub(/^\xEF\xBB\xBF/, "", $1)

    # Locate columns by name (exact match)
    idx_sim = idx_prompts = idx_src = idx_opt = 0
    for (i = 1; i <= NF; i++) {
      key = dequote($i)
      if (key == "similarity")       idx_sim     = i
      else if (key == "prompts")     idx_prompts = i
      else if (key == "source_model") idx_src    = i
      else if (key == "optimized_model") idx_opt = i
    }
    next
  }

  # Data rows: filter by similarity < THRESH
  FNR > 1 {
    if (idx_sim == 0) next  # no similarity column, skip file
    sim_raw = dequote($idx_sim)
    # Convert to number safely (coerce on arithmetic)
    sim_val = sim_raw + 0.0

    if (sim_val < THRESH) {
      # Derive context from path: <model>/<run_ts>/<ov>/<cb>/metrics_per_question.csv
      n = split(FILENAME, p, "/")
      model = (n >= 1 ? p[1] : "")
      runts = (n >= 2 ? p[2] : "")
      ov    = (n >= 3 ? p[3] : "")
      cb    = (n >= 4 ? p[4] : "")
      rel   = FILENAME

      prompts = (idx_prompts ? dequote($idx_prompts) : "")
      src_md  = (idx_src     ? dequote($idx_src)     : "")
      opt_md  = (idx_opt     ? dequote($idx_opt)     : "")

      row = ""
      row = row csv_escape(model)
      row = row OFS csv_escape(runts)
      row = row OFS csv_escape(ov)
      row = row OFS csv_escape(cb)
      row = row OFS csv_escape(rel)
      row = row OFS sim_val
      row = row OFS csv_escape(prompts)
      row = row OFS csv_escape(src_md)
      row = row OFS csv_escape(opt_md)

      print row >> OUT
    }
  }
  ' $files
)

echo "✅ Filtered rows (similarity < ${THRESHOLD}) written to: $OUT_CSV"
