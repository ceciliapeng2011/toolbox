#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="${1:-}"
if [[ -z "$LOG_DIR" ]]; then
  echo "Usage: $0 <log_dir>" >&2
  exit 1
fi

OUT_CSV="${LOG_DIR%/}.csv"
MODEL_PREFIX=".__home__ceciliapeng__llm_irs__"
PROMPT_MARKER=".__home__ceciliapeng__3rdparty__x-attention__eval__efficiency__"

tmp_csv=$(mktemp)
printf "ov_config,first_token_ms,second_token_ms,second_tokens_throughput,model_name\n" > "$tmp_csv"

find "$LOG_DIR" -type f | while read -r file; do
  line=$(grep -n "1st token latency:" "$file" | tail -n 1 || true)
  base_name=$(basename "$file")
  ov_config=${base_name%%${MODEL_PREFIX}*}
  if [[ "$ov_config" == "$base_name" ]]; then
    ov_config=""
  fi

  model_name=${base_name#*${MODEL_PREFIX}}
  if [[ "$model_name" == "$base_name" ]]; then
    model_name=""
  fi
  if [[ -n "$model_name" && "$model_name" == *"$PROMPT_MARKER"* ]]; then
    model_name=${model_name%%${PROMPT_MARKER}*}
  fi

  if [[ -z "$line" ]]; then
    printf "%s,,,,%s\n" "$ov_config" "$model_name" >> "$tmp_csv"
    continue
  fi

  first_ms=$(printf "%s" "$line" | sed -E 's/.*1st token latency: ([0-9.]+) ms.*/\1/')
  second_ms=$(printf "%s" "$line" | sed -E 's/.*2nd token latency: ([0-9.]+) ms.*/\1/')
  second_tps=$(printf "%s" "$line" | sed -E 's/.*2nd tokens throughput: ([0-9.]+) tokens\/s.*/\1/')

  if [[ "$first_ms" == "$line" ]]; then
    first_ms=""
  fi
  if [[ "$second_ms" == "$line" ]]; then
    second_ms=""
  fi
  if [[ "$second_tps" == "$line" ]]; then
    second_tps=""
  fi

  printf "%s,%s,%s,%s,%s\n" "$ov_config" "$first_ms" "$second_ms" "$second_tps" "$model_name" >> "$tmp_csv"
done

{ head -n 1 "$tmp_csv"; tail -n +2 "$tmp_csv" | sort -t, -k1,1; } > "$OUT_CSV"
rm -f "$tmp_csv"

echo "Wrote $OUT_CSV"