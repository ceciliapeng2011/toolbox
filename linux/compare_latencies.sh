#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <csv_a> <csv_b> [out_csv]" >&2
  exit 1
fi

CSV_A="$1"
CSV_B="$2"
NAME_A=$(basename "$CSV_A" .csv)
NAME_B=$(basename "$CSV_B" .csv)
OUT_CSV="${3:-$(dirname "$CSV_B")/compare_$(basename "$CSV_A" .csv)_vs_$(basename "$CSV_B" .csv).csv}"

awk -F',' -v OFS=',' '
  function norm_model(s,    i) {
    i = index(s, ".__home__ceciliapeng__3rdparty__x-attention__eval__efficiency__");
    if (i > 0) {
      return substr(s, 1, i-1);
    }
    return s;
  }
  NR==1 { next }
  FNR==1 { next }
  FILENAME==ARGV[1] {
    ov_config=$1; first_ms=$2; model_name=norm_model($5);
    if (model_name != "" && ov_config != "") {
      key=model_name SUBSEP ov_config;
      a_first[key]=first_ms;
      a_ov[key]=ov_config;
      a_model[key]=model_name;
    }
    next;
  }
  {
    ov_config=$1; first_ms=$2; model_name=norm_model($5);
    if (model_name == "" || ov_config == "") next;
    key=model_name SUBSEP ov_config;
    if (key in a_first) {
      a=a_first[key]; b=first_ms;
      boost="";
      ratio="";
      if (a != "" && b != "" && a+0 != 0) {
        boost=b-a;
        boost=sprintf("%.2f", boost) " ms";
        ratio=((b)/a)*100;
        ratio=sprintf("%.2f", ratio) "%";
      }
      if (a != "") a=a " ms";
      if (b != "") b=b " ms";
      printf "%s,%s,%s,%s,%s,%s\n", model_name, ov_config, a, b, boost, ratio;
    }
  }
' "$CSV_A" "$CSV_B" | sort -t, -k2,2 -k1,1 | {
  echo "model_name,ov_config,Base <${NAME_A}>,Opt <${NAME_B}>,boost_pct (Opt-Base),boost_ratio_pct (Opt/Base, lower is better)";
  cat;
} > "$OUT_CSV"

echo "Wrote $OUT_CSV"