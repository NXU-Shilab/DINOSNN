#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <raw_1kgp_data> <CaQTL_data> <hg38_file> <PhastCons_file>"
  exit 1
fi

IN_FILE="$1"
CAQTL="$2"
FA_ARG="$3"
PHA_ARG="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$PROJECT_DIR/PartII_data"
mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/CaQTL_data"

OUT_FILE="$OUT_DIR/filter_coding_1KGP.csv"

Rscript "$SCRIPT_DIR/filter_1kgp_coding.R" "$IN_FILE" "$OUT_FILE"

python "$SCRIPT_DIR/handle_1kgp.py" --fa "$FA_ARG" --pha "$PHA_ARG"

python "$SCRIPT_DIR/handle_variants.py" --caqtl "$CAQTL" --fa "$FA_ARG" --pha "$PHA_ARG"
