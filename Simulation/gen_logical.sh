#!/bin/bash
# Usage: ./gen_logical.sh output.json 4 4

if [ $# -lt 2 ]; then
  echo "Usage: $0 output.json dim1 [dim2 ...]"
  exit 1
fi

OUTPUT_FILE=$1
shift

# Build JSON array string
arr=""
for d in "$@"; do
  arr+="\"$d\","
done
arr="${arr%,}" # remove trailing comma

printf '{\n  "logical-dims": [%s]\n}\n' "$arr" >"$OUTPUT_FILE"
