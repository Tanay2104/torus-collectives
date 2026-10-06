#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
fi
export PYTHONPATH="$ROOT:$ROOT/Simulation/astra-sim:$ROOT/Simulation/astra-sim/extern/graph_frontend${PYTHONPATH:+:$PYTHONPATH}"

DIMS="4 4"
BANDWIDTHS="50 50"
LATENCIES="100 100"
COLL_SIZE=1
NPU_COUNT=""
CSV_FILE=""
LOWERER="forestcoll"
SIM_TIMEOUT_SECONDS=${SIM_TIMEOUT_SECONDS:-120}

while [[ $# -gt 0 ]]; do
  case $1 in
    --dims) DIMS=$2; shift 2 ;;
    --bandwidths) BANDWIDTHS=$2; shift 2 ;;
    --latencies) LATENCIES=$2; shift 2 ;;
    --coll-size) COLL_SIZE=$2; shift 2 ;;
    --npu-count) NPU_COUNT=$2; shift 2 ;;
    --csv) CSV_FILE=$2; shift 2 ;;
    --lowerer) LOWERER=$2; shift 2 ;;
    -h|--help)
      echo 'Usage: bash run.sh [--dims "4 4"] [--bandwidths "50 50"] [--latencies "100 100"] [--coll-size MB] [--npu-count N] [--lowerer forestcoll|msccldsl] [--csv FILE]'
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! grep -q 'class MSCCLReceiveCopySendStep' \
    "$ROOT/Simulation/collectiveapi/chakra_converter/mscclang2chakra_converter.py"; then
  echo "The pinned Collective API converter lacks rcs support. Run ./setup-deps.sh first." >&2
  exit 1
fi
if [[ "$LOWERER" != forestcoll && "$LOWERER" != msccldsl ]]; then
  echo "Unsupported lowerer: $LOWERER (choose forestcoll or msccldsl)" >&2
  exit 2
fi

if [[ -z "$NPU_COUNT" ]]; then
  NPU_COUNT=1
  for dim in $DIMS; do NPU_COUNT=$((NPU_COUNT * dim)); done
fi

mkdir -p "$ROOT/runs"
OUTPUT_DIR="$(mktemp -d "$ROOT/runs/run-XXXXXX")"
mkdir -p "$OUTPUT_DIR/scratch" "$OUTPUT_DIR/workload" \
  "$OUTPUT_DIR/spst-torus-et" "$OUTPUT_DIR/forestcoll-torus-et"
PREFIX_XML="$OUTPUT_DIR/torus"
cat > "$OUTPUT_DIR/settings.txt" <<SETTINGS
dims=$DIMS
bandwidths_gbps=$BANDWIDTHS
latencies_ns=$LATENCIES
collective_size_mb=$COLL_SIZE
npus=$NPU_COUNT
sim_timeout_seconds=$SIM_TIMEOUT_SECONDS
lowerer=$LOWERER
SETTINGS

printf 'Config: dims=[%s] bw=[%s] lat=[%s] npus=%s coll-size=%s\n' \
  "$DIMS" "$BANDWIDTHS" "$LATENCIES" "$NPU_COUNT" "$COLL_SIZE"
printf 'Run files: %s\n' "$OUTPUT_DIR"

python "$ROOT/Simulation/solve_torus.py" \
  --dims $DIMS --bandwidths $BANDWIDTHS --prefix "$PREFIX_XML" \
  --lowerer "$LOWERER"

PROTO_DIR="$ROOT/Simulation/astra-sim/extern/graph_frontend/chakra/schema/protobuf"
PROTO_OUT="$OUTPUT_DIR/python/chakra/schema/protobuf"
PROTO_ASTRA_OUT="$OUTPUT_DIR/python/extern/graph_frontend/chakra/schema/protobuf"
mkdir -p "$PROTO_OUT"
protoc --proto_path="$PROTO_DIR" --python_out="$PROTO_OUT" "$PROTO_DIR/et_def.proto"
mkdir -p "$PROTO_ASTRA_OUT"
cp "$PROTO_OUT/et_def_pb2.py" "$PROTO_ASTRA_OUT/"
export PYTHONPATH="$OUTPUT_DIR/python:$PYTHONPATH"

python "$ROOT/Simulation/collectiveapi/chakra_converter/et_converter.py" \
  --input_filename "$PREFIX_XML-spst.xml" \
  --output_filename "$OUTPUT_DIR/spst-torus-et/spst-torus" \
  --log_filename "$OUTPUT_DIR/spst-converter.log"
python "$ROOT/Simulation/collectiveapi/chakra_converter/et_converter.py" \
  --input_filename "$PREFIX_XML-forestcoll.xml" \
  --output_filename "$OUTPUT_DIR/forestcoll-torus-et/forestcoll-torus" \
  --log_filename "$OUTPUT_DIR/forestcoll-converter.log"

python "$ROOT/Simulation/gen_topo.py" \
  --dims $DIMS --bandwidths $BANDWIDTHS --latencies $LATENCIES \
  --output "$OUTPUT_DIR/topo.txt"
bash "$ROOT/Simulation/gen_logical.sh" "$OUTPUT_DIR/logical.json" $DIMS
printf '0\n' > "$OUTPUT_DIR/scratch/flow.txt"
{
  echo "$NPU_COUNT"
  for ((rank = 0; rank < NPU_COUNT; rank++)); do printf '%s ' "$rank"; done
  echo
} > "$OUTPUT_DIR/scratch/trace.txt"

# Upstream ASTRA-Sim writes under all_gather/<npus>npus_<MB>MB from its cwd.
(
  cd "$OUTPUT_DIR/workload"
  python "$ROOT/Simulation/astra-sim/examples/workload/microbenchmarks/generator_scripts/all_gather.py" \
    --npus-count "$NPU_COUNT" --coll-size "$COLL_SIZE"
)
WORKLOAD="$OUTPUT_DIR/workload/all_gather/${NPU_COUNT}npus_${COLL_SIZE}MB/all_gather"

sed "s|@OUTPUT_DIR@|$OUTPUT_DIR|g" "$ROOT/Simulation/config.txt.in" > "$OUTPUT_DIR/config.txt"
sed "s|@OUTPUT_DIR@|$OUTPUT_DIR|g" "$ROOT/Simulation/system-spst.json.in" > "$OUTPUT_DIR/system-spst.json"
sed "s|@OUTPUT_DIR@|$OUTPUT_DIR|g" "$ROOT/Simulation/system-forestcoll.json.in" > "$OUTPUT_DIR/system-forestcoll.json"

SIMULATOR="$ROOT/Simulation/astra-sim/extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default"
run_simulator() {
  local system_config=$1 log_file=$2
  (
    cd "$OUTPUT_DIR"
    timeout "${SIM_TIMEOUT_SECONDS}s" "$SIMULATOR" \
      --workload-configuration="$WORKLOAD" \
      --system-configuration="$OUTPUT_DIR/$system_config" \
      --network-configuration="$OUTPUT_DIR/config.txt" \
      --remote-memory-configuration="$ROOT/Simulation/no_mem_expansion.json" \
      --comm-group-configuration=empty \
      --logical-topology-configuration="$OUTPUT_DIR/logical.json" \
      > "$OUTPUT_DIR/$log_file"
  )
}
run_simulator system-spst.json spst.log
run_simulator system-forestcoll.json forestcoll.log

get_max_wall_time() {
  awk '/Wall time:/ {value=$NF} END {if (value != "") print value}' "$1"
}
SPST_TIME=$(get_max_wall_time "$OUTPUT_DIR/spst.log")
FORESTCOLL_TIME=$(get_max_wall_time "$OUTPUT_DIR/forestcoll.log")
printf 'SPST: %s cycles\nForestColl: %s cycles\n' "$SPST_TIME" "$FORESTCOLL_TIME"

if [[ -n "$CSV_FILE" ]]; then
  if [[ ! -f "$CSV_FILE" ]]; then
    echo 'algorithm,dims,bandwidths,latencies,collsize,time' > "$CSV_FILE"
  fi
  printf 'spst,"%s","%s","%s",%s,%s\n' "$DIMS" "$BANDWIDTHS" "$LATENCIES" "$COLL_SIZE" "$SPST_TIME" >> "$CSV_FILE"
  printf 'forestcoll,"%s","%s","%s",%s,%s\n' "$DIMS" "$BANDWIDTHS" "$LATENCIES" "$COLL_SIZE" "$FORESTCOLL_TIME" >> "$CSV_FILE"
fi
