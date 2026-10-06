#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
apply_once() {
  local patchfile=$1
  local target=$2
  if patch --dry-run -s -p1 -d "$target" < "$ROOT/$patchfile"; then
    patch -p1 -d "$target" < "$ROOT/$patchfile"
  elif patch --dry-run -R -s -p1 -d "$target" < "$ROOT/$patchfile"; then
    echo "Already applied: $patchfile"
  else
    echo "Patch does not match checkout: $patchfile" >&2
    return 1
  fi
}

case "${1:-}" in
  rcs)
    apply_once patches/collectiveapi-rcs.patch "$ROOT/Simulation/collectiveapi"
    ;;
  python314)
    apply_once patches/ns3-python314.patch "$ROOT/Simulation/astra-sim/extern/network_backend/ns-3"
    ;;
  *)
    echo "Usage: $0 {rcs|python314}" >&2
    exit 2
    ;;
esac
