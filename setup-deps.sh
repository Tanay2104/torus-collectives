#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
trim() {
  local value=$1
  value=${value#"${value%%[![:space:]]*}"}
  value=${value%"${value##*[![:space:]]}"}
  printf '%s' "$value"
}

checkout_pin() {
  local repo=$1 revision=$2 actual base merge
  if [[ "$revision" == *+* ]]; then
    base=${revision%%+*}
    merge=${revision#*+}
    actual=$(git -C "$repo" rev-parse HEAD)
    if [[ "$(git -C "$repo" rev-parse "$actual^1" 2>/dev/null || true)" == "$base" && \
          "$(git -C "$repo" rev-parse "$actual^2" 2>/dev/null || true)" == "$merge" ]]; then
      return 0
    fi
    if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
      echo "Cannot switch $repo to its pinned merge: checkout has local changes." >&2
      return 1
    fi
    git -C "$repo" fetch origin refs/heads/comm_size
    actual=$(git -C "$repo" rev-parse FETCH_HEAD)
    if [[ "$actual" != "$merge" ]]; then
      echo "ASTRA-Sim comm_size ref is $actual; expected pinned commit $merge." >&2
      return 1
    fi
    git -C "$repo" fetch origin "$base"
    git -C "$repo" checkout --detach "$base"
    git -C "$repo" -c user.name='CollComm setup' \
      -c user.email='collcomm@example.invalid' \
      merge --no-edit --no-ff "$merge"
  else
    actual=$(git -C "$repo" rev-parse HEAD)
    if [[ "$actual" == "$revision" ]]; then
      return 0
    fi
    if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
      echo "Cannot switch $repo to $revision: checkout has local changes." >&2
      return 1
    fi
    git -C "$repo" checkout --detach "$revision"
  fi
}

update_submodules() {
  local repo=$1 line missing=false
  while IFS= read -r line; do
    case "${line:0:1}" in
      +|U)
        echo "Submodule checkout differs from its pinned revision in $repo:" >&2
        echo "  $line" >&2
        return 1
        ;;
      -) missing=true ;;
    esac
  done < <(git -C "$repo" submodule status --recursive)
  if [[ "$missing" == true ]]; then
    git -C "$repo" submodule update --init --recursive
  fi
}

while IFS='|' read -r name relpath url revision role; do
  name=$(trim "$name")
  relpath=$(trim "$relpath")
  url=$(trim "$url")
  revision=$(trim "$revision")
  [[ -z "$name" || "$name" == name || "$name" == \#* ]] && continue
  checkout="$ROOT/$relpath"

  if [[ ! -e "$checkout" ]]; then
    echo "Cloning $name into $relpath"
    git clone "$url" "$checkout"
  fi
  if git -C "$checkout" rev-parse --git-dir >/dev/null 2>&1; then
    if checkout_pin "$checkout" "$revision"; then
      update_submodules "$checkout"
      echo "$name ready at its pinned revision"
    else
      exit 1
    fi
  else
    echo "$checkout exists but is not a Git checkout; leaving it unchanged." >&2
    exit 1
  fi
done < "$ROOT/dependencies.lock"

# Apply the project-owned compatibility patch needed by the pinned converter.
# Keep this in setup: launching an experiment should not mutate dependencies.
"$ROOT/apply-patches.sh" rcs
