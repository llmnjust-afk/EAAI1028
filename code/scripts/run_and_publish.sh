#!/usr/bin/env bash
# Run an experiment command and, on success, commit and push the artefacts.
#
# Usage:
#   scripts/run_and_publish.sh --publish --pat-file /path/to/PAT -- \
#       python3 scripts/run_e2.py ...
#
# The token is only read by publish_results.py and is never echoed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PUBLISH=0
INCLUDE_PAPER=0
PAT_FILE="${GITHUB_PAT_FILE:-}"
MESSAGE="Update experiment artefacts"

while (($#)); do
  case "$1" in
    --publish) PUBLISH=1; shift ;;
    --include-paper) INCLUDE_PAPER=1; shift ;;
    --pat-file) PAT_FILE="$2"; shift 2 ;;
    --message) MESSAGE="$2"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

if (($# == 0)); then
  echo "usage: $0 [--publish] [--pat-file PATH] [--message TEXT] -- <command...>" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/code/experiment_results"
LOG="${REPO_ROOT}/code/experiment_results/pipeline.log"
{
  printf '\n[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
} >> "${LOG}"

set +e
"$@" 2>&1 | tee -a "${LOG}"
status=${PIPESTATUS[0]}
set -e
if ((status != 0)); then
  echo "experiment command failed with status ${status}; not publishing" >&2
  exit "${status}"
fi

if ((PUBLISH)); then
  args=(--repo "${REPO_ROOT}" --message "${MESSAGE}")
  if [[ -n "${PAT_FILE}" ]]; then
    args+=(--pat-file "${PAT_FILE}")
  fi
  if ((INCLUDE_PAPER)); then
    args+=(--include-paper)
  fi
  python3 "${REPO_ROOT}/code/scripts/publish_results.py" "${args[@]}"
fi
