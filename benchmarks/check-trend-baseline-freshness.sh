#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <trend_baseline.tsv> [max_age_days]" >&2
  exit 2
fi

baseline_path="$1"
max_age_days="${2:-14}"

if [[ ! -f "$baseline_path" ]]; then
  echo "trend baseline not found: $baseline_path" >&2
  exit 2
fi

if ! [[ "$max_age_days" =~ ^[0-9]+$ ]]; then
  echo "max_age_days must be a non-negative integer, got: $max_age_days" >&2
  exit 2
fi

latest_timestamp="$(
  awk -F '\t' '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == "timestamp_utc") {
          ts_col = i
        }
      }
      if (ts_col == 0) {
        print "missing timestamp_utc column" > "/dev/stderr"
        exit 2
      }
      next
    }
    ts_col > 0 && $ts_col != "" {
      if ($ts_col > latest) {
        latest = $ts_col
      }
    }
    END {
      if (latest == "") {
        exit 3
      }
      print latest
    }
  ' "$baseline_path"
)" || {
  echo "failed to parse latest timestamp from trend baseline: $baseline_path" >&2
  exit 2
}

to_epoch_seconds() {
  local timestamp="$1"
  if date -u -d "$timestamp" +%s >/dev/null 2>&1; then
    date -u -d "$timestamp" +%s
    return 0
  fi
  if date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$timestamp" +%s >/dev/null 2>&1; then
    date -u -j -f "%Y-%m-%dT%H:%M:%SZ" "$timestamp" +%s
    return 0
  fi
  return 1
}

latest_epoch="$(to_epoch_seconds "$latest_timestamp")" || {
  echo "unsupported timestamp format in trend baseline: $latest_timestamp" >&2
  exit 2
}
now_epoch="$(date -u +%s)"
age_seconds=$((now_epoch - latest_epoch))
if [[ "$age_seconds" -lt 0 ]]; then
  age_seconds=0
fi
age_days=$((age_seconds / 86400))

echo "trend baseline freshness: file=$baseline_path latest_timestamp=$latest_timestamp age_days=$age_days max_age_days=$max_age_days"

if [[ "$age_days" -gt "$max_age_days" ]]; then
  message="trend baseline is stale: file=$baseline_path latest_timestamp=$latest_timestamp age_days=$age_days max_age_days=$max_age_days"
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "::warning title=Trend Baseline Stale::$message"
  else
    echo "WARNING: $message" >&2
  fi

  enforce="${YSCV_TREND_BASELINE_ENFORCE:-0}"
  if [[ "$enforce" == "1" ]]; then
    echo "failing due to trend baseline freshness enforcement" >&2
    exit 1
  fi
fi
