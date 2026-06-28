#!/usr/bin/env bash
set -euo pipefail

asset_dir="${1:?usage: download-assets.sh ASSET_DIR}"
runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
suite="${YSCV_PR_BENCH_SUITE:-$runner_dir/suite.json}"
mkdir -p "$asset_dir"

download() {
  local url="$1"
  local path="$2"
  if [[ -s "$path" ]]; then
    echo "asset exists: $path"
    return
  fi
  local tmp="${path}.tmp"
  rm -f "$tmp"
  echo "downloading: $url"
  curl --fail --location --retry 3 --retry-delay 2 --output "$tmp" "$url"
  mv "$tmp" "$path"
}

python3 - "$suite" <<'PY' | while IFS=$'\t' read -r rel_path url; do
import json
import os
import sys

suite = json.load(open(sys.argv[1]))
for asset in suite.get("assets", []):
    url = os.environ.get(asset.get("env", ""), asset["url"])
    print(f"{asset['path']}\t{url}")
PY
  download "$url" "$asset_dir/$rel_path"
done
