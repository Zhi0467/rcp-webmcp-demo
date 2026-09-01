#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../../.." && pwd)

sh "$script_dir/prepare-sidecar.sh"
cd "$repo_root/web"
if [ "${1:-}" = "--updater" ]; then
  npx tauri build \
    --config src-tauri/tauri.release.conf.json \
    --config src-tauri/tauri.updater.conf.json
else
  npx tauri build --config src-tauri/tauri.release.conf.json
fi
