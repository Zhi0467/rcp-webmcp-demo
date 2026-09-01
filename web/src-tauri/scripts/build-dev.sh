#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../../.." && pwd)

sh "$script_dir/prepare-dev-bundle.sh"
cd "$repo_root/web"
npx tauri build --debug --config src-tauri/tauri.dev-bundle.conf.json
