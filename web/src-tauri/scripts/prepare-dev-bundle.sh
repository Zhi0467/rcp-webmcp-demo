#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../../.." && pwd)
uv_executable=${RCP_DEV_UV:-}

if [ -z "$uv_executable" ]; then
  uv_executable=$(command -v uv || true)
fi
if [ -z "$uv_executable" ] || [ ! -x "$uv_executable" ]; then
  echo "RCP.app requires uv; set RCP_DEV_UV to its absolute executable path." >&2
  exit 1
fi

uv_directory=$(CDPATH= cd -- "$(dirname -- "$uv_executable")" && pwd)
uv_executable="$uv_directory/$(basename -- "$uv_executable")"
generated_directory="$repo_root/web/src-tauri/generated"
generated_plist="$generated_directory/Info.dev.plist"

mkdir -p "$generated_directory"
cp "$repo_root/web/src-tauri/Info.dev.template.plist" "$generated_plist"
plutil -replace RCPDevCheckout -string "$repo_root" "$generated_plist"
plutil -replace RCPDevUvExecutable -string "$uv_executable" "$generated_plist"
echo "$generated_plist"
