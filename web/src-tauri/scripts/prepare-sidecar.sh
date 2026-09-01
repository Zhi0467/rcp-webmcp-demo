#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../../.." && pwd)
target_triple=${RCP_TAURI_TARGET:-aarch64-apple-darwin}

if [ "$target_triple" != "aarch64-apple-darwin" ]; then
  echo "RCP desktop packaging currently supports only aarch64-apple-darwin." >&2
  exit 1
fi

sh "$repo_root/packaging/build-backend.sh"

source_binary="$repo_root/packaging/dist/rcp-backend"
target_directory="$repo_root/web/src-tauri/binaries"
target_binary="$target_directory/rcp-backend-$target_triple"

if [ ! -x "$source_binary" ]; then
  echo "Packaged backend is missing or not executable: $source_binary" >&2
  exit 1
fi

mkdir -p "$target_directory"
cp "$source_binary" "$target_binary"
chmod 755 "$target_binary"
echo "$target_binary"
