#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "The first RCP desktop release builds only on Apple Silicon macOS." >&2
  exit 1
fi

cd "$project_root"
npm --prefix web run build
uv run pyinstaller \
  --clean \
  --noconfirm \
  --distpath "$script_dir/dist" \
  --workpath "$script_dir/build" \
  "$script_dir/rcp_backend.spec"

backend="$script_dir/dist/rcp-backend"
if [ ! -x "$backend" ]; then
  echo "PyInstaller did not produce $backend" >&2
  exit 1
fi

echo "$backend"
