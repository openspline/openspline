#!/bin/sh
set -eu
install_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$install_dir"
if [ ! -x "$install_dir/.venv/bin/python" ]; then
  echo "openspline: run install.sh first" >&2
  exit 1
fi
exec "$install_dir/.venv/bin/python" -m openspline_server.native "$@"
