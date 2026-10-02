#!/bin/sh
set -eu

repository=${OPENSPLINE_REPOSITORY:-openspline/openspline}
version=${OPENSPLINE_VERSION:-main}
install_dir=${OPENSPLINE_INSTALL_DIR:-"${HOME}/.local/share/openspline"}
archive_url=${OPENSPLINE_ARCHIVE_URL:-"https://github.com/${repository}/archive/refs/heads/${version}.tar.gz"}
if [ "$version" != main ] && [ -z "${OPENSPLINE_ARCHIVE_URL:-}" ]; then
  archive_url="https://github.com/${repository}/archive/refs/tags/${version}.tar.gz"
fi

if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
  echo "openspline: the native GPU installer requires Linux x86_64" >&2
  exit 1
fi
for command in curl tar nvidia-smi; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "openspline: $command is required (install the NVIDIA driver if nvidia-smi is missing)" >&2
    exit 1
  fi
done
nvidia-smi -L >/dev/null

mkdir -p "$install_dir"
install_dir=$(cd "$install_dir" && pwd -P)
export OPENSPLINE_INSTALL_DIR="$install_dir"
export TMPDIR="$install_dir/runtime/tmp"
export UV_CACHE_DIR="$install_dir/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$install_dir/.tools/python"
export UV_NO_MODIFY_PATH=1
export PIP_CACHE_DIR="$install_dir/.cache/pip"
export XDG_CACHE_HOME="$install_dir/.cache"
mkdir -p "$TMPDIR" "$install_dir/.tools"
temporary_dir=$(mktemp -d "$TMPDIR/install.XXXXXX")
cleanup() { rm -rf "$temporary_dir"; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' HUP TERM

if [ -n "${OPENSPLINE_SOURCE_DIR:-}" ]; then
  source_dir=$(cd "$OPENSPLINE_SOURCE_DIR" && pwd -P)
else
  echo "Downloading openspline ${version}..."
  curl -fSL --retry 3 "$archive_url" -o "$temporary_dir/source.tar.gz"
  mkdir "$temporary_dir/source"
  tar -xzf "$temporary_dir/source.tar.gz" --strip-components=1 -C "$temporary_dir/source"
  source_dir="$temporary_dir/source"
fi
for file in packages/server/pyproject.toml requirements/server.txt requirements/cu126.txt requirements/cu128.txt run.sh; do
  if [ ! -f "$source_dir/$file" ]; then
    echo "openspline: source is missing $file" >&2
    exit 1
  fi
done

if [ "$source_dir" != "$install_dir" ]; then
  for file in .env workers.yaml; do
    if [ -f "$install_dir/$file" ]; then cp "$install_dir/$file" "$temporary_dir/saved-$file"; fi
  done
  cp -R "$source_dir/." "$install_dir/"
  for file in .env workers.yaml; do
    if [ -f "$temporary_dir/saved-$file" ]; then cp "$temporary_dir/saved-$file" "$install_dir/$file"; fi
  done
fi
if [ ! -f "$install_dir/.env" ]; then
  (umask 077; cp "$install_dir/.env.example" "$install_dir/.env")
fi

# Bootstrap a pinned package manager and Python locally, without changing the host.
uv="$install_dir/.tools/uv"
if [ ! -x "$uv" ]; then
  curl -fSL --retry 3 "https://github.com/astral-sh/uv/releases/download/0.8.17/uv-x86_64-unknown-linux-gnu.tar.gz" -o "$temporary_dir/uv.tar.gz"
  tar -xzf "$temporary_dir/uv.tar.gz" -C "$temporary_dir"
  cp "$temporary_dir/uv-x86_64-unknown-linux-gnu/uv" "$uv"
  chmod 755 "$uv"
fi
python="$install_dir/.venv/bin/python"
if [ ! -x "$python" ]; then
  "$uv" venv --no-project --python 3.11 "$install_dir/.venv"
fi
"$python" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Python 3.10–3.12 required"'
echo "Installing the GPU service and SDK..."
# Install common packages first so GPU selection can read workers.yaml and .env.
# --no-deps prevents torch from being pulled from PyPI before its CUDA build is selected.
"$uv" pip install --python "$python" --no-deps --require-hashes -r "$install_dir/requirements/server.txt"
cd "$install_dir"
export PYTHONPATH="$install_dir/packages/server/src${PYTHONPATH:+:$PYTHONPATH}"
cuda=$("$python" -m openspline_server.hardware)
case "$cuda" in cu126|cu128) ;; *) echo "openspline: could not select a CUDA runtime" >&2; exit 1 ;; esac
echo "Installing PyTorch for $cuda..."
"$uv" pip install --python "$python" --torch-backend "$cuda" --require-hashes -r "$install_dir/requirements/server.txt" -r "$install_dir/requirements/$cuda.txt"
"$uv" pip install --python "$python" --no-deps "$install_dir/packages/server" "$install_dir/packages/python"
"$python" -m openspline_server.hardware --check

sh "$install_dir/run.sh" --prepare --save-selection
echo "Installed in $install_dir. Restart with: sh \"$install_dir/run.sh\""
cleanup
trap - EXIT HUP INT TERM
if [ "${OPENSPLINE_NO_START:-0}" = 1 ]; then exit 0; fi
exec sh "$install_dir/run.sh"
