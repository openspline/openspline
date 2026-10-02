#!/bin/sh
set -eu

repository=${OPENSPLINE_REPOSITORY:-openspline/openspline}
version=${OPENSPLINE_VERSION:-main}
install_dir=${OPENSPLINE_INSTALL_DIR:-"${HOME}/.local/share/openspline"}
archive_url=${OPENSPLINE_ARCHIVE_URL:-"https://github.com/${repository}/archive/refs/heads/${version}.tar.gz"}

if [ "$version" != "main" ] && [ -z "${OPENSPLINE_ARCHIVE_URL:-}" ]; then
  archive_url="https://github.com/${repository}/archive/refs/tags/${version}.tar.gz"
fi

for command in curl tar docker; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "openspline: $command is required" >&2
    exit 1
  fi
done

if ! docker compose version >/dev/null 2>&1; then
  echo "openspline: Docker Compose v2 is required" >&2
  exit 1
fi

# A runtime entry in Docker's configuration can outlive the toolkit package.
# This installer targets a local Linux Docker host (also required by host networking).
if ! command -v nvidia-container-runtime >/dev/null 2>&1 \
    || ! nvidia-container-runtime --version >/dev/null 2>&1; then
  echo "openspline: nvidia-container-runtime is missing or cannot run" >&2
  echo "Install NVIDIA Container Toolkit on the Docker host before continuing:" >&2
  echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
  exit 1
fi

if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  echo "openspline: Docker cannot find the NVIDIA container runtime" >&2
  echo "Install NVIDIA Container Toolkit and configure Docker before continuing" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "openspline: an NVIDIA GPU and NVIDIA Container Toolkit are required" >&2
  exit 1
fi

temporary_dir=$(mktemp -d)
cleanup() {
  rm -rf "$temporary_dir"
}
trap cleanup EXIT HUP INT TERM

echo "Downloading openspline ${version}..."
curl -fsSL "$archive_url" -o "$temporary_dir/openspline.tar.gz"
mkdir -p "$temporary_dir/source" "$install_dir"
tar -xzf "$temporary_dir/openspline.tar.gz" --strip-components=1 -C "$temporary_dir/source"

if [ ! -f "$temporary_dir/source/compose.yaml" ]; then
  echo "openspline: downloaded archive is missing compose.yaml" >&2
  exit 1
fi

saved_env=
saved_workers=
if [ -f "$install_dir/.env" ]; then
  saved_env="$temporary_dir/existing.env"
  cp "$install_dir/.env" "$saved_env"
fi
if [ -f "$install_dir/workers.yaml" ]; then
  saved_workers="$temporary_dir/existing-workers.yaml"
  cp "$install_dir/workers.yaml" "$saved_workers"
fi

cp -R "$temporary_dir/source/." "$install_dir/"

if [ -n "$saved_env" ]; then
  cp "$saved_env" "$install_dir/.env"
else
  umask 077
  cp "$install_dir/.env.example" "$install_dir/.env"
fi

if [ -n "$saved_workers" ]; then
  cp "$saved_workers" "$install_dir/workers.yaml"
fi

if [ "${OPENSPLINE_NO_START:-0}" = "1" ]; then
  echo "openspline installed in $install_dir"
  exit 0
fi

echo "Building and starting openspline (the first model download may take a while)..."
docker compose --project-directory "$install_dir" -f "$install_dir/compose.yaml" up --build -d

echo "openspline is starting at http://localhost:7860"
echo "Configuration: $install_dir/.env"
