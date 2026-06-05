#!/usr/bin/env bash
# This script pre-pulls and tags GPU-related Docker images from specified registries.

set -euo pipefail

# Minimal script: pre-pull three images (builder/runtime/buildkit) and tag them to
# canonical names so downstream scripts can rely on the expected tags.

# You can override these by setting the env vars before running this script.
BUILDER_IMAGE=${BUILDER_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04}
RUNTIME_IMAGE=${RUNTIME_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.6.3-runtime-ubuntu22.04}
BUILDKIT_IMAGE=${BUILDKIT_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/moby/buildkit:buildx-stable-1}

retry_pull() {
  local img="$1"
  for i in 1 2 3; do
    echo "pull attempt $i for ${img}..."
    if docker pull "${img}"; then
      echo "pulled ${img}"
      return 0
    fi
    sleep $((i * 2))
  done
  echo "Failed to pull ${img} after retries" >&2
  return 1
}

echo "Pre-pulling images..."
echo "- builder: ${BUILDER_IMAGE}"
echo "- runtime: ${RUNTIME_IMAGE}"
echo "- buildkit: ${BUILDKIT_IMAGE}"

retry_pull "${BUILDER_IMAGE}" || true
retry_pull "${RUNTIME_IMAGE}" || true
retry_pull "${BUILDKIT_IMAGE}" || true

CANONICAL_BUILDKIT_TAG="moby/buildkit:buildx-stable-1"
if docker image inspect "${BUILDKIT_IMAGE}" >/dev/null 2>&1; then
  echo "Tagging ${BUILDKIT_IMAGE} -> ${CANONICAL_BUILDKIT_TAG} (local only)"
  docker tag "${BUILDKIT_IMAGE}" "${CANONICAL_BUILDKIT_TAG}" || true
fi

# Also tag the mirrored CUDA images to the original docker.io names expected by
# Dockerfiles and other scripts. This lets downstream tooling refer to
# docker.io/nvidia/cuda:12.6.3-... even when images were pulled from a mirror.
ORIG_BUILDER_TAG="docker.io/nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04"
ORIG_RUNTIME_TAG="docker.io/nvidia/cuda:12.6.3-runtime-ubuntu22.04"
if docker image inspect "${BUILDER_IMAGE}" >/dev/null 2>&1; then
  echo "Tagging ${BUILDER_IMAGE} -> ${ORIG_BUILDER_TAG}"
  docker tag "${BUILDER_IMAGE}" "${ORIG_BUILDER_TAG}" || true
fi
if docker image inspect "${RUNTIME_IMAGE}" >/dev/null 2>&1; then
  echo "Tagging ${RUNTIME_IMAGE} -> ${ORIG_RUNTIME_TAG}"
  docker tag "${RUNTIME_IMAGE}" "${ORIG_RUNTIME_TAG}" || true
fi

echo "Done pulling/tagging images."
echo "You can now run downstream build steps that expect these images to exist locally."
