#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-config/paperqa.yaml}"
IMAGE_NAME="$(awk -F': ' '$1=="service_container_name"{gsub(/"/,"",$2); print $2}' "${CONFIG_FILE}")"
PORT="$(awk -F': ' '$1=="service_base_url"{gsub(/"/,"",$2); print $2}' "${CONFIG_FILE}" | awk -F: '{print $3}' | tr -d /)"

if docker ps -a --format '{{.Names}}' | grep -qx "${IMAGE_NAME}"; then
  # Reuse the existing container if it exists (start it if stopped).
  docker start "${IMAGE_NAME}"
else
  # Start a new container from the built image with the standard mounts.
  docker run --rm -p "${PORT}:9070" --name "${IMAGE_NAME}" \
    --add-host host.docker.internal:host-gateway \
    -v "$(pwd)/storage":/app/storage \
    -v "$(pwd)/config":/app/config \
    -v "$(pwd)/prompts":/app/prompts \
    "${IMAGE_NAME}"
fi
