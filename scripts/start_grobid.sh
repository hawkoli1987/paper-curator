#!/usr/bin/env bash
set -euo pipefail

IMAGE="lfoppiano/grobid:0.8.0"
# Keep host port configurable to avoid collisions with LLM servers.
PORT="${PORT:-9070}"

docker pull "${IMAGE}"
# GROBID listens on 8070 inside the container; map host PORT -> container 8070.
docker run --rm -p "${PORT}:8070" --name grobid "${IMAGE}"
