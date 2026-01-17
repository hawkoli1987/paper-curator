#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://localhost:9002"
# Use a stable, well-known arXiv ID for repeatable parsing runs.
ARXIV_ID="1706.03762"
# Keep outputs under repo storage for reproducible workflows.
OUTPUT_DIR="storage/downloads"
OUTPUT_JSON="storage/outputs/extract.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --arxiv-id)
      ARXIV_ID="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-json)
      OUTPUT_JSON="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "${OUTPUT_JSON}")"
mkdir -p "${OUTPUT_DIR}"

# Fail fast on any HTTP error so downstream steps don't consume bad data.
echo "Resolving arXiv ID: ${ARXIV_ID}"
curl -f -sS -X POST "${BASE_URL}/arxiv/resolve" \
  -H "Content-Type: application/json" \
  -d "{\"arxiv_id\":\"${ARXIV_ID}\"}" \
  > /tmp/arxiv_resolve.json

echo "Downloading PDF/LaTeX"
curl -f -sS -X POST "${BASE_URL}/arxiv/download" \
  -H "Content-Type: application/json" \
  -d "{\"arxiv_id\":\"${ARXIV_ID}\",\"output_dir\":\"${OUTPUT_DIR}\"}" \
  > /tmp/arxiv_download.json

PDF_PATH="$(python - <<'PY'
import json
with open('/tmp/arxiv_download.json', 'r', encoding='utf-8') as handle:
    data = json.load(handle)
# Extract the downloaded PDF path from the API response.
pdf_path = data.get('pdf_path')
if not pdf_path:
    raise SystemExit("pdf_path missing in download response")
print(pdf_path)
PY
)"

echo "Extracting with GROBID: ${PDF_PATH}"
curl -f -sS -X POST "${BASE_URL}/pdf/extract" \
  -H "Content-Type: application/json" \
  -d "{\"pdf_path\":\"${PDF_PATH}\"}" \
  > "${OUTPUT_JSON}"

echo "Wrote extraction output: ${OUTPUT_JSON}"
