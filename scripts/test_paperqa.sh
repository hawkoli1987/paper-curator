#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://localhost:9003"
# Consume the GROBID extract output as the source text for LLM and embeddings.
EXTRACT_JSON="storage/outputs/extract.json"
# Persist results in repo storage for later inspection or downstream steps.
SUMMARY_JSON="storage/outputs/paperqa_summary.json"
EMBED_JSON="storage/outputs/paperqa_embed.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --extract-json)
      EXTRACT_JSON="$2"
      shift 2
      ;;
    --summary-json)
      SUMMARY_JSON="$2"
      shift 2
      ;;
    --embed-json)
      EMBED_JSON="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ ! -f "${EXTRACT_JSON}" ]]; then
  echo "Missing extract JSON: ${EXTRACT_JSON}"
  exit 1
fi

mkdir -p "$(dirname "${SUMMARY_JSON}")"
mkdir -p "$(dirname "${EMBED_JSON}")"

TEXT_PAYLOAD="$(EXTRACT_JSON="${EXTRACT_JSON}" python - <<'PY'
import json
from pathlib import Path

import os

path = Path(os.environ["EXTRACT_JSON"])
data = json.loads(path.read_text(encoding="utf-8"))
abstract = data.get("abstract") or ""
sections = data.get("sections") or []
section_texts = []
for section in sections:
    text = section.get("text")
    if text:
        section_texts.append(text)
full_text = "\n\n".join([abstract] + section_texts).strip()
if not full_text:
    raise SystemExit("No text extracted from GROBID output.")

# Limit payload to avoid oversized requests while keeping enough context.
payload = {"text": full_text[:12000]}
print(json.dumps(payload))
PY
)"

echo "Summarizing"
curl -f -sS -X POST "${BASE_URL}/summarize" \
  -H "Content-Type: application/json" \
  -d "${TEXT_PAYLOAD}" \
  > "${SUMMARY_JSON}"

echo "Embedding"
curl -f -sS -X POST "${BASE_URL}/embed" \
  -H "Content-Type: application/json" \
  -d "${TEXT_PAYLOAD}" \
  > "${EMBED_JSON}"

echo "Summary output: ${SUMMARY_JSON}"
echo "Embedding output: ${EMBED_JSON}"
