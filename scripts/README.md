# Scripts

## Prerequisites
- `ingest-parse-service` running on `http://localhost:9002`
- `paperqa-service` running on `http://localhost:9003`
- GROBID running on `http://localhost:9070`

## Start GROBID
```bash
bash scripts/start_grobid.sh
```

## Ingest + Parse test
```bash
bash scripts/test_ingest_parse.sh \
  --base-url http://localhost:9002 \
  --arxiv-id 1706.03762 \
  --output-dir storage/downloads \
  --output-json storage/outputs/extract.json
```

## PaperQA test (summary + embedding)
```bash
bash scripts/test_paperqa.sh \
  --base-url http://localhost:9003 \
  --extract-json storage/outputs/extract.json \
  --summary-json storage/outputs/paperqa_summary.json \
  --embed-json storage/outputs/paperqa_embed.json
```
