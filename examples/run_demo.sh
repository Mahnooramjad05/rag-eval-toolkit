#!/usr/bin/env bash
# End-to-end demo: evaluate the example RAG with top-3 and top-5 retrieval,
# compare the two runs, and check the quality gate. Runs offline with the mock judge
# unless LLM_PROVIDER and an API key are set.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=demo_runs
mkdir -p "$OUT"

echo "== Run A: top-3 retrieval"
rageval run --dataset data/sample_qa.jsonl --config configs/default.yaml \
  --adapter python:examples.baseline_rag:answer_top3 --output-dir "$OUT/top3"

echo
echo "== Run B: top-5 retrieval"
rageval run --dataset data/sample_qa.jsonl --config configs/default.yaml \
  --adapter python:examples.baseline_rag:answer_top5 --output-dir "$OUT/top5"

A=$(ls -t "$OUT"/top3/*.json | head -1)
B=$(ls -t "$OUT"/top5/*.json | head -1)

echo
echo "== Compare"
rageval compare "$A" "$B" --markdown "$OUT/comparison.md"

echo "== Quality gate on run A"
rageval gate "$A" --thresholds configs/ci_gate.yaml || echo "(gate failed: see thresholds above)"
