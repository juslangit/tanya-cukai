#!/bin/bash
# Start Tanya Cukai: two local llama.cpp servers (answers + embeddings) and the web page.
# Everything listens on 127.0.0.1 only — nothing is reachable from the network, nothing is sent out.
set -e
cd "$(dirname "$0")"
MODELS="${MODELS:-$HOME/.cache/llm-models}"
CHAT_MODEL="${CHAT_MODEL:-$MODELS/Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
EMBED_MODEL="$MODELS/bge-m3-q8_0.gguf"
LOGS=data/logs; mkdir -p "$LOGS"

wait_up() { for _ in $(seq 120); do curl -sf "http://127.0.0.1:$1/health" >/dev/null && return; sleep 1; done; echo "port $1 did not start, see $LOGS"; exit 1; }

llama-server -m "$CHAT_MODEL" --host 127.0.0.1 --port 8080 -c 6144 -ngl 99 >"$LOGS/chat.log" 2>&1 &
CHAT=$!
llama-server -m "$EMBED_MODEL" --host 127.0.0.1 --port 8081 --embedding -c 8192 -b 8192 -ub 8192 -ngl 99 >"$LOGS/embed.log" 2>&1 &
EMB=$!
trap 'kill $CHAT $EMB 2>/dev/null' EXIT
wait_up 8080; wait_up 8081

[ -f data/index.sqlite ] || { uv run scripts/fetch.py; uv run scripts/ingest.py; }
uv run app/server.py
