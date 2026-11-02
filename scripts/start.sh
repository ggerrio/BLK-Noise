#!/usr/bin/env bash
# Start FastAPI (internal) and Streamlit (public) in one container.
# Both share the SQLite database; either process exiting stops the container.

set -euo pipefail

API_PORT="${API_PORT:-8000}"
UI_PORT="${PORT:-7860}"
DB_PATH="${DB_PATH:-data/listings.db}"
CSV_PATH="${CSV_PATH:-ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv}"

export API_URL="${API_URL:-http://127.0.0.1:${API_PORT}}"
export DB_PATH

if [ ! -f "$DB_PATH" ]; then
    if [ -f "$CSV_PATH" ]; then
        echo "[start] no database at $DB_PATH; building from $CSV_PATH (slow path)"
        python scripts/load_data.py "$CSV_PATH" --db "$DB_PATH"
        python scripts/build_embeddings.py --db "$DB_PATH"
    else
        echo "[start] no database and no CSV; /search returns 503 until one exists" >&2
    fi
fi

python -m uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT" --proxy-headers &
api_pid=$!

streamlit run ui/app.py \
    --server.address 0.0.0.0 \
    --server.port "$UI_PORT" \
    --server.headless true &
ui_pid=$!

cleanup() {
    kill "$api_pid" "$ui_pid" 2>/dev/null || true
}
trap cleanup TERM INT

set +e
wait -n "$api_pid" "$ui_pid"
exit_code=$?
set -e

cleanup
exit "$exit_code"
