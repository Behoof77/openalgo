#!/bin/bash
set -e
cd /home/ubuntu/openalgo
API_KEY=$(grep ^OPENALGO_API_KEY .env | cut -d= -f2)
echo "API_KEY length: ${#API_KEY}"
curl -s --unix-socket /home/ubuntu/openalgo/openalgo.sock \
  -X POST \
  "http://localhost/api/v1/history" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{"symbol":"NIFTY","exchange":"NSE_INDEX","interval":"15m","start_date":"2026-07-01","end_date":"2026-07-28"}' \
  -o /tmp/hist_response.json
echo "Response size: $(wc -c < /tmp/hist_response.json) bytes"
head -c 300 /tmp/hist_response.json
