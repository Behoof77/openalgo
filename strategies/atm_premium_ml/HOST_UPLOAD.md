
# Hosting atm_premium_ml on the OpenAlgo /python page

This strategy is ready to upload and run as a self-hosted OpenAlgo strategy on the
Oracle VM (`https://skopaq.duckdns.org`). It runs in analyzer/papertrade mode: it
logs every decision to `decisions.jsonl` and only reports orders to OpenAlgo, which
simulates them instead of sending them to the broker.

## Prerequisites (already done on the VM)

- Model checkpoints are pre-placed at `strategies/atm_premium_ml/models/ce/option_transformer_complete.pt`
  and `strategies/atm_premium_ml/models/pe/option_transformer_pe_complete.pt`.
- `.env` has `OPENALGO_API_KEY` and `OPENALGO_HOST=https://skopaq.duckdns.org`.
- `torch` (CPU) is installed in the platform `.venv`.
- Analyzer mode is ON (verified: `analyzer_status` returns `mode=analyze`).

## Upload form values (`/python` -> New)

| Field            | Value                         |
|------------------|-------------------------------|
| strategy_name    | `atm_premium_ml`              |
| strategy_file    | `strategies/atm_premium_ml/strategy.py` |
| exchange         | `NFO`                         |
| schedule_start   | `09:15`                       |
| schedule_stop    | `15:15`                       |
| schedule_days    | Mon-Fri                       |

The strategy reads its own env defaults; the host injects `STRATEGY_ID`,
`STRATEGY_NAME`, `OPENALGO_STRATEGY_EXCHANGE` and `OPENALGO_API_KEY` into the child
process. `EXPIRY_DATE` defaults to `11AUG26` (edit line ~20 of `strategy.py` before
upload if you want a later expiry).

## Start and verify

1. Upload the file, then press **Start** on the strategy row.
2. Outside the schedule window the host reports `armed for scheduled start`; the
   APScheduler (IST) launches it at `schedule_start`.
3. Logs land at `log/strategies/<strategy_id>_<timestamp>_IST.log`.
4. The strategy writes its decision stream to `strategies/atm_premium_ml/decisions.jsonl`.
   Expect one line per second: `"side": "NONE", "action": "HOLD", "decision": "confidence-low"`
   while confidence stays below 0.80. Orders appear only at confidence >= 0.80, and
   analyzer mode simulates them (see `log/analyzer/`).

## Behavior notes

- Trades (when triggered) are LIMIT, MIS, quantity 75, with 20-pt target / 10-pt SL
  and a 15:15 square-off. All through the analyzer, so nothing hits the broker.
- The Kronos regime filter is a soft dependency: if the Kronos server is not running
  (`http://127.0.0.1:8000/predict`), the strategy logs `[KRONOS] ... neutral` and
  proceeds on the model signal alone.
