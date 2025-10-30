# Repository Guidelines

This repository hosts automated trading bots for OKX alongside supporting market insight utilities. Follow these practices to keep contributions predictable, auditable, and safe.

## Project Structure & Module Organization
- `ai_trade_bot_ok_plus.py` and `nofi_bot.py`: primary trading executors scheduled via `schedule` and logging into `logs/`.
- `demo.go`: Go helper for enriched market metrics under the `market` package; keep Go code grouped by package.
- `logs/`: rotated log outputs; never commit runtime log files.
- `requirements.txt`: canonical Python dependencies that should remain minimal and pinned when upgraded.
- Create new Python modules under the repository root or a `strategies/` package, mirroring descriptive snake_case filenames. Place reusable helpers in dedicated modules rather than expanding the main scripts.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create an isolated environment (Windows PowerShell: `.\.venv\Scripts\Activate.ps1`).
- `pip install -r requirements.txt`: install runtime dependencies.
- `python ai_trade_bot_ok_plus.py --symbols BTC/USDT ETH/USDT --timeframe 1h --klineNum 200`: run the enhanced OKX bot; monitor per-coin logs in `logs/`.
- `python nofi_bot.py --symbols BTC/USDT --timeframe 15m --klineNum 120`: execute the simplified variant for quicker validation cycles.
- `go build demo.go`: compile the market data helper; initialise a Go module before extending package boundaries.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, descriptive snake_case function names, and UpperCamelCase for classes or Go types. Keep constants uppercase (for example, `AI_MODEL`). Use pathlib-safe path handling and favour dependency injection for exchange clients. For Go files, run `gofmt` before committing and keep exported names documented via comments.

## Testing Guidelines
Automated tests are not yet committed; prefer `pytest` for Python modules and `go test ./...` for Go packages as they appear. Place new tests under `tests/` mirroring the module structure (for example, `tests/test_ai_trade_bot.py`). Include fixtures that mock exchange responses and sanitise secrets. Target at least 80% coverage on new logic and add regression cases whenever modifying trade signal calculations.

## Commit & Pull Request Guidelines
Follow the Conventional Commits style seen in history (`feat:`, `fix:`, `chore:`). Group logical changes per commit and supply concise imperative subject lines. Pull requests should describe the change, link tracking issues, outline manual verification steps, and include relevant log or CLI samples. Request review from at least one maintainer and ensure checks complete before merging.

## Environment & Secrets
Store API credentials in a strategy-local `.env` file as described in `README.md`. Never commit secrets; instead, update `.env.example` or documentation when configuration keys change. Validate environment variables with `load_dotenv()` defaults to prevent runtime failures and log redaction for sensitive values.
