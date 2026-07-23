# Decisions

## 2026-07-22
- Chose a `src/` package layout with `pyproject.toml` and setuptools.
- Kept dependencies minimal for Phase 0: only `pytest` and `ruff`.
- Added data guidance to `data/README.md` and excluded raw data from Git.
- Created `AGENTS.md` to guide future agents on project purpose and workflow.

## 2026-07-23
- Added PaySim ingestion and validation modules.
- Decided to treat `isFraud` as the target and `isFlaggedFraud` as a benchmark only.
- Decided to reject negative amounts and balances, while warning on zero amounts.
- Decided to detect duplicate rows without dropping or rejecting them in Phase 1.
