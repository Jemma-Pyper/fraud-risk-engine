# Agent Guidance

This repository is a Python portfolio project for an explainable fraud risk engine using PaySim-style transaction data.

## Purpose
- Build a transparent fraud-risk model for transaction review prioritization.
- Keep model outputs interpretable and business-focused.
- Avoid claiming automated fraud decisions.

## Structure
- `src/`: Python package source code.
- `data/`: dataset guidance and processed artifacts.
- `tests/`: automated test coverage.
- `docs/`: design notes, decisions, and model documentation.
- `.github/workflows/`: CI for linting and tests.

## Python conventions
- Use `src/` layout with package importable after install.
- Prefer clear, explicit code and small helper functions.
- Use type hints on public functions where useful.
- Keep dependencies minimal until each phase justifies new packages.

## Installation
- `python -m pip install --upgrade pip`
- `python -m pip install .[dev]`

## Linting and testing
- `ruff check src tests`
- `pytest`

## Data and secrets
- Do not commit raw dataset files.
- Exclude `.env` and raw data directories from Git.
- Add dataset instructions to `data/README.md`.

## Workflow rules
- Work in phase branches like `phase/00-project-scaffold`.
- Make small incremental changes per phase.
- Run tests locally before claiming success.
- Update documentation and decision notes as functionality evolves.
