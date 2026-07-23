# Interview Notes

This project scaffold establishes the foundation for an explainable fraud risk engine.

- What was built: Initial repo structure, packaging, and documentation for a phased portfolio project.
- Why it matters: A clean foundation helps future development stay organized and makes the project easier to explain.
- How it works: `pyproject.toml` defines package metadata and developer requirements; GitHub Actions runs lint and tests.
- Trade-offs: Minimal dependencies now reduce risk, but more packages will be added later when features require them.
- Non-technical explanation: This is the starter setup for a fraud model project that will later load data, train models, and explain risks.
- Phase 1 adds a safe CSV loader and validation checks so the project can detect invalid transactions before modelling begins.
- The validation step separates hard failures from data-quality warnings, which prevents bad data from silently corrupting later analysis.
