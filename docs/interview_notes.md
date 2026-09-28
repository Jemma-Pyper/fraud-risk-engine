# Interview Notes

This project scaffold establishes the foundation for an explainable fraud risk engine.

- What was built: Initial repo structure, packaging, and documentation for a phased portfolio project.
- Why it matters: A clean foundation helps future development stay organized and makes the project easier to explain.
- How it works: `pyproject.toml` defines package metadata and developer requirements; GitHub Actions runs lint and tests.
- Trade-offs: Minimal dependencies now reduce risk, but more packages will be added later when features require them.
- Non-technical explanation: This is the starter setup for a fraud model project that will later load data, train models, and explain risks.
- Phase 1 adds a safe CSV loader and validation checks so the project can detect invalid transactions before modelling begins.
- The validation step separates hard failures from data-quality warnings, which prevents bad data from silently corrupting later analysis.
- Fraud detection is an imbalanced classification problem: a model can achieve high accuracy by mostly predicting the majority non-fraud class while missing fraud.
- Fraud prevalence is the overall proportion of fraudulent transactions; fraud rate by type is the conditional proportion within each transaction type. A high-volume type and a high-rate type are not necessarily the same.
- Precision measures how many flagged transactions are actually fraud, while recall measures how much fraud the review process catches. Both matter because investigations have limited capacity and false alerts consume analyst time.
- The `isFlaggedFraud` comparison shows what an existing deterministic rule catches and misses; it is a benchmark, not a feature or a replacement for model evaluation.
- EDA is separated from feature engineering and modelling so observed patterns can be documented before choices about transformations, leakage, splits, or algorithms are made.
- Phase 3 intentionally excludes variables that might be predictive but would not be available at scoring time. A feature that improves an offline score through post-transaction information is not a credible production feature; excluding it protects the validity of the fraud-risk system.
- The baseline keeps only amount, origin-side pre-transaction information, simple explainable ratios/indicators, and fixed transaction-type flags. This makes each column easy to describe and audit.
- Phase 4 uses train steps 1-446, validation steps 447-594, and test steps 595-743. Whole steps stay together because the dataset cannot reliably order transactions within a shared step.
- If asked, “Why didn't you use a random train/test split for fraud detection?”, I would explain: “The intended use is to score future transactions. A random split could place future patterns in training and make performance look more reliable than it would be operationally. A chronological split tests whether the approach transfers forward in time.”
- The row counts are not equal across chronological periods because transaction volume changes over time. The later data also has a higher observed fraud rate, so that temporal distribution shift is preserved rather than corrected away.
- The test period is held back until final evaluation. Any future historical account feature must use only earlier steps, because same-step ordering is not available in the schema.
- Logistic regression was chosen first because it is interpretable, produces a continuous score, and provides a defensible linear baseline before more complex models.
- Scaling is fitted on training data only so validation observations cannot influence the learned feature transformation.
- Class weighting was tested because fraud represents only about `0.129%` overall; it changes the fitting emphasis on rare fraud and can improve recall at the cost of many more false alerts.
- Average Precision was the primary validation metric because it focuses on precision-recall behaviour for rare fraud cases; accuracy would mostly reflect the majority class.
- The test partition was deliberately not inspected because using it during model selection would turn the final holdout into another validation set.
- Observed validation results: unweighted logistic regression achieved AP `0.729030`, ROC-AUC `0.984607`, and recall `0.203247` at `0.5`; balanced logistic regression achieved AP `0.692946`, ROC-AUC `0.994853`, and recall `1.000000` but precision `0.133333`.
