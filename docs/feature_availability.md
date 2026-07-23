# Feature Availability

This table documents the expected PaySim fields, when they are available, whether they are used for modelling, and the leakage risk.

| Feature | When available | Used for model? | Leakage risk | Reason |
|---|---|---|---|---|
| `step` | At transaction time | Yes | Low | Represents transaction order/time window used for splits and model context. |
| `type` | At transaction time | Yes | Low | Transaction category is a natural signal for fraud patterns. |
| `amount` | At transaction time | Yes | Low | Transaction amount is a core fraud-risk feature. |
| `nameOrig` | At transaction time | No | Medium | Identifier-level data can leak customer identities and is unsuitable as a raw feature. |
| `oldbalanceOrg` | At transaction time | Yes | Medium | Origin balance before the transaction is available and useful, but should be handled carefully. |
| `newbalanceOrig` | After transaction | No | High | Post-transaction balance may leak the transaction outcome. |
| `nameDest` | At transaction time | No | High | Counterparty identifier is sensitive and not used directly in the model. |
| `oldbalanceDest` | At transaction time | Yes | Medium | Destination opening balance is available before the transaction and may help detect anomalies. |
| `newbalanceDest` | After transaction | No | High | Post-transaction balance can leak target behavior. |
| `isFraud` | After transaction | Target only | N/A | This is the label used for supervised training; not a feature. |
| `isFlaggedFraud` | After transaction | Benchmark only | N/A | Used as an existing rule-based comparison baseline, not a model feature. |

> Note: These availability judgments are preliminary and will be revisited after data exploration.
