# Feature Availability

This table documents the expected PaySim fields, when they are available, whether they are used for modelling, and the leakage risk.

| Feature | When available | Used for model? | Leakage risk | Reason |
|---|---|---|---|---|
| `step` | At transaction time | No in Phase 3 baseline | Low to medium | Retained for future chronological splitting; exact PaySim time semantics are not yet verified. |
| `type` | At transaction time | Yes | Low | Transaction category is a natural signal for fraud patterns and is encoded in a fixed order. |
| `amount` | At transaction time | Yes | Low | Transaction amount is a core fraud-risk feature. |
| `nameOrig` | At transaction time | No | Medium | Identifier-level data can leak customer identities and is unsuitable as a raw feature. |
| `oldbalanceOrg` | At transaction time | Yes | Medium | Origin balance before the transaction is used for conservative origin-side relationships. |
| `newbalanceOrig` | After transaction | No | High | Post-transaction balance may leak the transaction outcome. Any derived feature using it is also excluded. |
| `nameDest` | At transaction time | No | High | Counterparty identifier is sensitive and not used directly in the model; historical features are deferred. |
| `oldbalanceDest` | Availability not established for external recipients | No in Phase 3 baseline | Medium to high | Destination balance information and derived features are deferred until decision-time availability is justified. |
| `newbalanceDest` | After transaction | No | High | Post-transaction balance can leak target behavior; any derived feature using it is also excluded. |
| `isFraud` | After transaction | Target only | N/A | This is the label used for supervised training; not a feature. |
| `isFlaggedFraud` | After transaction | Benchmark only | N/A | Used as an existing rule-based comparison baseline, not a model feature. |

> Note: These availability judgments are preliminary and will be revisited after data exploration.

## Phase 3 baseline features

The feature matrix contains exactly these columns, in this order:

1. `amount`
2. `log1p_amount`
3. `oldbalanceOrg`
4. `origin_zero_balance`
5. `amount_exceeds_origin_balance`
6. `amount_to_origin_balance`
7. `type_CASH_IN`
8. `type_CASH_OUT`
9. `type_DEBIT`
10. `type_PAYMENT`
11. `type_TRANSFER`

When `oldbalanceOrg` is zero, `amount_to_origin_balance` is explicitly set to `0.0`; `origin_zero_balance` preserves the information that the denominator was zero. No scaling, imputation, fitting, historical aggregation, or model-specific preprocessing occurs in this phase.

## Phase 4 chronology

The raw `step` field is used as the available chronological ordering variable only. It is not included in `X`, and its interpretation as an hourly index remains unverified. Phase 4 uses whole-step boundaries:

- Train: steps `1-446`
- Validation: steps `447-594`
- Test: steps `595-743`

Transactions sharing a step remain in one partition because the current schema does not provide reliable ordering within a step. Future historical features may use information from strictly earlier steps, but must not treat another transaction from the same step as known prior history. The test partition must remain untouched while models and settings are selected in Phase 5.

## Final modelling status

TEST was later opened once in Phase 6B for the final out-of-time evaluation. It was not used for preprocessing fitting, model fitting, model selection, or threshold selection before that final evaluation, and no post-TEST tuning occurred.
