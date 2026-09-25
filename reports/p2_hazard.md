# Prototype 2 discrete-time hazard model

Logistic regression on bank-quarter rows whose event is *fails inside the next quarter* (`y_1q`: failure date in `(avail_date, avail_date + 3 months]`), with the Prototype 1 preprocessing pipeline (winsorise, median-impute with missing flags, scale), no class weighting and L2 at `C = 0.0003` on all 83 `features_v2` registry features. Every quarter a bank is observed is one row with that quarter's covariates (Shumway 2001); a bank that leaves by merger or voluntary closing has no rows after it exits, which is right-censoring and needs no deletion. Rows selected through `fixed_split_masks` at 1q and checked with `assert_no_leakage`: training reports 2002-03-31 to 2008-12-31 (252,330 rows, 75 one-quarter failures); test reports 2010-03-31 to 2013-12-31 (118,696 rows, 256).

## Choice of C (inner validation slice, rule 6.7)

Validation = usable reports 2007Q1-2008Q4; inner training = fixed-split training rows whose 1q window closed before 2007-05-30. Each candidate is scored on the validation rows at 1q and, converted, at 4q; the winner is the highest 4q PR-AUC, the horizon the model is compared on.

| C | pr_auc_1q | roc_auc_1q | pr_auc_4q | recall_at_2pct_4q | brier_4q |
|---|---|---|---|---|---|
| 0.0003 | 0.0963 | 0.9861 | 0.2844 | 0.6031 | 0.0074 |
| 0.0030 | 0.0919 | 0.9864 | 0.2837 | 0.6051 | 0.0072 |
| 0.0010 | 0.0871 | 0.9863 | 0.2737 | 0.6031 | 0.0074 |
| 0.0100 | 0.0691 | 0.9827 | 0.2144 | 0.5246 | 0.0072 |
| 0.0300 | 0.0643 | 0.9794 | 0.1928 | 0.5029 | 0.0072 |
| 0.1000 | 0.0606 | 0.9763 | 0.1815 | 0.4892 | 0.0071 |
| 1.0000 | 0.0591 | 0.9746 | 0.1760 | 0.4853 | 0.0071 |

## One-quarter test metrics (the hazard's own event)

| model | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | brier | n | n_failures |
|---|---|---|---|---|---|---|---|
| hazard | 0.2598 | 0.9834 | 0.8086 | 0.1562 | 0.0021 | 118696 | 256 |

## Conversion to 4 and 8 quarters

`p_Hq = 1 - (1 - h)^H` assumes the quarterly hazard estimated from today's covariates persists at the current level for the next H quarters; covariates are not projected forward, so a bank whose condition is deteriorating is under-predicted at long horizons and one that is recovering over-predicted.

### 4q test rows: hazard converted with `1 - (1 - h)^4` against the models fitted on that label

| model | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | brier | n | n_failures |
|---|---|---|---|---|---|---|---|
| hazard (converted) | 0.3633 | 0.9759 | 0.6903 | 0.0792 | 0.0064 | 118696 | 833 |
| logit_v2 | 0.4437 | 0.9823 | 0.7815 | 0.0840 | 0.0051 | 118696 | 833 |
| gbdt | 0.4324 | 0.9840 | 0.7791 | 0.0732 | 0.0052 | 118696 | 833 |

Brier references at 4q: raw unconverted hazard 0.00684, constant at the realised failure rate 0.00697. Ranking metrics of the hazard are identical before and after conversion; only Brier moves.

### 8q test rows: hazard converted with `1 - (1 - h)^8` against the models fitted on that label

| model | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | brier | n | n_failures |
|---|---|---|---|---|---|---|---|
| hazard (converted) | 0.3948 | 0.9673 | 0.6046 | 0.0564 | 0.0096 | 118696 | 1295 |
| logit_v2 | 0.3764 | 0.9658 | 0.5846 | 0.0471 | 0.0084 | 118696 | 1295 |
| gbdt | 0.1328 | 0.9029 | 0.2911 | 0.0263 | 0.0105 | 118696 | 1295 |

Brier references at 8q: raw unconverted hazard 0.01068, constant at the realised failure rate 0.01079. Ranking metrics of the hazard are identical before and after conversion; only Brier moves.

## Odds ratios per one training standard deviation (95% confidence intervals)

Method: statsmodels Logit, unpenalised, cluster-robust standard errors by cert, on the standardised design of these 15 features alone (no missing indicators). `full_model_odds_ratio` is the L2-shrunk coefficient of the same feature in the 83-feature hazard.

| feature | camels_group | odds_ratio | ci_low | ci_high | p_value | full_model_odds_ratio |
|---|---|---|---|---|---|---|
| equity_to_assets | capital | 0.0490 | 0.0014 | 1.7218 | 0.0968 | 0.9914 |
| noncurrent_ratio | asset_quality | 1.8895 | 1.6290 | 2.1917 | 0.0000 | 1.0754 |
| roa_q | earnings | 0.7342 | 0.6298 | 0.8558 | 0.0001 | 0.9479 |
| brokered_share | liquidity | 1.3213 | 1.1759 | 1.4847 | 0.0000 | 1.0442 |
| construction_to_capital | concentration | 1.1339 | 0.9688 | 1.3271 | 0.1176 | 1.0395 |
| log_assets | structure | 1.2421 | 0.9445 | 1.6334 | 0.1208 | 1.0084 |
| afs_unrealized_to_tier1 | sensitivity | 0.9883 | 0.5334 | 1.8313 | 0.9702 | 0.9973 |
| htm_unrealized_to_tier1 | sensitivity | 0.8166 | 0.5545 | 1.2026 | 0.3050 | 0.9957 |
| unrealized_loss_to_tier1 | sensitivity | 1.1631 | 0.5577 | 2.4256 | 0.6870 | 0.9961 |
| adjusted_tier1_leverage | capital | 0.0668 | 0.0043 | 1.0388 | 0.0533 | 0.9916 |
| securities_to_assets | sensitivity | 0.3072 | 0.1633 | 0.5779 | 0.0003 | 0.9940 |
| htm_share_of_securities | sensitivity | 1.0310 | 0.7790 | 1.3645 | 0.8311 | 1.0003 |
| uninsured_share | liquidity | 0.9875 | 0.6661 | 1.4640 | 0.9502 | 0.9940 |
| uninsured_to_liquid_assets | liquidity | 0.6696 | 0.3949 | 1.1353 | 0.1365 | 0.9942 |
| large_time_deposit_share | liquidity | 0.8761 | 0.7051 | 1.0887 | 0.2329 | 0.9990 |

## Sensitivity: censored rows dropped (spec 5, rule 3), 1q

| pr_auc | roc_auc | recall_at_1pct | recall_at_2pct | recall_at_5pct | recall_at_top50 | recall_at_top100 | n | n_failures | n_dropped |
|---|---|---|---|---|---|---|---|---|---|
| 0.2624 | 0.9834 | 0.6992 | 0.8125 | 0.9414 | 0.0859 | 0.1562 | 117743 | 256 | 953 |

## Per failure event (spec 5, rule 6), 1q

| pr_auc | roc_auc | recall_at_1pct | recall_at_2pct | recall_at_5pct | recall_at_top50 | recall_at_top100 | n | n_failures | n_events | n_multi_bank_events | n_banks_in_multi_events |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2607 | 0.9834 | 0.7036 | 0.8142 | 0.9407 | 0.0870 | 0.1581 | 118693 | 253 | 253 | 3 | 6 |
