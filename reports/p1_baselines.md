# Prototype 1 baselines, 4-quarter horizon

Fixed out-of-time split (spec 8.2, rule 6.2 applied through `fixed_split_masks`).

- Training reports actually used: 2002-03-31 to 2008-12-31 (nominal 2002-03-31 to 2008-12-31; 252,330 rows, 554 positives).
- Test reports: 2010-03-31 to 2013-12-31 (118,696 rows, 833 positives); first test prediction date 2010-05-30 with a 60-day lag.

## Test-split metrics

| model | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | n | n_failures |
|---|---|---|---|---|---|---|
| texas | 0.3726 | 0.9739 | 0.7611 | 0.0720 | 118696 | 833 |
| logit_small | 0.3720 | 0.9791 | 0.7383 | 0.0792 | 118696 | 833 |
| logit | 0.1985 | 0.9635 | 0.5390 | 0.0396 | 118696 | 833 |

Acceptance check (spec 9): the regularised logit does not beat the Texas ratio on PR-AUC (0.1985 vs 0.3726).

## Per-year metrics, logit_small

| year | n | n_failures | pr_auc | roc_auc | recall_at_1pct | recall_at_2pct | recall_at_5pct | recall_at_top50 | recall_at_top100 |
|---|---|---|---|---|---|---|---|---|---|
| 2010 | 31431 | 408 | 0.4153 | 0.9717 | 0.3652 | 0.5980 | 0.8750 | 0.0907 | 0.1593 |
| 2011 | 30165 | 231 | 0.3773 | 0.9823 | 0.4632 | 0.6667 | 0.9048 | 0.1342 | 0.2251 |
| 2012 | 29130 | 121 | 0.3585 | 0.9877 | 0.6033 | 0.8099 | 0.9339 | 0.2397 | 0.3554 |
| 2013 | 27970 | 73 | 0.1721 | 0.9712 | 0.6027 | 0.6849 | 0.8082 | 0.1096 | 0.2466 |
| pooled | 118696 | 833 | 0.3720 | 0.9791 | 0.5186 | 0.7383 | 0.9040 | 0.0444 | 0.0792 |

## Odds ratios, logit_small (per training-fold standard deviation)

| feature | coef | odds_ratio |
|---|---|---|
| equity_to_assets | -2.1897 | 0.1119 |
| roa_q | -0.7818 | 0.4576 |
| missingindicator_noncurrent_ratio | -0.7321 | 0.4809 |
| noncurrent_ratio | 0.6852 | 1.9842 |
| construction_to_capital | 0.4655 | 1.5928 |
| brokered_share | 0.4003 | 1.4923 |
| missingindicator_brokered_share | -0.2024 | 0.8167 |
| missingindicator_equity_to_assets | -0.1355 | 0.8733 |
| missingindicator_roa_q | -0.1355 | 0.8733 |
| log_assets | -0.0823 | 0.9210 |
| missingindicator_construction_to_capital | -0.0181 | 0.9821 |

## Per-year metrics, logit

| year | n | n_failures | pr_auc | roc_auc | recall_at_1pct | recall_at_2pct | recall_at_5pct | recall_at_top50 | recall_at_top100 |
|---|---|---|---|---|---|---|---|---|---|
| 2010 | 31431 | 408 | 0.2630 | 0.9619 | 0.2500 | 0.4387 | 0.7426 | 0.0466 | 0.1078 |
| 2011 | 30165 | 231 | 0.2190 | 0.9700 | 0.3290 | 0.5022 | 0.8095 | 0.0866 | 0.1645 |
| 2012 | 29130 | 121 | 0.0902 | 0.9621 | 0.2893 | 0.4545 | 0.7851 | 0.0579 | 0.0909 |
| 2013 | 27970 | 73 | 0.0578 | 0.9151 | 0.2877 | 0.4521 | 0.7671 | 0.0685 | 0.1370 |
| pooled | 118696 | 833 | 0.1985 | 0.9635 | 0.3385 | 0.5390 | 0.8043 | 0.0204 | 0.0396 |

## Top 10 |coefficient| features, logit

| feature | coef | odds_ratio |
|---|---|---|
| tier1_leverage | -3.6657 | 0.0256 |
| tangible_equity_to_assets | -3.4169 | 0.0328 |
| total_rbc_ratio | 2.8723 | 17.6770 |
| cre_to_capital | -1.6039 | 0.2011 |
| construction_to_capital | 0.9593 | 2.6100 |
| brokered_share | 0.7996 | 2.2247 |
| equity_to_assets | 0.7844 | 2.1912 |
| share_multifamily | 0.7471 | 2.1108 |
| noncurrent_ratio | 0.7155 | 2.0453 |
| share_consumer | -0.7043 | 0.4945 |

Leakage sanity check: the largest coefficient (`tier1_leverage`) carries 12% of the total absolute coefficient mass across 71 inputs; no single feature dominates, consistent with no look-ahead leakage.
