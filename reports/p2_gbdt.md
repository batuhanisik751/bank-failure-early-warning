# Prototype 2 gradient boosting, 4-quarter horizon

Fixed out-of-time split (rule 6.2 through `fixed_split_masks`), `features_v2` (83 registry features). Training reports 2002-03-31 to 2008-12-31 (252,330 rows, 554 positives); test reports 2010-03-31 to 2013-12-31 (118,696 rows, 833 positives).

Backend `lightgbm`; parameters chosen by `scripts/tune_gbdt.py` on the inner validation slice (reports 2007Q1-2008Q4, inner training windows closed before 2007-05-30): learning_rate=0.03, min_samples_leaf=200, n_estimators=200, num_leaves=63. `inner_pr_auc` is that slice's PR-AUC; every other column is the untouched test split. `gbdt_mono` applies the registry's monotone signs.

## Inner-validation and test metrics

| model | inner_pr_auc | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | n | n_failures |
|---|---|---|---|---|---|---|---|
| texas | n/a | 0.3726 | 0.9739 | 0.7611 | 0.0720 | 118696 | 833 |
| logit_v2 | 0.2589 | 0.4437 | 0.9823 | 0.7815 | 0.0840 | 118696 | 833 |
| gbdt | 0.2299 | 0.4324 | 0.9840 | 0.7791 | 0.0732 | 118696 | 833 |
| gbdt_mono | 0.1968 | 0.4755 | 0.9838 | 0.8019 | 0.0864 | 118696 | 833 |

Decision Point 2: `settings.models.gbdt.monotone = false` (the variant with the higher inner-validation PR-AUC); the owner decides after comparing both rows above.

## Sensitivity: censored rows dropped (spec 5, rule 3)

| model | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | n | n_failures | n_dropped |
|---|---|---|---|---|---|---|---|
| texas | 0.3815 | 0.9741 | 0.7611 | 0.0732 | 114819 | 833 | 3877 |
| logit_v2 | 0.4535 | 0.9826 | 0.7827 | 0.0840 | 114819 | 833 | 3877 |
| gbdt | 0.4435 | 0.9844 | 0.7803 | 0.0744 | 114819 | 833 | 3877 |
| gbdt_mono | 0.4865 | 0.9840 | 0.8031 | 0.0876 | 114819 | 833 | 3877 |

## Per failure event (spec 5, rule 6)

| model | pr_auc | roc_auc | recall_at_2pct | n_events |
|---|---|---|---|---|
| texas | 0.3728 | 0.9739 | 0.7661 | 821 |
| logit_v2 | 0.4433 | 0.9822 | 0.7820 | 821 |
| gbdt | 0.4330 | 0.9841 | 0.7820 | 821 |
| gbdt_mono | 0.4773 | 0.9839 | 0.8063 | 821 |

## Top 15 features of `gbdt` by gain

| feature | importance |
|---|---|
| d4q_texas_ratio | 42303.04 |
| d4q_equity_to_assets | 14159.69 |
| construction_to_capital | 12273.82 |
| texas_ratio | 4009.12 |
| equity_to_assets | 3558.14 |
| total_rbc_ratio | 3350.17 |
| log_assets | 3261.57 |
| macro_hpi_change_4q | 2494.49 |
| share_residential | 2394.95 |
| macro_unemp_change_4q | 2380.54 |
| adjusted_tier1_leverage | 2304.79 |
| asset_growth_12q | 1988.82 |
| share_nonfarm_nonres | 1932.78 |
| afs_unrealized_to_tier1 | 1922.57 |
| securities_to_assets | 1902.01 |

## Per-year metrics, gbdt

| year | n | n_failures | pr_auc | roc_auc | recall_at_1pct | recall_at_2pct | recall_at_5pct | recall_at_top50 | recall_at_top100 |
|---|---|---|---|---|---|---|---|---|---|
| 2010 | 31431 | 408 | 0.4934 | 0.9791 | 0.4412 | 0.6716 | 0.8750 | 0.0858 | 0.1593 |
| 2011 | 30165 | 231 | 0.4405 | 0.9877 | 0.5801 | 0.7749 | 0.9351 | 0.1126 | 0.2727 |
| 2012 | 29130 | 121 | 0.3073 | 0.9889 | 0.5950 | 0.8017 | 0.9587 | 0.1901 | 0.3306 |
| 2013 | 27970 | 73 | 0.2858 | 0.9701 | 0.6027 | 0.7397 | 0.8767 | 0.2740 | 0.4247 |
| pooled | 118696 | 833 | 0.4324 | 0.9840 | 0.5858 | 0.7791 | 0.9244 | 0.0384 | 0.0732 |

## Per-year metrics, gbdt_mono

| year | n | n_failures | pr_auc | roc_auc | recall_at_1pct | recall_at_2pct | recall_at_5pct | recall_at_top50 | recall_at_top100 |
|---|---|---|---|---|---|---|---|---|---|
| 2010 | 31431 | 408 | 0.5313 | 0.9787 | 0.4730 | 0.6740 | 0.9118 | 0.0882 | 0.1765 |
| 2011 | 30165 | 231 | 0.4859 | 0.9907 | 0.6190 | 0.8052 | 0.9697 | 0.1472 | 0.2641 |
| 2012 | 29130 | 121 | 0.3952 | 0.9923 | 0.6612 | 0.8512 | 0.9835 | 0.2066 | 0.3802 |
| 2013 | 27970 | 73 | 0.2760 | 0.9581 | 0.6575 | 0.7945 | 0.9178 | 0.2603 | 0.4521 |
| pooled | 118696 | 833 | 0.4755 | 0.9838 | 0.6050 | 0.8019 | 0.9412 | 0.0456 | 0.0864 |
