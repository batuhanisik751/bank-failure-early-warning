# Prototype 2 gradient boosting, 4-quarter horizon

Fixed out-of-time split (rule 6.2 through `fixed_split_masks`), `features_v2` (83 registry features). Training reports 2002-03-31 to 2008-12-31 (252,330 rows, 554 positives); test reports 2010-03-31 to 2013-12-31 (118,696 rows, 833 positives).

Backend `lightgbm`; parameters chosen by `scripts/tune_gbdt.py` on the inner validation slice (reports 2007Q1-2008Q4, inner training windows closed before 2007-05-30): learning_rate=0.03, min_samples_leaf=50, n_estimators=200, num_leaves=31. `inner_pr_auc` is that slice's PR-AUC; every other column is the untouched test split. `gbdt_mono` applies the registry's monotone signs.

## Inner-validation and test metrics

| model | inner_pr_auc | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | n | n_failures |
|---|---|---|---|---|---|---|---|
| texas | n/a | 0.3726 | 0.9739 | 0.7611 | 0.0720 | 118696 | 833 |
| logit_v2 | 0.2589 | 0.4437 | 0.9823 | 0.7815 | 0.0840 | 118696 | 833 |
| gbdt | 0.1583 | 0.3913 | 0.9814 | 0.7887 | 0.0792 | 118696 | 833 |
| gbdt_mono | 0.2142 | 0.4250 | 0.9832 | 0.7959 | 0.0804 | 118696 | 833 |

Decision Point 2: `settings.models.gbdt.monotone = true` (the variant with the higher inner-validation PR-AUC); the owner decides after comparing both rows above.

## Sensitivity: censored rows dropped (spec 5, rule 3)

| model | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | n | n_failures | n_dropped |
|---|---|---|---|---|---|---|---|
| texas | 0.3815 | 0.9741 | 0.7611 | 0.0732 | 114819 | 833 | 3877 |
| logit_v2 | 0.4535 | 0.9826 | 0.7827 | 0.0840 | 114819 | 833 | 3877 |
| gbdt | 0.4048 | 0.9819 | 0.7923 | 0.0792 | 114819 | 833 | 3877 |
| gbdt_mono | 0.4351 | 0.9835 | 0.7971 | 0.0804 | 114819 | 833 | 3877 |

## Per failure event (spec 5, rule 6)

| model | pr_auc | roc_auc | recall_at_2pct | n_events |
|---|---|---|---|---|
| texas | 0.3728 | 0.9739 | 0.7661 | 821 |
| logit_v2 | 0.4433 | 0.9822 | 0.7820 | 821 |
| gbdt | 0.3918 | 0.9815 | 0.7929 | 821 |
| gbdt_mono | 0.4268 | 0.9834 | 0.8051 | 821 |

## Top 15 features of `gbdt` by gain

| feature | importance |
|---|---|
| d4q_texas_ratio | 42362.22 |
| d4q_equity_to_assets | 11109.92 |
| nim_q | 7920.50 |
| macro_hpi_change_4q | 7426.67 |
| total_rbc_ratio | 6193.41 |
| asset_growth_12q | 4079.87 |
| texas_ratio | 3829.48 |
| log_assets | 3584.48 |
| cre_to_capital | 3495.92 |
| d4q_roa_q | 3093.68 |
| share_residential | 2942.75 |
| equity_to_assets | 2845.30 |
| tangible_equity_to_assets | 2566.25 |
| d4q_noncurrent_ratio | 2145.00 |
| share_consumer | 2033.57 |

## Per-year metrics, gbdt

| year | n | n_failures | pr_auc | roc_auc | recall_at_1pct | recall_at_2pct | recall_at_5pct | recall_at_top50 | recall_at_top100 |
|---|---|---|---|---|---|---|---|---|---|
| 2010 | 31431 | 408 | 0.4438 | 0.9762 | 0.3627 | 0.6029 | 0.9069 | 0.0833 | 0.1397 |
| 2011 | 30165 | 231 | 0.4099 | 0.9853 | 0.4762 | 0.7792 | 0.9221 | 0.1342 | 0.2294 |
| 2012 | 29130 | 121 | 0.2833 | 0.9890 | 0.6033 | 0.8264 | 0.9421 | 0.1653 | 0.2562 |
| 2013 | 27970 | 73 | 0.2159 | 0.9627 | 0.5068 | 0.6438 | 0.8493 | 0.2466 | 0.3699 |
| pooled | 118696 | 833 | 0.3913 | 0.9814 | 0.5330 | 0.7887 | 0.9076 | 0.0468 | 0.0792 |

## Per-year metrics, gbdt_mono

| year | n | n_failures | pr_auc | roc_auc | recall_at_1pct | recall_at_2pct | recall_at_5pct | recall_at_top50 | recall_at_top100 |
|---|---|---|---|---|---|---|---|---|---|
| 2010 | 31431 | 408 | 0.4662 | 0.9775 | 0.4020 | 0.6544 | 0.8971 | 0.0858 | 0.1495 |
| 2011 | 30165 | 231 | 0.4650 | 0.9883 | 0.5844 | 0.8139 | 0.9567 | 0.1169 | 0.2468 |
| 2012 | 29130 | 121 | 0.3119 | 0.9913 | 0.6198 | 0.8512 | 0.9917 | 0.1488 | 0.3140 |
| 2013 | 27970 | 73 | 0.3003 | 0.9647 | 0.7123 | 0.7808 | 0.8630 | 0.3014 | 0.4932 |
| pooled | 118696 | 833 | 0.4250 | 0.9832 | 0.5870 | 0.7959 | 0.9316 | 0.0408 | 0.0804 |
