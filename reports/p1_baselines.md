# Prototype 1 baselines, 4-quarter horizon

Fixed out-of-time split (spec 8.2, rule 6.2 applied through `fixed_split_masks`).

- Training reports actually used: 2002-03-31 to 2008-12-31 (nominal 2002-03-31 to 2008-12-31; 252,330 rows, 554 positives).
- Test reports: 2010-03-31 to 2013-12-31 (118,696 rows, 833 positives); first test prediction date 2010-05-30 with a 60-day lag.

## Test-split metrics

| model | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | n | n_failures |
|---|---|---|---|---|---|---|
| texas | 0.3726 | 0.9739 | 0.7611 | 0.0720 | 118696 | 833 |
| logit_small | 0.3720 | 0.9791 | 0.7383 | 0.0792 | 118696 | 833 |
| logit | 0.3867 | 0.9755 | 0.7143 | 0.0804 | 118696 | 833 |

Acceptance check (spec 9): the regularised logit beats the Texas ratio on PR-AUC (0.3867 vs 0.3726).

## Sensitivity: censored rows dropped (spec 5, rule 3)

Test rows where the bank left the industry without failing inside the window (`censored_in_window_4q`: merged, closed voluntarily, ...) are removed before scoring; `n_dropped` counts them. They carry `y = 0` in the main run.

| model | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | n | n_failures | n_dropped |
|---|---|---|---|---|---|---|---|
| texas | 0.3815 | 0.9741 | 0.7611 | 0.0732 | 114819 | 833 | 3877 |
| logit_small | 0.3840 | 0.9794 | 0.7407 | 0.0804 | 114819 | 833 | 3877 |
| logit | 0.3973 | 0.9758 | 0.7155 | 0.0804 | 114819 | 833 | 3877 |

## Per failure event (spec 5, rule 6)

Sister banks of one holding company (`rssdhcr`) that failed on the same day are collapsed into one event per report quarter, scored by the best-ranked sister; every other row stays one per bank. `n_events` is the number of positive units after collapsing, `n_multi_bank_events` how many of them bundle several banks.

| model | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | n_events | n_multi_bank_events | n_banks_in_multi_events |
|---|---|---|---|---|---|---|---|
| texas | 0.3728 | 0.9739 | 0.7661 | 0.0731 | 821 | 12 | 24 |
| logit_small | 0.3706 | 0.9790 | 0.7418 | 0.0804 | 821 | 12 | 24 |
| logit | 0.3858 | 0.9753 | 0.7125 | 0.0816 | 821 | 12 | 24 |

## Figures

- pr curve texas: [`figures/pr_curve_texas.png`](figures/pr_curve_texas.png)
- score distributions texas: [`figures/score_distributions_texas.png`](figures/score_distributions_texas.png)
- pr curve logit small: [`figures/pr_curve_logit_small.png`](figures/pr_curve_logit_small.png)
- score distributions logit small: [`figures/score_distributions_logit_small.png`](figures/score_distributions_logit_small.png)
- pr curve logit: [`figures/pr_curve_logit.png`](figures/pr_curve_logit.png)
- score distributions logit: [`figures/score_distributions_logit.png`](figures/score_distributions_logit.png)
- recall at k: [`figures/recall_at_k.png`](figures/recall_at_k.png)

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
| 2010 | 31431 | 408 | 0.4458 | 0.9727 | 0.3946 | 0.6127 | 0.8799 | 0.0784 | 0.1544 |
| 2011 | 30165 | 231 | 0.3858 | 0.9771 | 0.4805 | 0.6710 | 0.9134 | 0.1385 | 0.2338 |
| 2012 | 29130 | 121 | 0.3186 | 0.9880 | 0.5702 | 0.7438 | 0.9669 | 0.1901 | 0.3058 |
| 2013 | 27970 | 73 | 0.2046 | 0.9454 | 0.5890 | 0.6712 | 0.8904 | 0.1918 | 0.3014 |
| pooled | 118696 | 833 | 0.3867 | 0.9755 | 0.5222 | 0.7143 | 0.9148 | 0.0396 | 0.0804 |

## Top 10 |coefficient| features, logit

| feature | coef | odds_ratio |
|---|---|---|
| texas_ratio | 0.4085 | 1.5046 |
| construction_to_capital | 0.2613 | 1.2986 |
| roa_q | -0.2542 | 0.7755 |
| share_multifamily | 0.1514 | 1.1635 |
| equity_to_assets | -0.1365 | 0.8724 |
| tangible_equity_to_assets | -0.1254 | 0.8821 |
| noncurrent_ratio | 0.1215 | 1.1292 |
| wholesale_funding_ratio | 0.1165 | 1.1236 |
| early_delinquency | 0.1162 | 1.1233 |
| tier1_leverage | -0.1161 | 0.8904 |

Leakage sanity check: the largest coefficient (`texas_ratio`) carries 10% of the total absolute coefficient mass across 71 inputs; no single feature dominates, consistent with no look-ahead leakage.
