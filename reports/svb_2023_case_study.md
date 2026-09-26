# The 2023 case study: credit-only against rate-aware

Training rows: every bank-quarter whose 4q outcome window closed before the 2022-12-31 prediction date (2023-03-01), reports 2001-03-31..2021-09-30, 622,341 rows and 2,226 failures. Scored reports: 2022-09-30, 2022-12-31, 2023-03-31 (every bank). Views: `credit_only` = the 43 `features_v1` columns, `rate_aware` = the 83 `features_v2` columns. Learners: `logit` at the P1 `LOGIT_C`, `gbdt` at `settings.models.gbdt`. Rank 1 is the riskiest bank of the quarter; percentile is the share of scored banks ranked below it. Run records: `runs/case_study_2023/`.

## credit_only / logit

| bank | cert | quarter | probability | rank | percentile | n_scored |
|---|---|---|---|---|---|---|
| Signature Bank | 57053 | 2022Q3 | 0.0011 | 172 | 96.4263 | 4813 |
| First Republic Bank | 59017 | 2022Q3 | 0.0003 | 1760 | 63.4324 | 4813 |
| Silicon Valley Bank | 24735 | 2022Q3 | 0.0002 | 2336 | 51.4648 | 4813 |
| Signature Bank | 57053 | 2022Q4 | 0.0014 | 113 | 97.6325 | 4773 |
| First Republic Bank | 59017 | 2022Q4 | 0.0003 | 1570 | 67.1066 | 4773 |
| Silicon Valley Bank | 24735 | 2022Q4 | 0.0003 | 1677 | 64.8649 | 4773 |
| First Republic Bank | 59017 | 2023Q1 | 0.0009 | 308 | 93.5021 | 4740 |

Scored reports with a complete label: pr_auc 0.0145, recall_at_top100 0.2308, n 14326, n_failures 13

## credit_only / gbdt

| bank | cert | quarter | probability | rank | percentile | n_scored |
|---|---|---|---|---|---|---|
| First Republic Bank | 59017 | 2022Q3 | 0.0001 | 1423 | 70.4342 | 4813 |
| Silicon Valley Bank | 24735 | 2022Q3 | 0.0001 | 1577 | 67.2346 | 4813 |
| Signature Bank | 57053 | 2022Q3 | 0.0000 | 2515 | 47.7457 | 4813 |
| First Republic Bank | 59017 | 2022Q4 | 0.0001 | 1167 | 75.5500 | 4773 |
| Silicon Valley Bank | 24735 | 2022Q4 | 0.0001 | 1412 | 70.4169 | 4773 |
| Signature Bank | 57053 | 2022Q4 | 0.0001 | 1648 | 65.4724 | 4773 |
| First Republic Bank | 59017 | 2023Q1 | 0.0001 | 904 | 80.9283 | 4740 |

Scored reports with a complete label: pr_auc 0.0292, recall_at_top100 0.3077, n 14326, n_failures 13

## rate_aware / logit

| bank | cert | quarter | probability | rank | percentile | n_scored |
|---|---|---|---|---|---|---|
| Signature Bank | 57053 | 2022Q3 | 0.0002 | 1031 | 78.5788 | 4813 |
| First Republic Bank | 59017 | 2022Q3 | 0.0001 | 2698 | 43.9435 | 4813 |
| Silicon Valley Bank | 24735 | 2022Q3 | 0.0001 | 3315 | 31.1240 | 4813 |
| Signature Bank | 57053 | 2022Q4 | 0.0005 | 627 | 86.8636 | 4773 |
| First Republic Bank | 59017 | 2022Q4 | 0.0002 | 1750 | 63.3354 | 4773 |
| Silicon Valley Bank | 24735 | 2022Q4 | 0.0002 | 1854 | 61.1565 | 4773 |
| First Republic Bank | 59017 | 2023Q1 | 0.0008 | 225 | 95.2532 | 4740 |

Scored reports with a complete label: pr_auc 0.0228, recall_at_top100 0.2308, n 14326, n_failures 13

## rate_aware / gbdt

| bank | cert | quarter | probability | rank | percentile | n_scored |
|---|---|---|---|---|---|---|
| Silicon Valley Bank | 24735 | 2022Q3 | 0.0003 | 274 | 94.3071 | 4813 |
| Signature Bank | 57053 | 2022Q3 | 0.0001 | 1768 | 63.2662 | 4813 |
| First Republic Bank | 59017 | 2022Q3 | 0.0001 | 1889 | 60.7521 | 4813 |
| Silicon Valley Bank | 24735 | 2022Q4 | 0.0003 | 245 | 94.8670 | 4773 |
| First Republic Bank | 59017 | 2022Q4 | 0.0001 | 1416 | 70.3331 | 4773 |
| Signature Bank | 57053 | 2022Q4 | 0.0000 | 2634 | 44.8146 | 4773 |
| First Republic Bank | 59017 | 2023Q1 | 0.0001 | 1299 | 72.5949 | 4740 |

Scored reports with a complete label: pr_auc 0.0180, recall_at_top100 0.0769, n 14326, n_failures 13

## Drivers, credit_only / logit (baseline log-odds -7.947, contributions sum -0.168)

| feature | value | contribution | direction |
|---|---|---|---|
| log_assets | 19.1580 | -0.4007 | safer |
| cre_to_capital | 0.1338 | -0.3810 | safer |
| nim_q | 0.0210 | 0.2822 | riskier |
| liquid_assets_ratio | 0.6244 | -0.2642 | safer |
| texas_ratio | 0.0088 | -0.2455 | safer |
| share_nonfarm_nonres | 0.0235 | 0.2345 | riskier |
| share_ci | 0.2572 | 0.2324 | riskier |
| bkclass_SM | True | 0.1876 | riskier |
| early_delinquency | 0.0014 | -0.1830 | safer |
| construction_to_capital | 0.0237 | -0.1685 | safer |

## Drivers, credit_only / gbdt (baseline log-odds -9.615, contributions sum -0.169)

| feature | value | contribution | direction |
|---|---|---|---|
| texas_ratio | 0.0088 | -0.6216 | safer |
| liquid_assets_ratio | 0.6244 | -0.2664 | safer |
| equity_to_assets | 0.0739 | 0.2624 | riskier |
| cre_to_capital | 0.1338 | 0.1974 | riskier |
| share_nonfarm_nonres | 0.0235 | 0.1720 | riskier |
| share_multifamily | 0.0085 | 0.1106 | riskier |
| total_rbc_ratio | 16.0507 | -0.1035 | safer |
| tier1_leverage | 7.9626 | 0.1027 | riskier |
| asset_growth_12q | 1.0948 | 0.0703 | riskier |
| tangible_equity_to_assets | 0.0727 | 0.0575 | riskier |

## Drivers, rate_aware / logit (baseline log-odds -8.304, contributions sum -0.092)

| feature | value | contribution | direction |
|---|---|---|---|
| macro_fedfunds_change_4q | 4.4900 | -0.6408 | safer |
| uninsured_share | 0.8644 | -0.4610 | safer |
| unrealized_loss_to_tier1 | -1.0406 | 0.4038 | riskier |
| cre_to_capital | 0.1338 | -0.3874 | safer |
| afs_unrealized_to_tier1 | -0.1486 | 0.3725 | riskier |
| macro_hpi_change_4q | 0.1312 | -0.3063 | safer |
| securities_to_assets | 0.5612 | -0.2936 | safer |
| share_nonfarm_nonres | 0.0235 | 0.2473 | riskier |
| d4q_unrealized_loss_to_tier1 | -0.9639 | 0.2050 | riskier |
| adjusted_tier1_leverage | -0.3301 | 0.1957 | riskier |

## Drivers, rate_aware / gbdt (baseline log-odds -9.794, contributions sum 1.639)

| feature | value | contribution | direction |
|---|---|---|---|
| adjusted_tier1_leverage | -0.3301 | 2.0801 | riskier |
| macro_fedfunds_change_4q | 4.4900 | -0.2821 | safer |
| d4q_unrealized_loss_to_tier1 | -0.9639 | 0.2766 | riskier |
| unrealized_loss_to_tier1 | -1.0406 | 0.2583 | riskier |
| securities_to_assets | 0.5612 | -0.1851 | safer |
| macro_hpi_change_4q | 0.1312 | -0.1592 | safer |
| share_residential | 0.1226 | 0.1397 | riskier |
| nim_q | 0.0210 | -0.1258 | safer |
| large_time_deposit_share | 0.0080 | -0.1111 | safer |
| equity_to_assets | 0.0739 | 0.1000 | riskier |
