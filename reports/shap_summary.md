# SHAP driver summary (walk-forward gradient boosters)

`shap.TreeExplainer` on the per-year `gbdt` walk-forward models (2008-2024, each explaining its own test rows) and on the production model (the 2024 booster scoring every quarter after 2024). Values are log-odds contributions; `mean_abs_shap` is the mean |SHAP| over a year's rows, pooled as the plain average over test years. The `drivers` table keeps the five largest positive and five largest negative contributions per bank-quarter.

## Mean |SHAP| per feature, pooled and 2009 versus 2023 (top 20)

| feature | mean_abs_shap | share | mean_abs_2009 | mean_abs_2023 |
|---|---|---|---|---|
| large_time_deposit_share | 3.1621 | 0.2409 | 0.0311 | 0.0211 |
| unrealized_loss_to_tier1 | 2.7087 | 0.2063 | 0.0280 | 0.1071 |
| macro_dgs10 | 0.9435 | 0.0719 | 0.0201 | 0.0225 |
| nco_rate | 0.7826 | 0.0596 | 0.0177 | 0.0013 |
| d1q_unrealized_loss_to_tier1 | 0.6753 | 0.0514 | 0.0134 | 0.0088 |
| share_nonfarm_nonres | 0.5644 | 0.0430 | 0.0916 | 0.0453 |
| texas_ratio | 0.4758 | 0.0362 | 0.3836 | 0.1081 |
| macro_hpi_change_4q | 0.4152 | 0.0316 | 0.4155 | 0.0433 |
| securities_to_assets | 0.2129 | 0.0162 | 0.1566 | 0.1152 |
| adjusted_tier1_leverage | 0.2043 | 0.0156 | 0.0947 | 0.2262 |
| afs_unrealized_to_tier1 | 0.1752 | 0.0133 | 0.0415 | 0.1969 |
| macro_fedfunds_change_4q | 0.1426 | 0.0109 | 0.1896 | 0.3606 |
| macro_unemp_change_4q | 0.1206 | 0.0092 | 0.0245 | 0.0618 |
| neg_roa_quarters_last_8 | 0.1206 | 0.0092 | 0.0272 | 0.1402 |
| equity_to_assets | 0.1011 | 0.0077 | 0.0494 | 0.0845 |
| uninsured_share | 0.1003 | 0.0076 | 0.0353 | 0.0092 |
| total_rbc_ratio | 0.0966 | 0.0074 | 0.0248 | 0.0747 |
| d4q_texas_ratio | 0.0904 | 0.0069 | 0.0379 | 0.0485 |
| log_assets | 0.0896 | 0.0068 | 0.0734 | 0.0443 |
| share_consumer | 0.0773 | 0.0059 | 0.0205 | 0.0206 |

Shift in mean |SHAP| from 2009 to 2023: rose most for `macro_fedfunds_change_4q` (+0.171), `afs_unrealized_to_tier1` (+0.155), `adjusted_tier1_leverage` (+0.131); fell most for `macro_hpi_change_4q` (-0.372), `texas_ratio` (-0.276), `npa_to_assets` (-0.084). The crisis-year booster leans on credit quality and the housing cycle, the recent one on rate sensitivity (unrealised losses, the funds-rate change) and capital.

## Feature-importance smoke test (spec rule 6.6)

Largest share of the total mean |SHAP|: `large_time_deposit_share` at 24.1%; the top three together carry 51.9%. Threshold 40%: no feature is flagged; importance is spread across capital, asset-quality, earnings and sensitivity ratios, with no single field acting as a failure marker.

## Per-year top feature

| year | top_feature | top_share | second_feature | third_feature |
|---|---|---|---|---|
| 2008 | share_construction | 0.1477 | share_nonfarm_nonres | macro_hpi_change_4q |
| 2009 | macro_hpi_change_4q | 0.1156 | texas_ratio | macro_fedfunds_change_4q |
| 2010 | macro_hpi_change_4q | 0.0877 | adjusted_tier1_leverage | securities_to_assets |
| 2011 | texas_ratio | 0.1212 | macro_hpi_change_4q | npa_to_assets |
| 2012 | texas_ratio | 0.0895 | npa_to_assets | adjusted_tier1_leverage |
| 2013 | texas_ratio | 0.0908 | securities_to_assets | neg_roa_quarters_last_8 |
| 2014 | texas_ratio | 0.0919 | securities_to_assets | equity_to_assets |
| 2015 | texas_ratio | 0.0868 | neg_roa_quarters_last_8 | total_rbc_ratio |
| 2016 | texas_ratio | 0.1086 | equity_to_assets | neg_roa_quarters_last_8 |
| 2017 | texas_ratio | 0.1036 | neg_roa_quarters_last_8 | macro_fedfunds_change_4q |
| 2018 | neg_roa_quarters_last_8 | 0.0870 | texas_ratio | securities_to_assets |
| 2019 | texas_ratio | 0.0910 | securities_to_assets | neg_roa_quarters_last_8 |
| 2020 | large_time_deposit_share | 0.2997 | unrealized_loss_to_tier1 | macro_dgs10 |
| 2021 | texas_ratio | 0.0933 | neg_roa_quarters_last_8 | securities_to_assets |
| 2022 | macro_fedfunds_change_4q | 0.1378 | adjusted_tier1_leverage | d4q_unrealized_loss_to_tier1 |
| 2023 | macro_fedfunds_change_4q | 0.1477 | adjusted_tier1_leverage | afs_unrealized_to_tier1 |
| 2024 | adjusted_tier1_leverage | 0.1006 | unrealized_loss_to_tier1 | neg_roa_quarters_last_8 |
| production | adjusted_tier1_leverage | 0.0851 | neg_roa_quarters_last_8 | unrealized_loss_to_tier1 |

![SHAP beeswarm, latest booster](figures/shap_summary_latest.png)
