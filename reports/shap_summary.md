# SHAP driver summary (walk-forward gradient boosters)

`shap.TreeExplainer` on the per-year `gbdt_mono` walk-forward models (2008-2024, each explaining its own test rows) and on the production model (the 2024 booster scoring every quarter after 2024). Values are log-odds contributions; `mean_abs_shap` is the mean |SHAP| over a year's rows, pooled as the plain average over test years, and `share` is the per-year normalised importance: each year's mean |SHAP| divided by that year's total, then averaged over years, so a booster on a larger log-odds scale (the four-failure 2020 fit) counts as one year like every other. The `drivers` table keeps the five largest positive and five largest negative contributions per bank-quarter.

## Mean |SHAP| per feature, pooled and 2009 versus 2023 (top 20)

| feature | mean_abs_shap | share | mean_abs_2009 | mean_abs_2023 |
|---|---|---|---|---|
| texas_ratio | 0.2028 | 0.0696 | 0.3425 | 0.1751 |
| macro_unemp_change_4q | 0.2099 | 0.0580 | 0.1925 | 0.1009 |
| macro_hpi_change_4q | 0.2472 | 0.0551 | 1.0587 | 0.0729 |
| securities_to_assets | 0.1701 | 0.0544 | 0.2988 | 0.1588 |
| total_rbc_ratio | 0.1493 | 0.0517 | 0.0030 | 0.1830 |
| macro_dgs10 | 0.1572 | 0.0440 | 0.1783 | 0.1747 |
| adjusted_tier1_leverage | 0.1629 | 0.0407 | 0.3030 | 0.1520 |
| neg_roa_quarters_last_8 | 0.1193 | 0.0394 | 0.0430 | 0.1560 |
| equity_to_assets | 0.1073 | 0.0328 | 0.0684 | 0.0968 |
| d4q_texas_ratio | 0.1033 | 0.0303 | 0.0318 | 0.0742 |
| share_consumer | 0.1589 | 0.0280 | 0.0259 | 0.0624 |
| log_assets | 0.1016 | 0.0280 | 0.0774 | 0.0774 |
| roa_q | 0.0744 | 0.0267 | 0.1066 | 0.0722 |
| early_delinquency | 0.0849 | 0.0254 | 0.0459 | 0.0535 |
| large_time_deposit_share | 0.0845 | 0.0242 | 0.0064 | 0.1056 |
| d4q_equity_to_assets | 0.0652 | 0.0219 | 0.0178 | 0.0586 |
| share_agri | 0.0790 | 0.0187 | 0.0837 | 0.0600 |
| share_residential | 0.0906 | 0.0185 | 0.0514 | 0.0451 |
| brokered_share | 0.0562 | 0.0180 | 0.0988 | 0.0416 |
| bkclass_NM | 0.0659 | 0.0165 | 0.0557 | 0.0285 |

Shift in mean |SHAP| from 2009 to 2023: rose most for `total_rbc_ratio` (+0.180), `neg_roa_quarters_last_8` (+0.113), `large_time_deposit_share` (+0.099); fell most for `macro_hpi_change_4q` (-0.986), `asset_growth_12q` (-0.186), `texas_ratio` (-0.167). The crisis-year booster leans on credit quality and the housing cycle, the recent one on rate sensitivity (unrealised losses, the funds-rate change) and capital.

## Feature-importance smoke test (spec rule 6.6)

Largest per-year normalised share: `texas_ratio` at 7.0%; the top three together carry 18.3%. Threshold 40%: no feature is flagged; importance is spread across capital, asset-quality, earnings and sensitivity ratios, with no single field acting as a failure marker.

## Per-year top feature

| year | top_feature | top_share | second_feature | third_feature |
|---|---|---|---|---|
| 2008 | macro_hpi_change_4q | 0.1569 | log_assets | asset_growth_12q |
| 2009 | macro_hpi_change_4q | 0.1957 | texas_ratio | adjusted_tier1_leverage |
| 2010 | macro_hpi_change_4q | 0.1429 | texas_ratio | adjusted_tier1_leverage |
| 2011 | macro_hpi_change_4q | 0.0879 | macro_unemp_change_4q | adjusted_tier1_leverage |
| 2012 | texas_ratio | 0.0955 | macro_hpi_change_4q | macro_dgs10 |
| 2013 | texas_ratio | 0.1108 | securities_to_assets | total_rbc_ratio |
| 2014 | texas_ratio | 0.0833 | total_rbc_ratio | macro_dgs10 |
| 2015 | texas_ratio | 0.1132 | total_rbc_ratio | securities_to_assets |
| 2016 | texas_ratio | 0.0857 | total_rbc_ratio | macro_unemp_change_4q |
| 2017 | total_rbc_ratio | 0.0751 | texas_ratio | macro_unemp_change_4q |
| 2018 | texas_ratio | 0.0828 | total_rbc_ratio | macro_unemp_change_4q |
| 2019 | texas_ratio | 0.0764 | total_rbc_ratio | securities_to_assets |
| 2020 | macro_unemp_change_4q | 0.1040 | securities_to_assets | texas_ratio |
| 2021 | macro_unemp_change_4q | 0.0927 | securities_to_assets | texas_ratio |
| 2022 | total_rbc_ratio | 0.0749 | adjusted_tier1_leverage | macro_unemp_change_4q |
| 2023 | total_rbc_ratio | 0.0655 | texas_ratio | macro_dgs10 |
| 2024 | share_consumer | 0.1498 | nim_q | share_residential |
| production | share_consumer | 0.1684 | nim_q | share_residential |

![SHAP beeswarm, latest booster](figures/shap_summary_latest.png)
