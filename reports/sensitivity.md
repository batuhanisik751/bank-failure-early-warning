# Sensitivity analyses

Fixed out-of-time split (train reports 2002-03-31..2008-12-31, rule 6.2 trimmed; test 2010-03-31..2013-12-31). Models: `logit` (P1 regularised logit on `features_v2`, `LOGIT_C`) and `gbdt` (`settings.models.gbdt`). Every cell is one refit on the split with a single assumption changed; the records live in `runs/sensitivity/`. Metrics are ranking metrics on the test rows (PR-AUC, ROC-AUC, recall in the top 2 percent and top 100) plus the Brier score of the probability. The unchanged cell of every analysis (4q, kept, 60d) is the same fit as the model's fixed-split `train` run, so the tables read against `reports/p2_gbdt.md`; the 8q training set is trimmed a year earlier by rule 6.2 and a lag change moves both the outcome windows and the test positives, so `n_failures` differs between cells.

## Horizon: 4 against 8 quarters

| model | variant | n_train | positives_train | train_repdte_max | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | brier | n | n_failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logit | 4q | 252330 | 554 | 2008-12-31 | 0.4437 | 0.9823 | 0.7815 | 0.0840 | 0.0051 | 118696 | 833 |
| logit | 8q | 218358 | 660 | 2007-12-31 | 0.3764 | 0.9658 | 0.5846 | 0.0471 | 0.0084 | 118696 | 1295 |
| gbdt | 4q | 252330 | 554 | 2008-12-31 | 0.4324 | 0.9840 | 0.7791 | 0.0732 | 0.0052 | 118696 | 833 |
| gbdt | 8q | 218358 | 660 | 2007-12-31 | 0.1328 | 0.9029 | 0.2911 | 0.0263 | 0.0105 | 118696 | 1295 |

What moves: gbdt recall at 2% goes from 0.7791 (4q) to 0.2911 (8q, -0.4880), the largest shift in the table.
What does not: logit is the less sensitive model (largest shift 0.1970), and the PR-AUC ranking of the models is the same under every variant.

## Censored rows: kept against dropped from training and test

| model | variant | n_train | positives_train | train_repdte_max | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | brier | n | n_failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logit | kept | 252330 | 554 | 2008-12-31 | 0.4437 | 0.9823 | 0.7815 | 0.0840 | 0.0051 | 118696 | 833 |
| logit | dropped | 243793 | 554 | 2008-12-31 | 0.4559 | 0.9826 | 0.7839 | 0.0876 | 0.0051 | 114819 | 833 |
| gbdt | kept | 252330 | 554 | 2008-12-31 | 0.4324 | 0.9840 | 0.7791 | 0.0732 | 0.0052 | 118696 | 833 |
| gbdt | dropped | 243793 | 554 | 2008-12-31 | 0.4467 | 0.9857 | 0.8055 | 0.0756 | 0.0053 | 114819 | 833 |

What moves: gbdt recall at 2% goes from 0.7791 (kept) to 0.8055 (dropped, +0.0264), the largest shift in the table.
What does not: logit is the less sensitive model (largest shift 0.0122), and the PR-AUC ranking of the models is the same under every variant.

## Availability lag: 45 against 60 against 90 days

| model | variant | n_train | positives_train | train_repdte_max | pr_auc | roc_auc | recall_at_2pct | recall_at_top100 | brier | n | n_failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logit | 45d | 252339 | 537 | 2008-12-31 | 0.4692 | 0.9831 | 0.7944 | 0.0870 | 0.0050 | 118737 | 851 |
| logit | 60d | 252330 | 554 | 2008-12-31 | 0.4437 | 0.9823 | 0.7815 | 0.0840 | 0.0051 | 118696 | 833 |
| logit | 90d | 252311 | 606 | 2008-12-31 | 0.4273 | 0.9817 | 0.7787 | 0.0816 | 0.0050 | 118634 | 809 |
| gbdt | 45d | 252339 | 537 | 2008-12-31 | 0.4494 | 0.9841 | 0.7697 | 0.0799 | 0.0055 | 118737 | 851 |
| gbdt | 60d | 252330 | 554 | 2008-12-31 | 0.4324 | 0.9840 | 0.7791 | 0.0732 | 0.0052 | 118696 | 833 |
| gbdt | 90d | 252311 | 606 | 2008-12-31 | 0.3613 | 0.9796 | 0.7244 | 0.0717 | 0.0054 | 118634 | 809 |

What moves: gbdt PR-AUC goes from 0.4324 (60d) to 0.3613 (90d, -0.0711), the largest shift in the table.
What does not: logit is the less sensitive model (largest shift 0.0255), and the PR-AUC ranking of the models is the same under every variant.
