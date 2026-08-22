# SQuAD LoRA Fixed-Variance Online LiRA

## Controlled Setting

This experiment reuses the completed one-epoch, five-shadow SQuAD LoRA target adapters, candidate texts, evaluator labels, and five-column inclusion masks. Both model families retain the original data revisions, seeds, LoRA configuration, optimizer, scheduler, precision, batch size, and gradient accumulation settings.

The five shadow adapters are retrained from the inherited masks. Shadow filler records follow the current safety rule and have zero overlap with target training records. Each shadow therefore replaces 36--41 historical filler records; comparisons with the historical RMIA results are descriptive and cannot be attributed strictly to the attack algorithm.

## Attack Definition

The observation is normalized completion mean log-probability. For every candidate, LiRA estimates separate IN and OUT means. IN and OUT variances are pooled across candidates using the unbiased within-candidate estimator. The score is the IN Gaussian log-likelihood minus the OUT Gaussian log-likelihood. Missing IN/OUT observations, nonpositive variance, insufficient global degrees of freedom, or non-finite values fail the run without numerical smoothing.

Metrics comprise ROC-AUC and TPR at 1% and 5% FPR against each non-member comparison group, with 2,000 bootstrap replicates for 95% confidence intervals. No RMIA population calibration is computed.
