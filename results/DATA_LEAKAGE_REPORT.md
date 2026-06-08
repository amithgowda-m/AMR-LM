# Data Leakage Report

**Generated**: 2026-06-08T14:55:07.431168
**Method**: K-mer overlap analysis (k=31, threshold=90%)

## Summary

| Database | Sequences Before | Removed | Remaining | % Removed |
|----------|-----------------|---------|-----------|-----------|
| MEGARes  | 8729 | 4152 | 4577 | 47.6% |
| SARG     | 7315 | 0 | 7315 | 0.0% |

## Methods (copy-paste for paper)

> To prevent data leakage between training and evaluation sets, we performed
> pairwise k-mer overlap analysis (k=31) between all CARD training sequences
> and both external test databases. Sequences from MEGARes and SARG with
> greater than 90% k-mer overlap to any training sequence were removed.
> This resulted in the removal of 4152 sequences from MEGARes
> (47.6%) and 0 sequences from SARG (0.0%).
