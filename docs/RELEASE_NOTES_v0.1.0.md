# Release notes: v0.1.0 — Research portfolio prototype

## Overview

First public source-code release of **AI Delivery Risk Intelligence**, a reproducible research prototype for modeling historical e-commerce delivery risk and prioritizing orders for review.

## Included

- XGBoost training pipeline with chronological cohort selection and purchase-time feature engineering
- Logistic regression, random forest and dummy baseline comparisons in exploratory notebooks
- Independent calibration, retrospective evaluation and bootstrap confidence-interval analyses
- Streamlit dashboard for single-order and CSV batch risk scoring
- Configurable review capacity, risk-score histogram and export of ranked orders
- Input validation, dataset-independent automated unit tests and GitHub Actions CI
- Machine-readable `CITATION.cff` and dashboard screenshots

## Research status and limitations

**Not validated for real-world operational deployment.** The target concerns confirmed delivered orders only; exclusion of missing/unconfirmed outcomes may introduce selection bias. Absolute scores are not consistently calibrated across historical periods.

Retrospective holdout development results recorded for the original XGBoost model: ROC-AUC 0.6917; average precision 0.0854; Brier score 0.0449; precision@10% 0.0871; lift@10% 1.8132. The holdout was inspected during development and is not an independent prospective validation.

Results may differ slightly with package versions or execution environment. The historical notebook documents calibration instability and limitations of simple order-level Bootstrap intervals.

## Reproduction

The repository deliberately omits original Olist CSV files and serialized `.joblib` models. Obtain the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) from its source (CC BY-NC-SA 4.0) and follow [README.md](../README.md) to train and run the demo.

## Citation and archive

Published as [GitHub release v0.1.0](https://github.com/DR-ZohrehKaheh/ai-delivery-risk-intelligence/releases/tag/v0.1.0) and archived in [Zenodo](https://zenodo.org/records/23324488).

- Version DOI: https://doi.org/10.5281/zenodo.23324488
- Concept DOI: https://doi.org/10.5281/zenodo.23324487
- Citation files on the default branch: [CITATION.cff](../CITATION.cff) and [CITATION.bib](../CITATION.bib)
- Archive provenance and future-version instructions: [ARCHIVING.md](ARCHIVING.md)

## Rights

Copyright © 2026 Zohreh Kaheh. All rights reserved. Public availability is for technical review and scholarly citation; no general software reuse licence is granted. See [RIGHTS.md](../RIGHTS.md). Raw Olist datasets and generated model binaries are excluded.
