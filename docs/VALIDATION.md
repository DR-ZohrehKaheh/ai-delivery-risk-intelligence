# Portfolio audit — 17 October 2026

The current Python source, notebook code cells, citation metadata, documentation links and publication contents were reviewed. This audit does not establish production readiness.

## Executed checks

- A fresh Python 3.12 environment installed the pinned `requirements.txt` dependencies successfully.
- All **six dataset-independent tests** passed, including a new regression test for spreadsheet-formula safety in downloaded batch CSVs.
- The training script completed on separately obtained original Olist orders, customers and order items.
- All code cells in the four notebooks executed sequentially, in an isolated namespace per notebook, including the 1,000-resample bootstrap calculation. This checks cell execution, not Jupyter's interactive frontend.
- Streamlit's application test harness loaded the regenerated model and completed a single-order assessment without exceptions. Browser-based batch-upload interaction was not retested in this audit.
- Original CSVs, regenerated model binaries and local execution logs were excluded from the publication changes.

## Reproduction results and limits

The current run used 30,621 training observations and 20,623 validation observations. XGBoost validation ROC-AUC was **0.6362**, average precision **0.1992** and Brier score **0.1171**. The notebook's retrospective-test ROC-AUC was **0.6903**.

These results are close to, but do not exactly reproduce, the earlier development tables in the README and archived release. Those tables remain labelled as historical development results. The cause of the differences has not been isolated; this audit does not claim bit-for-bit metric reproducibility or replace the archived results. Dataset selection, environment and numerical behaviour should be compared before drawing conclusions about the differences.

## Export correction

Downloaded batch CSVs now escape formula-like text values and column headers, including extra text columns supplied by a user. Numeric scores are unchanged. This reduces spreadsheet formula interpretation when opening downloaded results. The regression test checks escaping, unchanged numerical values and preservation of the input frame.

## Continuous integration and archive

[GitHub Actions](https://github.com/DR-ZohrehKaheh/ai-delivery-risk-intelligence/actions/workflows/tests.yml) runs the dataset-independent tests using current official Node 24 actions. It does not download Olist data or run the historical training/notebook experiment.

The existing release tag and Zenodo DOI continue to identify their original archived snapshot. This correction is on `main`; cite its commit when referring to the revised CSV export.
