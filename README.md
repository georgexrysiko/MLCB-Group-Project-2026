# Predictive and Mechanistic Signatures of Immune Checkpoint Inhibitor Response from Single-Cell RNA-Sequencing of Melanoma

**Machine Learning in Computational Biology - Term Project**

A re-analysis of the Sade-Feldman et al. melanoma immunotherapy single-cell cohort (GEO accession **GSE120575**), building an end-to-end, leakage-controlled pipeline from raw single-cell counts to predictive modelling, hyperparameter tuning, and model explainability, benchmarked against the independently published PRECISE study (Pinhasi & Yizhak, 2025).

---

## 1. Overview

Response to immune checkpoint inhibitor (ICI) therapy is difficult to predict from pre- or on-treatment biopsies. This project re-analyses 16,291 CD45+ immune cells from 48 biopsies (32 melanoma patients) to answer three questions:

1. **Which biological pathways and cell-to-cell communication axes differ between responders and non-responders?**
2. **Can a compact, biologically summarised feature set (PROGENy pathway scores) predict response as well as - or better than - raw gene expression, under strict patient-level cross-validation?**
3. **What do the resulting models actually learn**, and how does that compare with an independently published gene-level signature for the same dataset?

All modelling uses **Leave-One-Patient-Out (LOPO)** cross-validation - every sample and cell belonging to a held-out patient is excluded from training in that fold - to avoid patient-level leakage, which is a common and serious pitfall with this kind of repeated-measures single-cell cohort.

---

## 2. Repository / Notebook Structure

Run the notebooks **in this order**. Each stage's output feeds the next.

| Stage | Notebook | Purpose |
|---|---|---|
| **A** | `TaskA_EDA_Preprocessing.ipynb` | Merge GEO expression matrix + metadata, QC filtering (mitochondrial/ribosomal/non-coding gene removal, ≥3%-of-cells expression filter), sparsity/PCA/UMAP exploration. **Output:** `filtered_anndata.h5ad` |
| **B.1** | `TaskB_1_Pathway_Scoring_add.ipynb` | Per-cell PROGENy pathway-activity scoring (14 pathways) via `decoupler.mt.mlm`. QC, visualization, correlation diagnostics. **Output:** `filtered_anndata_pathways.h5ad` |
| **B.2** | `TaskB_2_Differential_Pathway_Analysis.ipynb` | Patient-aware **mixed-effects model** (pathway ~ response + timepoint + (1\|patient)) testing which pathways differ significantly between Responders and Non-responders, plus Mann-Whitney sensitivity checks at sample- and patient-level. |
| **B.3** | `TaskB_3_CCC_Scoring.ipynb` | Compact, immune/TME-scoped **ligand-receptor co-expression** feature panel (CellPhoneDB-derived, response-independent selection), sample-level and cell-conditioned representations. |
| **B.4** | `TaskB_4_Models_run.ipynb` | Main **LOPO ablation**: 9 feature representations (genes, pathways, PCA, MI-selected genes, RFE, and their combinations with pathways) × 3 classifiers (Logistic Regression, XGBoost, Random Forest), fixed hyperparameters. Patient-cluster bootstrap for uncertainty. |
| **B.5** | `TaskB_5_SHAP___ErrorAnalysis.ipynb` | SHAP explainability + error analysis for the best pathway and best gene pipelines from Task B.4 (no-tuning models). |
| **C** | `TaskC_Tuning_*.ipynb` | Per-fold **Optuna hyperparameter tuning** (inner patient-grouped CV) for pathway and gene feature sets. |
| **D** | `TaskD_Explainability_*.ipynb` | SHAP explainability + error analysis for the **tuned** best pipelines from Task C. |
| shared | `ablation_shared.py` | Single source of truth for the LOPO framework (config, feature builders, `run_condition_all_models`, checkpointing) - imported by both the Task B.4 baseline and any downstream ablation extension, to guarantee identical methodology across notebooks. |

> **Why a shared module?** Task B.4's ablation logic (feature builders, model parameters, the LOPO loop itself) must be byte-for-byte identical wherever it is reused (e.g. for a future CCC ablation or a diagnostic test), otherwise any AUC difference between conditions cannot be cleanly attributed to the feature representation. `ablation_shared.py` exists specifically to remove that risk.

---

## 3. Data and Outputs

```
data/
  external/                        cached external resources (CellPhoneDB, DA-seq G1-G11 file)
outputs/
  filtered_anndata.h5ad                            Task A output
  filtered_anndata_pathways.h5ad                   Task B.1 output (Task A + 14 PROGENy_* obs columns)
  differential_pathway_analysis/                   Task B.2 statistical tables
  filtered_anndata_pathways_cellSpecificCCC.h5ad   Task B.3 
  task_b_lopo_no_tuning/                      Task B.4 checkpoints (metrics.json, predictions.csv per condition/model)
  task_b5_explainability/                     Task B.5 SHAP tables and fold-artifact exports
figures/
  taska_all_figures/, taskb_2_all_figures/, ... one consolidated subfolder per task
```

All expensive steps (LOPO folds, bootstrap draws) are **checkpointed to disk** keyed by a fixed `RUN_ID`; re-running a notebook re-uses existing results instead of recomputing them, as long as `RUN_ID` and the results directory are left unchanged.

---

## 4. Key Findings

- **JAK-STAT pathway activity is the single strongest and most consistently replicated biological correlate of non-response**, confirmed by (i) a patient-aware mixed-effects model, (ii) two independent Mann-Whitney sensitivity analyses, and (iii) SHAP explainability on an independently trained predictive model. EGFR and PI3K (higher in non-responders) and MAPK (higher in responders) are secondary, equally robust findings.
- A compact **14-feature PROGENy pathway representation** matched the original PRECISE paper's best full-transcriptome result (AUC 0.894 vs. 0.89) using ~700-fold fewer features and a stricter, patient-grouped validation scheme.
- The **cell-cell communication (CCC)** feature family is biologically motivated and response-independent by construction, but underperformed both genes and pathways (best AUC 0.801) in its current per-sample co-expression form; SHAP indicates real signal (chemokine- and HLA-mediated axes) that a more pathway-like aggregation may better exploit.
- The JAK-STAT/interferon-response axis identified here **converges with an independently published Boruta-confirmed 11-gene signature** for the same dataset (Pinhasi & Yizhak, 2025), strengthening confidence that this is a genuine, dataset-independent signal.

See the accompanying project report for full methodology, statistics, and figures.

---

## 5. Requirements

```
python >= 3.10
scanpy, anndata, decoupler
scikit-learn, xgboost, lightgbm
statsmodels, scipy
shap
pandas, numpy, matplotlib, seaborn
optuna          # Task C tuning only
```

Install with:
```bash
pip install scanpy anndata decoupler scikit-learn xgboost lightgbm statsmodels scipy shap pandas numpy matplotlib seaborn optuna
```

External reference data (CellPhoneDB interaction/gene tables) are fetched automatically on first run from public GitHub repositories and cached under `data/external/`; no manual download is required.

---

## 6. Reproducibility Notes

- A single `SEED = 42` is used throughout (NumPy, scikit-learn, Optuna).
- All file paths are relative (`../data`, `../outputs`, `../figures`); no absolute/user-specific paths.
- Every LOPO fold includes an explicit patient/sample-overlap leakage check that raises an error if violated.
- Adaptive preprocessing (PCA, mutual-information ranking, RFE, scaling) is refit **inside every outer fold**, on training patients only.
- `response == 1` is explicitly verified to correspond to `response_label == "Responder"` before being used as the positive class anywhere (Sensitivity/Specificity, SHAP direction).

---

## 7. Known Limitations

- Small cohort (48 samples / 32 patients): bootstrap 95% CIs on AUC are wide (±0.1–0.15) and should be read as indicative, not tightly conclusive.
- No external validation cohort; generalisability beyond this dataset is untested.
- The CCC feature family is not yet combined with genes+pathways in a single ablation condition.
- Gene-level SHAP findings depend on the feature-selection method (Mutual Information here vs. Boruta in the original PRECISE study) and should be treated as exploratory; the pathway-level findings are corroborated across three independent statistical procedures and are considered the primary result.

---

## 8. Authors

- Chrysikopoulos Georgios
- Dimitrakou Paraskevi
- Karalexi Maria-Evangelia
## Main References
Sade-Feldman et al., *Cell* (2018) [dataset]; Pinhasi & Yizhak, *npj Precision Oncology* (2025) [PRECISE, benchmark comparison];
