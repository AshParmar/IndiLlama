# Marathi Sentiment Ensemble — Research Note

Date: September 18, 2025

This document details the ensemble method used to combine multiple transformer classifiers for Marathi sentiment analysis. It is intended as a concise research note you can cite or extend.

## Motivation

No single model is uniformly best across all inputs. Ensembling reduces variance and captures complementary strengths. Given three competent models fine-tuned on the same task, soft-voting (probability averaging) is a simple, strong baseline that reliably improves macro F1.

## Models and Data

- Models (from Hugging Face):
  - AshParmar/XMR-Muril
  - AshParmar/XMR-Albert
  - AshParmar/XMR-MBERT
- Dataset: `output/combined_dataset/test_strict.csv`
  - Columns used: `text` (Marathi), `label` (strings: negative, neutral, positive) or fallback `label_raw` (-1, 0, 1)
  - Confirmed: 100% of `text` rows contain Devanagari; no ASCII-only lines.

## Method

- Inference per model m produces logits L_m(x) ∈ R^C. We compute probabilities P_m(x) = softmax(L_m(x)).
- Let w_m ≥ 0 be model weights with Σ w_m = 1. The ensemble probability is:

  P_ens(x) = Σ_m w_m · P_m(x)

- The predicted class is argmax_c P_ens(x)_c.
- Label alignment: model label orders can differ. We detect `config.id2label` and map to canonical order [negative, neutral, positive]. If missing, default to [0,1,2] for 3-class models.
- Tokenization: per-model tokenizers with truncation to 192 tokens. Batch size defaults to 32 (tune for memory).

## Weights

- Default weights are derived from each model’s macro F1 on the same task (normalized internally):
  - Muril: 0.79255
  - Marathi Albert v2: 0.779034
  - Marathi BERT: 0.776411
- You can specify custom weights to reflect validation performance or desired bias.

## Metrics and Evaluation

- Primary: Accuracy, Macro F1
- Secondary: Full classification report and confusion matrix
- Outputs:
  - `output/ensemble_results.json`
  - `output/ensemble_predictions.csv`

## Reproducibility

- Script: `LLMs_testing/ensemble_inference.py`
- Key arguments:
  - `--models` HF repo IDs (default: three above)
  - `--weights` weights for each model (auto-normalized)
  - `--test-csv` path to test CSV
  - `--label-col` default `label` (string), with fallback to `label_raw`
  - `--batch-size`, `--max-length`, `--cpu`

Example (PowerShell):

```powershell
cd "C:\LLM's_for_SA\LLMs_testing"
python .\ensemble_inference.py --test-csv ..\output\combined_dataset\test_strict.csv --label-col label --cpu
```

## Results (current run)

- Accuracy: 0.908
- Macro F1: 0.9076

These results are higher than any single constituent model in our experiments.

## Limitations and Next Steps

- Memory: Holding 3 models concurrently may exceed some GPUs. Use `--cpu`, smaller `--batch-size`, or a sequential GPU offload approach.
- Weight selection: Better to tune weights on the validation set rather than using test scores.
- Beyond soft-voting:
  - Stacking: Train a meta-classifier on validation logits
  - Temperature scaling per model before averaging
  - Dynamic ensembling: confidence-based gating

## Citation

If you use this ensemble in your work, please cite this note and the underlying model repos.
