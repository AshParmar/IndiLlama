import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


CANONICAL_LABELS = ["negative", "neutral", "positive"]
CANONICAL_TO_ID = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}


def resolve_label_mapping(model_id: str, num_labels: int, id2label: Dict[int, str]) -> List[int]:
    """
    Build an index mapping from model's label order to canonical order.

    Returns a list idx_map where probs_model[:, j] should be added to
    probs_ensemble[:, idx_map[j]]. If an exact name match isn't found,
    best-effort fallbacks are used for typical 3-class sentiment.
    """
    if id2label:
        # Normalize names (lowercase, strip)
        norm = {i: id2label[i].lower().strip() for i in id2label}
        # Try direct name-based mapping
        name_to_idx = {v: k for k, v in norm.items()}
        if all(lbl in name_to_idx for lbl in CANONICAL_LABELS):
            return [name_to_idx[lbl] for lbl in CANONICAL_LABELS]

        # Common alternative spellings or tags
        aliases = {
            "neg": "negative",
            "pos": "positive",
            "neu": "neutral",
        }
        rev = {}
        for i, name in norm.items():
            key = aliases.get(name, name)
            rev[key] = i
        if all(lbl in rev for lbl in CANONICAL_LABELS):
            return [rev[lbl] for lbl in CANONICAL_LABELS]

    # Fallback: assume label indices 0,1,2 map to negative, neutral, positive
    if num_labels == 3:
        return [0, 1, 2]

    raise ValueError(
        f"Model {model_id} has {num_labels} labels and no recognizable id2label mapping; cannot ensemble."
    )


def load_models(model_repos: List[str], device: torch.device) -> Tuple[List[AutoTokenizer], List[AutoModelForSequenceClassification], List[List[int]]]:
    tokenizers = []
    models = []
    reorder_indices = []
    for repo in model_repos:
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForSequenceClassification.from_pretrained(repo)
            mdl.to(device)
            mdl.eval()
            num_labels = mdl.config.num_labels
            id2label = None
            # id2label may be list or dict
            if hasattr(mdl.config, "id2label") and mdl.config.id2label:
                if isinstance(mdl.config.id2label, dict):
                    id2label = {int(k): v for k, v in mdl.config.id2label.items()}
                elif isinstance(mdl.config.id2label, list):
                    id2label = {i: v for i, v in enumerate(mdl.config.id2label)}
            idx_map = resolve_label_mapping(repo, num_labels, id2label or {})

            tokenizers.append(tok)
            models.append(mdl)
            reorder_indices.append(idx_map)
            print(f"Loaded {repo} (num_labels={num_labels})")
        except Exception as e:
            print(f"Warning: failed to load {repo}: {e}")

    if not models:
        raise RuntimeError("No models could be loaded. Check repo IDs and internet connectivity.")

    return tokenizers, models, reorder_indices


def batch_iter(texts: List[str], batch_size: int):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)


def run_ensemble(
    model_repos: List[str],
    weights: List[float],
    test_csv: Path,
    text_col: str = "text",
    label_col: str = "label",
    batch_size: int = 32,
    max_length: int = 192,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    assert len(model_repos) == len(weights), "Number of weights must match number of models"
    # Normalize weights
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()

    df = pd.read_csv(test_csv)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {test_csv}")

    # Build y_true using preferred label column
    if label_col in df.columns:
        # Map string labels to canonical ids
        def label_to_id(x: str) -> int:
            key = str(x).strip().lower()
            if key not in CANONICAL_TO_ID:
                raise ValueError(f"Unknown label '{x}' in column '{label_col}'. Expected one of {CANONICAL_LABELS}")
            return CANONICAL_TO_ID[key]

        y_true = df[label_col].map(label_to_id).tolist()
    elif "label_raw" in df.columns:
        raw_to_canonical = {-1: 0, 0: 1, 1: 2}
        y_true = df["label_raw"].map(raw_to_canonical).tolist()
    else:
        raise ValueError("No suitable label column found. Expected 'label' or 'label_raw'.")

    texts = df[text_col].astype(str).tolist()

    tokenizers, models, reorder = load_models(model_repos, device)

    # Inference loop
    all_probs = []
    with torch.no_grad():
        for chunk in batch_iter(texts, batch_size):
            # For each model, compute probs and reorder to canonical
            probs_accum = None
            for m_idx, (tok, mdl, idx_map) in enumerate(zip(tokenizers, models, reorder)):
                enc = tok(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = mdl(**enc).logits  # [B, C_model]
                probs = softmax(logits)
                # Reorder columns from model order -> canonical order
                probs = probs[:, idx_map]
                # Weighted sum
                weighted = probs * w[m_idx]
                probs_accum = weighted if probs_accum is None else probs_accum + weighted

            all_probs.append(probs_accum.cpu())

    probs_cat = torch.cat(all_probs, dim=0)  # [N, 3]
    y_pred = probs_cat.argmax(dim=1).tolist()

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=CANONICAL_LABELS, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def default_top_models() -> Tuple[List[str], List[float]]:
    # Defaults based on README results (macro F1)
    repos = [
        "AshParmar/XMR-Muril",
        "AshParmar/XMR-Albert",
        "AshParmar/XMR-MBERT",
    ]
    weights = [0.79255, 0.779034, 0.776411]
    return repos, weights


def main():
    parser = argparse.ArgumentParser(description="Ensemble inference with weighted soft-voting for Marathi sentiment.")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Hugging Face repo IDs for models to ensemble. Default: top 3 from README.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        help="Weights for each model (will be normalized). Default derived from README macro F1.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default=str(Path("..") / "output" / "combined_dataset" / "test_strict.csv"),
        help="Path to test CSV with columns 'text' and 'label' or 'label_raw'",
    )
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("..") / "output"),
        help="Directory to write ensemble results and predictions",
    )

    args = parser.parse_args()

    if args.models is None or len(args.models) == 0:
        models, weights = default_top_models()
    else:
        models = args.models
        if args.weights is None or len(args.weights) != len(models):
            # uniform weights if not provided or mismatched
            weights = [1.0] * len(models)
        else:
            weights = args.weights

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")
    print(f"Models: {models}")
    print(f"Weights: {weights}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_ensemble(
        model_repos=models,
        weights=weights,
        test_csv=Path(args.test_csv),
        text_col=args.text_col,
        label_col=args.label_col,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    # Save results and predictions
    results_path = out_dir / "ensemble_results.json"
    preds_path = out_dir / "ensemble_predictions.csv"

    # Attach metadata for reproducibility
    payload = {
        "metrics": {k: v for k, v in results.items() if k not in ("y_pred",)},
        "metadata": {
            "models": models,
            "weights": weights,
            "device": str(device),
            "test_csv": str(Path(args.test_csv).resolve()),
            "text_col": args.text_col,
            "label_col": args.label_col,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Attach predictions to the input CSV rows
    df = pd.read_csv(args.test_csv)
    df_out = df.copy()
    inv_map = {i: lbl for i, lbl in enumerate(CANONICAL_LABELS)}
    df_out["pred_label"] = [inv_map[i] for i in results["y_pred"]]
    df_out.to_csv(preds_path, index=False)

    print("\nEnsemble metrics:")
    print(json.dumps({k: v for k, v in results.items() if k in ["accuracy", "macro_f1"]}, indent=2))
    print(f"Saved metrics to: {results_path}")
    print(f"Saved predictions to: {preds_path}")


if __name__ == "__main__":
    main()
