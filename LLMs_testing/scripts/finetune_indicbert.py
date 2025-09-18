import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)
from torch.utils.data import Dataset


LABEL_ORDER = {
    # Map human-readable labels to ids
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {v: k for k, v in LABEL_ORDER.items()}


class TextClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, text_col: str, label_col: str, max_length: int = 256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        label = int(row[self.label_col])
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = label
        return enc


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_and_prepare_dataset(csv_path: Path, output_split_dir: Path, test_size: float = 0.1, val_size: float = 0.1, seed: int = 42):
    df = pd.read_csv(csv_path)

    # Identify text and label columns
    if "text" not in df.columns:
        raise ValueError(f"Expected 'text' column in {csv_path}, found {df.columns.tolist()}")

    # Prefer numeric label_raw if present; else map 'label' string to ids
    if "label_raw" in df.columns:
        # Map -1,0,1 to 0,1,2 to keep ordering [neg, neutral, positive]
        def map_raw(x):
            if x == -1:
                return 0
            if x == 0:
                return 1
            if x == 1:
                return 2
            raise ValueError(f"Unexpected label_raw value: {x}")

        df["label_id"] = df["label_raw"].apply(map_raw)
    elif "label" in df.columns:
        # Clean and map
        df["label_id"] = df["label"].astype(str).str.lower().map(LABEL_ORDER)
        if df["label_id"].isna().any():
            bad = df[df["label_id"].isna()]["label"].unique()
            raise ValueError(f"Unrecognized label values: {bad}")
        df["label_id"] = df["label_id"].astype(int)
    else:
        raise ValueError("No usable label column found. Expect 'label_raw' or 'label'.")

    # Stratified splits: first split out temp test, then split remaining into train/val
    X = df["text"].values
    y = df["label_id"].values

    strat1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(strat1.split(X, y))
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    X_tv = df_train_val["text"].values
    y_tv = df_train_val["label_id"].values
    strat2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / (1.0 - test_size), random_state=seed)
    train_idx, val_idx = next(strat2.split(X_tv, y_tv))

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    ensure_dir(output_split_dir)
    df_train.to_csv(output_split_dir / "train.csv", index=False)
    df_val.to_csv(output_split_dir / "val.csv", index=False)
    df_test.to_csv(output_split_dir / "test.csv", index=False)

    return df_train, df_val, df_test


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "precision_macro": precision, "recall_macro": recall, "f1_macro": f1}


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(5.5, 4.5), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[ID2LABEL[i] for i in [0, 1, 2]],
                yticklabels=[ID2LABEL[i] for i in [0, 1, 2]])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def generate_wordclouds(df: pd.DataFrame, out_dir: Path, font_path: Path | None = None):
    try:
        from wordcloud import WordCloud
    except Exception as e:
        print(f"wordcloud not available: {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    # Build per-class text corpus
    for label_id, label_name in ID2LABEL.items():
        text_blob = "\n".join(df[df["label_id"] == label_id]["text"].astype(str).tolist())
        if not text_blob.strip():
            continue
        wc = WordCloud(width=1200, height=800, background_color="white",
                       font_path=str(font_path) if font_path else None,
                       collocations=False).generate(text_blob)
        out_file = out_dir / f"wordcloud_{label_name}.png"
        wc.to_file(str(out_file))
        print(f"Saved wordcloud: {out_file}")


def evaluate_model(model, tokenizer, dataset: Dataset, device: torch.device) -> Dict:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=64, collate_fn=collator)
    model.to(device)
    with torch.no_grad():
        for batch in loader:
            labels = batch.get("labels")
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
            if labels is None:
                continue
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(np.argmax(logits, axis=-1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=[ID2LABEL[i] for i in [0, 1, 2]], zero_division=0, output_dict=True)
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "classification_report": report,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndicBERT for Marathi sentiment classification")
    parser.add_argument("--dataset_csv", type=str, default=str(Path.cwd().parent / "output" / "balanced_mode_strict_domain.csv"),
                        help="Path to dataset CSV containing 'text' and 'label' or 'label_raw'")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base",
                        help="HuggingFace model name or local path")
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_root", type=str, default=str(Path.cwd() / "results" / "indicbert"),
                        help="Directory to save outputs")
    parser.add_argument("--font_path", type=str, default=str(Path.cwd().parent / "fonts" / "NotoSansDevanagari-Regular.ttf"),
                        help="Path to a Devanagari-capable TTF for wordclouds")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / ts
    ensure_dir(output_dir)

    splits_dir = output_dir / "splits"
    df_train, df_val, df_test = load_and_prepare_dataset(Path(args.dataset_csv), splits_dir, seed=args.seed)

    # Wordclouds from the full dataset (pre-training)
    try:
        font_path = Path(args.font_path)
        font_path = font_path if font_path.exists() else None
        generate_wordclouds(pd.concat([df_train, df_val, df_test], ignore_index=True), output_dir / "wordclouds", font_path)
    except Exception as e:
        print(f"Wordcloud generation failed: {e}")

    print("Loading tokenizer and model:", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_ORDER,
        use_safetensors=True,
    )

    # Datasets
    train_ds = TextClassificationDataset(df_train, tokenizer, text_col="text", label_col="label_id", max_length=args.max_length)
    val_ds = TextClassificationDataset(df_val, tokenizer, text_col="text", label_col="label_id", max_length=args.max_length)
    test_ds = TextClassificationDataset(df_test, tokenizer, text_col="text", label_col="label_id", max_length=args.max_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Baseline evaluation before fine-tuning
    print("Running baseline evaluation (pre-finetune)...")
    baseline_metrics = evaluate_model(model, tokenizer, test_ds, device)
    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in baseline_metrics.items() if k != "y_true" and k != "y_pred"}, f, ensure_ascii=False, indent=2)
    save_confusion_matrix(np.array(baseline_metrics["y_true"]), np.array(baseline_metrics["y_pred"]),
                          output_dir / "confusion_matrix_baseline.png", title="Baseline Confusion Matrix")
    print("Baseline metrics:", {k: v for k, v in baseline_metrics.items() if k in ["accuracy", "f1_macro"]})

    # Training
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(8, args.batch_size),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=args.seed,
        report_to=["none"],
        fp16=torch.cuda.is_available(),
    )

    def compute_metrics_trainer(eval_pred):
        logits, labels = eval_pred
        return compute_metrics_fn((logits, labels))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_trainer,
    )

    print("Starting fine-tuning...")
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))

    # Post-finetune evaluation
    print("Evaluating fine-tuned model on test set...")
    metrics_after = trainer.evaluate(test_ds)
    # Also compute full report and confusion matrix
    model.eval()
    final_metrics = evaluate_model(model, tokenizer, test_ds, device)

    with open(output_dir / "finetuned_metrics.json", "w", encoding="utf-8") as f:
        json.dump({**metrics_after, **{k: v for k, v in final_metrics.items() if k not in ["y_true", "y_pred"]}}, f, ensure_ascii=False, indent=2)

    save_confusion_matrix(np.array(final_metrics["y_true"]), np.array(final_metrics["y_pred"]),
                          output_dir / "confusion_matrix_finetuned.png", title="Finetuned Confusion Matrix")

    # Save quick summary
    summary = {
        "model": args.model_name,
        "device": str(device),
        "dataset": str(Path(args.dataset_csv).resolve()),
        "splits_dir": str(splits_dir.resolve()),
        "outputs": str(output_dir.resolve()),
        "baseline": {k: v for k, v in baseline_metrics.items() if k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]},
        "finetuned": {k: v for k, v in final_metrics.items() if k in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]},
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done. Summary saved to:", output_dir / "summary.json")


if __name__ == "__main__":
    main()
