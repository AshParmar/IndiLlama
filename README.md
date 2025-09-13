Marathi Sentiment Lexicon + LLM Fine-Tuning
📖 Overview

This repository provides a Marathi sentiment lexicon and an LLM fine-tuning pipeline for sentiment analysis.

Step 1: Translate Marathi words into English using Google Translate.

Step 2: Map English words to SentiWordNet synsets.

Step 3: Assign positive, negative, and objective scores to each word.

Step 4: Use the lexicon to perform sentiment analysis on the L3Cube Marathi Tweets dataset.

Step 5: Fine-tune Large Language Models (LLMs) with this enriched dataset to improve Marathi sentiment classification.

⚙️ Features

✔️ Marathi → English word translation
✔️ SentiWordNet-based sentiment mapping
✔️ Ready-to-use Marathi SentiWordNet lexicon
✔️ Sentiment analysis of L3Cube tweets
✔️ LLM fine-tuning pipeline (Hugging Face Transformers)

🚀 Installation
git clone https://github.com/your-username/marathi-senti-llm.git
cd marathi-senti-llm
pip install -r requirements.txt

📂 Dataset

Input 1: Marathi lexicon (marathi_lexicon_correct_pos.csv)

Input 2: L3Cube Marathi Tweets Dataset (publicly available)

Output 1: marathi_sentiwordnet_final.csv

Output 2: Fine-tuned LLM sentiment model

🔧 Usage
# Load the final lexicon
import pandas as pd
df = pd.read_csv("marathi_sentiwordnet_final.csv")

# Example: lookup word
print(df[df['marathi_word'] == "सुंदर"])

# Fine-tune LLM (example with Hugging Face Trainer)
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

📊 Applications

Sentiment analysis of Marathi tweets/news/reviews

Fine-tuned Marathi LLM for classification tasks

Cross-lingual NLP research in low-resource settings

Baseline resource for Indian language sentiment datasets

🛠️ Requirements

Python 3.8+

googletrans==4.0.0-rc1

nltk, transformers, datasets

torch (GPU recommended)

📌 Credits

L3Cube Pune – Marathi Tweet Sentiment Dataset

SentiWordNet – English sentiment lexicon

Hugging Face – Transformers & fine-tuning framework

