# Marathi Sentiment Analysis with Lexical + LLM Hybrid Approach

> Adapting Large Language Models for Low-Resource Indic (Marathi) Sentiment Analysis using a constructed Marathi SentiWordNet, lexicon refinement, and supervised fine-tuning on L3Cube MahaSent.

## 1. Motivation
Marathi is comparatively low-resource for high quality sentiment tools. Pure end-to-end fine-tuning of transformers often under-utilizes rich lexical structure present in IndoWordNet / Marathi WordNet. This project builds a reproducible pipeline that:
- Parses the Marathi WordNet (v1.3) resource into a structured lexical graph.
- Derives a Marathi SentiWordNet style lexicon by aligning Marathi synsets with English WordNet + SentiWordNet polarity/objectivity scores (via `nltk`).
- Cleans and corrects POS / synset identifiers and glosses.
- Combines lexicon-derived priors with supervised labels from the L3Cube MahaSent dataset for robust classifier / LLM adaptation.
- Explores lightweight fine-tuning / instruction-tuning strategies for multilingual / Indic-capable models (IndicBERT, mBERT, XLM-R, Aya, Mistral / LLaMA variants with Marathi coverage) augmented by lexicon signals.

## 2. Repository Structure (key parts)
```
MarathiWN_1_3/        # Raw Marathi WordNet distribution (database + config + jars)
scripts/parse_mwn.py  # Parser to extract synsets + gloss + polarity (averaged from SentiWordNet)
MahaSent_MR_Train.csv # L3Cube MahaSent training data (sentence,label)
marathi_lexicon_correct_pos.csv          # Canonical cleaned lexicon with padded synset ids
marathi_sentiwordnet_google.csv          # Variant produced via Google translation mapping
marathi_sentiwordnet_mariante.csv        # Variant produced via Marian / alternative MT & scoring
```
Other Jupyter notebooks (e.g. `google_translate.ipynb`, `mariante_translate.ipynb`, `lexi.ipynb`) appear to handle translation experiments and lexicon enrichment.

## 3. Datasets & Lexical Assets
### 3.1 L3Cube MahaSent
- Source: L3Cube Pune (multi-domain Marathi sentiment corpus). (Ensure you follow original license / citation.)
- Schema: `marathi_sentence,label` where label currently appears as {-1, 0, 1} (only -1 observed in sampled head; confirm full distribution). Negative examples dominate in the shown sample—perform class balance analysis.

### 3.2 Marathi WordNet (v1.3)
- Files under `MarathiWN_1_3/database/` (idx* and data_txt) store lemma → synset id mappings and glosses.
- POS mapping applied in `parse_mwn.py` (adjective, noun, verb, adverb). Some entries have `unknown` POS after downstream translation / merging.

### 3.3 Constructed Marathi SentiWordNet
Two (or more) derivations:
1. `marathi_sentiwordnet_google.csv` – lexicon derived using Google translated alignments (columns: `marathi_word,pos,synset_id,gloss,english_word`). Currently lacks numeric sentiment scores.
2. `marathi_sentiwordnet_mariante.csv` – similar layout but includes added columns: `positive,negative,objective` (scores presently mostly 0.0 placeholders; requires validation).
3. `marathi_lexicon_correct_pos.csv` – canonical cleaned lexicon with zero-padded synset ids and POS normalization.

### 3.4 Quality / Noise Notes
- Several English gloss translations or `english_word` fields appear noisy (Artifacts like `[ Map on page 22]`, random phrases). Set up filters (regex for bracketed patterns, punctuation-only tokens) to drop / flag.
- Many sentiment score columns are zero → need a backfill strategy: (a) re-align via `nltk.corpus.sentiwordnet` using English lemma forms; (b) propagate average from sibling synsets with same Marathi lemma; (c) fallback heuristic: adjectives/verbs more sentiment-bearing → assign small prior magnitude if lexeme appears in MahaSent with strongly skewed class conditional probability.

## 4. Pipeline Overview
1. Parse WordNet: `python scripts/parse_mwn.py` (assumes current working directory root; auto-discovers `MarathiWN_1_3/database`).
2. Produce base CSV (`marathi_sentiwordnet.csv`).
3. Augment with MT variants (Google / Marian) to enrich `english_word` for alignment when original direct mapping fails.
4. Clean & normalize IDs / POS → `marathi_lexicon_correct_pos.csv`.
5. Recompute sentiment scores:
   - Map Marathi lemma → candidate English synsets (via `wn.synset_from_pos_and_offset`).
   - Fetch SentiWordNet pos/neg/obj per synset; average across matches.
   - Store non-zero scores; maintain objectivity = 1 - (pos+neg) if not given.
6. Merge with MahaSent sentences:
   - Tokenize (Indic-friendly: use simple whitespace + punctuation split, or Indic NLP Library if available; ensure normalization of nukta / diacritics).
   - Generate sentence-level lexicon features: sum/avg of pos, neg, (pos-neg), count of strong polarity tokens, proportion of unknown tokens, etc.
7. Modeling Strategies:
   - Baseline classical: Logistic Regression / SVM on TF-IDF + lexicon features.
   - Transformer baseline: Fine-tune XLM-R-base / IndicBERT directly on MahaSent.
   - Lexicon-informed Transformer: Concatenate lexicon feature vector to CLS representation (simple two-branch head) or prompt-inject lexical prior (e.g., prepend a constructed sentiment hint sentence summarizing aggregated polarity).
   - Instruction / PEFT (LoRA, QLoRA) fine-tune a lightweight LLaMA / Mistral variant with few-shot lexical explanations.
8. Evaluation: Stratified k-fold (if dataset size permits) or held-out dev split; metrics: Macro F1, Accuracy, per-class F1, confusion matrix, calibration (ECE), ablations (w/ vs w/o lexicon features).

## 5. Environment Setup
Install dependencies (Python 3.10+ recommended):
```
pip install -r requirements.txt
```
Ensure `nltk` corpora:
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('sentiwordnet')
```
(Place inside a setup cell / script before running `parse_mwn.py`.)

## 6. Example: Parsing the WordNet
```bash
python scripts/parse_mwn.py
```
Outputs: `marathi_sentiwordnet.csv` with columns:
- marathi_word
- pos (coarse)
- synset_ids (comma-separated)
- gloss
- english_synsets (comma-separated WordNet synset names)
- positive / negative / objective (averaged scores; may be zero if lookup failed)

## 7. Feature Engineering (Planned)
Lexicon features per sentence:
- sum_pos, sum_neg, max_pos, max_neg
- polarity_sum = sum_pos - sum_neg
- polarity_ratio = (sum_pos+1) / (sum_neg+1)
- sentiment_token_density = (# tokens with |pos-neg|>τ) / len(tokens)
- negation_adjusted_polarity: simple rule-based flip for phrases following Marathi negators ("नाही", "नसले", etc.)
- uncertainty_count (words meaning doubt / possibility) to modulate intensity.

## 8. Fine-Tuning Strategy
### 8.1 Classical Baselines
1. TF-IDF (char 3–5 ngrams + word unigrams)
2. Add lexicon derived features → evaluate incremental gain.

### 8.2 Transformer Baselines
- `xlm-roberta-base`, `ai4bharat/indic-bert`, `google/muril-base-cased` (if added) — standard classification head.

### 8.3 Lexicon-Augmented Transformer
Option A: Dual Encoder Head
- Forward pass obtains CLS embedding (d_model).
- Compute lexicon feature vector f (k dims).
- Concatenate [CLS || f] → feed to MLP classifier.

Option B: Soft Prompt / Prefix
- Create a textual prefix: "Lexicon sentiment summary: total_pos=X total_neg=Y polarity=P." prepend to each sentence during training.

Option C: Adapter / LoRA with Extra Feature Token
- Insert a learned [LEX] token whose embedding is initialized from linear projection of lexicon features at each step.

### 8.4 Large Language Model (Instruction) Adaptation
Use QLoRA on a 7B multilingual model (e.g., `meta-llama-3.1-8B-instruct` if license permits or `OpenHermes` variant with Marathi support). Create instruction-style training pairs:
```
### Instruction
निर्णय : खालील वाक्याचे भावनिक वर्गीकरण द्या. वाक्य: <S>
### Lexicon Hints
सकारात्मक गुण: {sum_pos} नकारात्मक गुण: {sum_neg} ध्रुवता: {polarity}
### Response
{label}
```
Compare w/o hints.

## 9. Evaluation Plan
| Model | Macro F1 | Acc | Δ vs Baseline | Notes |
|-------|----------|-----|--------------|-------|
| TF-IDF | - | - | - | baseline |
| TF-IDF + Lexicon | - | - | +? | feature gain |
| XLM-R | - | - | - | transformer baseline |
| XLM-R + Features | - | - | +? | concatenated features |
| XLM-R + Prompt | - | - | +? | lexical prefix |
| QLoRA LLM (no hints) | - | - | - | instruction |
| QLoRA LLM (hints) | - | - | +? | effect of hints |

Add: confusion matrix, per-class breakdown, statistical test (McNemar) for significance between top two models.

## 9.1 Lexicon Scoring Notebook (`sentiwordnet.ipynb`)
This notebook operationalizes the re-scoring of the Google-translation derived lexicon (`marathi_sentiwordnet_google.csv`) using **NLTK SentiWordNet** lookups on the English translation token of each Marathi entry.

### Outputs
- `marathi_word_sentiments.csv` with columns:
   - `marathi_word`
   - `sentiment_label` (one of `+`, `-`, `neutral`)
   - `pos_score`, `neg_score`, `obj_score` (averaged over all WordNet synsets for the English lemma)
   - `english_translation`
   - (Optional passthrough) `synset_id`, `pos` if present in the source file.

### Labeling Heuristic
1. Normalize English translation (strip brackets, punctuation, keep first token).
2. Collect all WordNet synsets for the lemma; acquire each synset's SentiWordNet triple (pos / neg / obj).
3. Average scores across synsets (coarse approximation; future improvement: disambiguate using POS or gloss similarity).
4. Assign label:
    - If `pos==0 and neg==0` → `neutral`.
    - Else if `pos - neg > 0.05` → `+`.
    - Else if `neg - pos > 0.05` → `-`.
    - Else `neutral`.

### Word Cloud Visualization
The notebook includes a Devanagari-aware word cloud step that:
- Auto-detects a system font (tries: Nirmala UI, Mangal, Kokila, Arial Unicode MS).
- Falls back to downloading **Noto Sans Devanagari** if none found.
- Colors tokens by sentiment label (`+` green, `-` red, neutral gray).

### Diagnostics Cell
Before rendering the cloud, a diagnostics cell reports:
- Total rows, unique Marathi words.
- Count of words containing Devanagari codepoints (U+0900–U+097F).
- Sample of first words & top frequency items.
If *zero* words contain Devanagari characters, the source file likely uses transliteration or has encoding issues; the cloud will appear empty.

### Common Issues & Fixes
| Issue | Symptom | Resolution |
|-------|---------|------------|
| No Devanagari glyphs rendered | Empty / blank word cloud | Ensure `marathi_word` column truly contains native script; open CSV in UTF-8 aware editor. |
| Font warning printed | Marathi squares/tofu in cloud | Manually set `font_path = r"C:/Windows/Fonts/Mangal.ttf"` (or another valid font) in the word cloud cell. |
| All labels `neutral` | Value counts dominated by neutral | Threshold 0.05 may be high; try 0.02 or perform POS filtering to reduce averaging noise. |
| Very sparse non-zero scores | Most lemmas not in WordNet | Implement backoff: morphological simplification, bilingual pivot, or embedding similarity search to nearest English lemma. |
| Mixed noisy English translations | Bracketed artifacts remain | Strengthen cleaning regex (already strips `[ ... ]` segments) or filter multi-word phrases. |

### Planned Enhancements
- Synset disambiguation via gloss similarity (cosine over sentence embeddings).
- POS-constrained scoring (restrict synsets to the Marathi POS tag).
- Importance weighting by synset frequency / sense rank instead of uniform mean.
- Integration of polarity magnitude into model input features (e.g., per sentence aggregated sums & proportions).

### Quick Use
Run cells sequentially in `sentiwordnet.ipynb` to produce `marathi_word_sentiments.csv`, then:
```python
import pandas as pd
lex_sent = pd.read_csv('marathi_word_sentiments.csv')
lex_sent['sentiment_label'].value_counts()
```

## 10. Roadmap
- [x] Recompute polarity scores prototype (averaged lemma-based SentiWordNet via notebook).
- [ ] Refine scoring with POS & sense disambiguation.
- [ ] Implement lexicon feature extractor module.
- [ ] Train baseline classical + transformer models.
- [ ] Add prompt-based augmentation experiment.
- [ ] Implement dual-branch head (CLS + features).
- [ ] QLoRA fine-tune multilingual LLM with / without hints.
- [ ] Evaluation + ablation report.
- [ ] Error analysis: inspect top misclassifications; refine lexicon or negation handling.
- [ ] Package pipeline into reproducible scripts + notebook.
- [ ] Add automated font detection & fallback test in CI (optional).
- [ ] Add gloss-based synset disambiguation prototype.

## 11. Potential Improvements
- Morphological normalization: integrate light stemmer for Marathi (rule-based or unsupervised) to boost lexicon coverage.
- Contextual polarity adjustment: shift prior if negation scope or intensifiers ("फार", "खूप") present.
- Semi-supervised label propagation: use lexicon polarity to pseudo-label unlabeled Marathi corpora → self-training.
- Active learning loop: highlight uncertain sentences for human annotation to expand balanced dataset.

## 12. Citation & Licensing
Please cite original sources:
- L3Cube MahaSent dataset paper.
- Marathi WordNet / IndoWordNet references.
- SentiWordNet 3.0.
- Any pretrained transformer / LLM checkpoints you fine-tune.

(Insert BibTeX entries once gathered.)

## 13. Quick Start Snippet
```python
import pandas as pd
lex = pd.read_csv('marathi_sentiwordnet_mariante.csv')
mahasent = pd.read_csv('MahaSent_MR_Train.csv')
print('Lexicon size:', len(lex), 'Sample:', lex.head(3))
print('MahaSent size:', len(mahasent))
```

## 14. Disclaimer
This lexicon & derived scores are experimental and may contain translation noise. Always validate manually before deployment in sensitive applications.

---
Feel free to open issues / feature requests to refine the lexicon scoring, add evaluation scripts, or integrate additional Indic languages (scalable design aims for cross-lingual extension).
