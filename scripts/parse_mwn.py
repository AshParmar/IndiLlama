import os
import re
import pandas as pd
from pathlib import Path

try:
    from nltk.corpus import wordnet as wn
    from nltk.corpus import sentiwordnet as swn
except Exception:
    wn = None
    swn = None

# ---------------- CONFIG ----------------
POS_FILES = {
    'adjective': 'idxadjective_txt',
    'noun':      'idxnoun_txt',
    'verb':      'idxverb_txt',
    'adverb':    'idxadverb_txt'
}
POS_MAP = {'01':'n', '02':'v', '03':'a', '04':'r'}
DATA_FILE_NAME = 'data_txt'

# ---------------- FIND DATABASE FOLDER ----------------
root = Path.cwd()
db_folder = None
for p in root.glob('**/MarathiWN_1_3/database'):
    if p.is_dir():
        db_folder = p
        break

if db_folder is None:
    raise FileNotFoundError("Could not locate MarathiWN_1_3/database folder")

print("Found MarathiWN database at:", db_folder)
print('Files in db:', sorted([p.name for p in db_folder.iterdir() if p.is_file()])[:50])

# ---------------- PARSE GLOSSES ----------------
gloss_dict = {}
data_file = db_folder / DATA_FILE_NAME
if not data_file.exists() and (db_folder / (DATA_FILE_NAME + '.txt')).exists():
    data_file = db_folder / (DATA_FILE_NAME + '.txt')

if data_file.exists():
    with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|', maxsplit=1)
            if len(parts) < 2:
                continue
            left, gloss = parts
            left_parts = left.split()
            synset_id = left_parts[-1]
            gloss_dict[synset_id] = gloss.strip()
else:
    print(f"Warning: gloss data file not found (looked for {data_file}) - continuing without glosses")

# ---------------- PARSE POS FILES ----------------
records = []
synid_re = re.compile(r'^\d{5,9}$')

for pos_name, pos_file_name in POS_FILES.items():
    pos_path = db_folder / pos_file_name
    if not pos_path.exists() and (db_folder / (pos_file_name + '.txt')).exists():
        pos_path = db_folder / (pos_file_name + '.txt')
    if not pos_path.exists():
        print(f"Warning: {pos_file_name} not found. Skipping {pos_name}.")
        continue

    with open(pos_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            word = parts[0].lstrip('-')
            pos_code = parts[1]
            synset_ids = [t for t in parts if synid_re.match(t)]
            glosses = [gloss_dict.get(sid, '') for sid in synset_ids]

            en_words, pos_scores, neg_scores, obj_scores = [], [], [], []
            for sid in synset_ids:
                try:
                    pos_char = POS_MAP.get(pos_code)
                    if pos_char is None:
                        continue
                    if wn is None:
                        continue
                    offset = int(sid)
                    syn = wn.synset_from_pos_and_offset(pos_char, offset)
                    en_words.append(syn.name())
                    if swn is not None:
                        try:
                            swn_syn = swn.senti_synset(syn.name())
                            pos_scores.append(swn_syn.pos_score())
                            neg_scores.append(swn_syn.neg_score())
                            obj_scores.append(swn_syn.obj_score())
                        except Exception:
                            pass
                except Exception:
                    continue

            records.append({
                'marathi_word': word,
                'pos': pos_name,
                'synset_ids': ','.join(synset_ids),
                'gloss': ' || '.join(glosses),
                'english_synsets': ','.join(en_words),
                'positive': round(sum(pos_scores)/len(pos_scores), 3) if pos_scores else 0.0,
                'negative': round(sum(neg_scores)/len(neg_scores), 3) if neg_scores else 0.0,
                'objective': round(sum(obj_scores)/len(obj_scores), 3) if obj_scores else 0.0,
            })

out_csv = root / 'marathi_sentiwordnet.csv'
pd.DataFrame(records).to_csv(out_csv, index=False, encoding='utf-8')
print(f"Saved structured Marathi SentiWordNet CSV to: {out_csv}")
print(f"Total records: {len(records)}")

# Print a small sample
import itertools
for r in itertools.islice(records, 5):
    print(r)
