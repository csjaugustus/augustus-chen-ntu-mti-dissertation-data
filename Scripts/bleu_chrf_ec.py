"""
Description:
This script computes sentence-level BLEU and chrF scores for three English-to-Chinese
translations (TT1, TT2, TT3) generated for each English source sentence in a dataset.
It uses SacreBLEU's scoring tools with Chinese-specific tokenization and evaluates the
quality of the translations by comparing them to the human reference translation in Chinese. 
This is for the Isolated Translation Task.

The script:
1. Loads a JSON file containing source sentences and machine-generated translations.
2. Computes BLEU and chrF scores for each translation against the Chinese reference.
3. Stores the computed scores back into the corresponding JSON structure.
4. Writes the updated data with scores to a new JSON file.

Note:
- Punctuation is stripped from the translations before scoring.
- File paths must be updated to your project structure and excluded from version control.
"""

from sacrebleu.metrics import BLEU, CHRF
import json
import re

def strip_punctuation(text):
    # Remove all non-word, non-space characters except Chinese
    pattern = r'[^\w\s\u4e00-\u9fff]'
    return re.sub(pattern, '', text)

# Load dataset (update path accordingly)
with open("path/to/translations-with-ref-ec.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Compute BLEU and chrF scores
for d in data:
    bleu = BLEU(trg_lang="zh", effective_order=True, tokenize="zh")
    chrf = CHRF()

    for tt_key in ["TT1", "TT2", "TT3"]:
        hypothesis = strip_punctuation(d[tt_key]["text"])
        reference = d["ST"]["zh"]

        d[tt_key]["scores"]["bleu"] = bleu.sentence_score(reference, [hypothesis]).score
        d[tt_key]["scores"]["chrf"] = chrf.sentence_score(reference, [hypothesis]).score

# Save results to new file
with open("path/to/translations-with-ref-ec-scored.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)