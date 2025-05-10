"""
Description:
This script computes sentence-level BLEU and chrF scores for three Chinese-to-English
translations (TT1, TT2, TT3) generated for each Chinese source sentence in a dataset.
It uses SacreBLEU's scoring tools with English-specific tokenization and evaluates the
quality of the translations by comparing them to the human reference translation in English.

The script:
1. Loads a JSON file containing source sentences and machine-generated translations.
2. Computes BLEU and chrF scores for each translation against the English reference.
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
    # Remove all non-word and non-space characters
    pattern = r'[^\w\s]'
    return re.sub(pattern, '', text)

# Load dataset (update path accordingly)
with open("path/to/translations-with-ref-ce.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Compute BLEU and chrF scores
for d in data:
    bleu = BLEU(trg_lang="en", effective_order=True)
    chrf = CHRF()

    for tt_key in ["TT1", "TT2", "TT3"]:
        hypothesis = strip_punctuation(d[tt_key]["text"])
        reference = d["ST"]["en"]

        d[tt_key]["scores"]["bleu"] = bleu.sentence_score(reference, [hypothesis]).score
        d[tt_key]["scores"]["chrf"] = chrf.sentence_score(reference, [hypothesis]).score

# Save results to new file
with open("path/to/translations-with-ref-ce-scored.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)