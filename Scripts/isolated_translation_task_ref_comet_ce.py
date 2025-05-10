"""
Description:
This script computes reference-based COMET scores for Chinese-to-English translations
for the Isolated Translation Task using the WMT22 COMET-DA model.

The script:
1. Loads a JSON file containing Chinese source sentences and English translations.
2. Constructs COMET input dictionaries (src, mt, ref) for each of the three translations (TT1–TT3).
3. Uses the COMET model to compute quality scores based on the reference translation.
4. Designed specifically for Isolated Translation Tasks.

Note:
- File paths must be customized for your project.
"""

from comet import download_model, load_from_checkpoint
import json

# Load COMET model
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

# Load translation data
with open("path/to/translations-with-ref-ce.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Prepare COMET input data
data = []
for d in raw_data:
    for key in ["TT1", "TT2", "TT3"]:
        data.append({
            "src": d["ST"]["zh"],      # Chinese source
            "mt": d[key]["text"],      # English translation
            "ref": d["ST"]["en"]       # Human reference (English)
        })

# Run model prediction
model_output = model.predict(data, batch_size=8, gpus=1)