"""
Description:
This script computes reference-based COMET scores for English-to-Chinese translations
for the Isolated Translation Task using the WMT22 COMET-DA model.

The script:
1. Loads a JSON file containing English source sentences and Chinese translations.
2. Constructs (src, mt, ref) inputs for COMET scoring across TT1–TT3.
3. Outputs model predictions using a reference-based metric.
4. Optimized for Isolated Translation Tasks (evaluated individually).

Note:
- Human references are required.
- File paths must be adapted to your environment.
"""

from comet import download_model, load_from_checkpoint
import json

# Load COMET model
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

# Load translation data
with open("path/to/translations-with-ref-ec.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Prepare COMET input data
data = []
for d in raw_data:
    for key in ["TT1", "TT2", "TT3"]:
        data.append({
            "src": d["ST"]["en"],      # English source
            "mt": d[key]["text"],      # Chinese translation
            "ref": d["ST"]["zh"]       # Human reference (Chinese)
        })

# Run model prediction
model_output = model.predict(data, batch_size=8, gpus=1)