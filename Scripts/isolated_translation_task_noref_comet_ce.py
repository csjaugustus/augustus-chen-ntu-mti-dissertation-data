"""
Description:
This script computes reference-free COMET scores for Chinese-to-English translations
for the Isolated Translation Task. It uses the WMT22 COMETKiwi-DA model from
Hugging Face to evaluate the quality of each machine translation based only on the
source and translated segments.

The script:
1. Loads a JSON file containing Chinese source sentences and English translations.
2. Prepares data entries for COMET scoring using src and mt only.
3. Loads the COMETKiwi model and predicts quality scores.
4. Does not require human reference translations.

Note:
- File paths should be customized to your environment.
- Scores can optionally be saved back into the JSON structure.
"""

from comet import download_model, load_from_checkpoint
import json

# Load the COMETKiwi model
model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

# Load your translation dataset
with open("path/to/translations-with-ref-ce.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Prepare input examples
data = []
for d in raw_data:
    for key in ["TT1", "TT2", "TT3"]:
        data.append({
            "src": d["ST"]["zh"],      # Chinese source
            "mt": d[key]["text"]       # English translation
        })

# Run prediction
model_output = model.predict(data, batch_size=8, gpus=1)