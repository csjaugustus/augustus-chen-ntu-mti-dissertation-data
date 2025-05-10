"""
Description:
This script computes reference-free COMET scores for English-to-Chinese translations
for the Isolated Translation Task using the WMT22 COMETKiwi-DA model.

The script:
1. Loads a JSON file with English source sentences and Chinese translations.
2. Prepares (src, mt) input pairs for COMET.
3. Runs reference-free quality estimation with COMETKiwi.

Note:
- Reference translations are not needed for this evaluation.
- Designed for one-shot (isolated) translation tasks.
"""

from comet import download_model, load_from_checkpoint
import json

model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

with open("path/to/translations-with-ref-ec.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

data = []
for d in raw_data:
    for key in ["TT1", "TT2", "TT3"]:
        data.append({
            "src": d["ST"]["en"],      # English source
            "mt": d[key]["text"]       # Chinese translation
        })

model_output = model.predict(data, batch_size=8, gpus=1)