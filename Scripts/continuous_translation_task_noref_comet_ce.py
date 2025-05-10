"""
Description:
This script computes reference-free COMET scores for Chinese-to-English translations
for the Continuous Translation Task. Using the WMT22 COMETKiwi-DA model,
it evaluates the translation quality without requiring reference translations.

The script:
1. Loads a JSON dataset with Chinese source and English MT outputs.
2. Constructs COMET-compatible (src, mt) input objects.
3. Evaluates in batch mode using COMETKiwi.

Note:
- Designed for continuous evaluation workflows.
- Adapt batch size and GPU setting if running locally.
"""

from comet import download_model, load_from_checkpoint
import json

model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

with open("path/to/translations-with-ref-ce.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

data = []
for d in raw_data:
    for key in ["TT1", "TT2", "TT3"]:
        data.append({
            "src": d["ST"]["zh"],
            "mt": d[key]["text"]
        })

model_output = model.predict(data, batch_size=8, gpus=1)