"""
Description:
This script computes reference-free COMET scores for English-to-Chinese translations
for the Continuous Translation Task using the COMETKiwi-DA model.

The script:
1. Reads a JSON dataset of English source and machine-translated Chinese output.
2. Creates input records for COMET scoring (src, mt).
3. Runs model predictions in batch mode without using references.

Note:
- Suitable for large-scale continuous translation evaluations.
- COMETKiwi enables quality estimation without human references.
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
            "src": d["ST"]["en"],
            "mt": d[key]["text"]
        })

model_output = model.predict(data, batch_size=8, gpus=1)