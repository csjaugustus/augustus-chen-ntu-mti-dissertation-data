"""
Description:
This script computes reference-based COMET scores for English-to-Chinese translations
for the Continuous Translation Task. It uses the WMT22 COMET-DA model from Hugging Face
to evaluate the quality of three translations (TT1, TT2, TT3) against the human reference.

The script:
1. Loads a JSON file containing English source sentences and Chinese translations.
2. Prepares the input data for COMET scoring (with src, mt, and ref fields).
3. Authenticates with the Hugging Face Hub and loads the COMET model.
4. Computes COMET scores in batch mode.

Note:
- Requires a Hugging Face token stored in a secure variable (e.g., Colab `userdata`).
- Update paths and input handling if running outside of Colab.
"""

from comet import download_model, load_from_checkpoint
from huggingface_hub import login
from google.colab import userdata  # Only needed if running in Google Colab
import json

# Authenticate with Hugging Face (requires token in Colab environment)
user_token = userdata.get("huggingface_token")
login(token=user_token)

# Load the COMET model (reference-based)
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

# Load your translation dataset
with open("path/to/translations-with-ref-ec.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Prepare input examples for COMET
data = []
for d in raw_data:
    for key in ["TT1", "TT2", "TT3"]:
        data.append({
            "src": d["ST"]["en"],       # English source
            "mt": d[key]["text"],       # Machine translation (Chinese)
            "ref": d["ST"]["zh"]        # Human reference (Chinese)
        })

# Run COMET scoring
model_output = model.predict(data, batch_size=8, gpus=1)