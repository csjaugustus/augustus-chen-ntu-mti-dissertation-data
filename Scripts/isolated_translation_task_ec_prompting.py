"""
Description:
This script generates three Chinese translations for each English source sentence
in a JSON dataset (based on the WMT19 100-sentence test set). It uses OpenAI's
Chat Completions API in a one-shot translation setting, prompting GPT-4o to produce
Chinese translations based on a single example.

For each source sentence (ST) that lacks translations (TT1, TT2, TT3), the script:
1. Constructs a prompt using a one-shot example.
2. Calls the OpenAI API until three valid translations are collected.
3. Populates the TT1, TT2, and TT3 fields with the generated translations.
4. Writes the updated dataset back to the JSON file after each sentence.

Note:
- File paths must be updated to match your local/project directory structure.
- Requires an API key in a plain text config file (excluded from version control).
"""

from openai import OpenAI
import json
import re
import os

def generate(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

# Load API key (stored in a secure config file outside the repo)
with open("path/to/config.txt", 'r') as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

# Load the JSON data
with open("path/to/translations-with-ref-ec.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i, d in enumerate(data):
    if not d["TT1"]["text"] and not d["TT2"]["text"] and not d["TT3"]["text"]:
        st = d["ST"]["en"]
        prompt = f"Translate the following sentence into Chinese. Here is an example: ST: I love cats. TT: 我喜欢猫。Now translate the following. ST: {st}"
        valid_tts = []
        while len(valid_tts) < 3:
            tt = generate(prompt)
            tt_pattern = r"TT:\s*(.*)"
            match = re.search(tt_pattern, tt)

            if match:
                valid_tts.append(match.group(1))

        for j, tt_text in enumerate(valid_tts, start=1):
            d[f"TT{j}"] = {
                "text": tt_text,
                "scores": {
                    "bleu": None,
                    "chrf": None,
                    "comet-ref": None,
                    "comet-noref": None
                }
            }

        # Save progress after each update
        with open("path/to/translations-with-ref-ec.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"{i+1}/{len(data)} done.")