import argparse
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

assets = json.loads(Path(__file__).with_name("assets.json").read_text(encoding="utf-8"))
parser = argparse.ArgumentParser()
parser.add_argument("model")
args = parser.parse_args()

revision = assets["models"][args.model]
model_path = Path(snapshot_download(
    args.model,
    revision=revision,
    ignore_patterns=["original/*"],
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model", "tokenizer*"],
))
refs = model_path.parent.parent / "refs"
refs.mkdir(exist_ok=True)
(refs / "main").write_text(revision, encoding="utf-8")
print(f"{args.model}: {model_path}")

for name, spec in assets["datasets"].items():
    dataset = load_dataset(**spec)
    for split, files in dataset.cache_files.items():
        for file in files:
            print(f"{name}/{split}: {file['filename']}")
