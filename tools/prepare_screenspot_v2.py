from pathlib import Path
import json
import random

import pandas as pd
from datasets import load_dataset


ROOT = Path("/home/kwangryeol/workspace/Qwen3-8B-Instruct")
DATA_ROOT = ROOT / "datasets" / "screenspot-v2"
RAW_DIR = DATA_ROOT / "raw"
OUT_DIR = DATA_ROOT / "easyr1"


def normalize_bbox(bbox, width, height):
    x, y, w, h = map(float, bbox)

    return [
        1000.0 * x / width,
        1000.0 * y / height,
        1000.0 * (x + w) / width,
        1000.0 * (y + h) / height,
    ]


def build_image_index():
    index = {}

    for path in RAW_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".webp"
        }:
            index[path.name] = path.resolve()

    return index


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        str(RAW_DIR),
        split="test",
    )

    image_index = build_image_index()

    rows = []

    for i, sample in enumerate(dataset):
        image_name = Path(sample["img_url"]).name
        image_path = image_index.get(image_name)

        if image_path is None:
            raise FileNotFoundError(image_name)

        width, height = map(int, sample["img_size"])

        bbox = normalize_bbox(
            sample["bbox"],
            width,
            height,
        )

        x1, y1, x2, y2 = bbox

        answer = {
            "bbox": bbox,
            "center": [
                (x1 + x2) / 2,
                (y1 + y2) / 2,
            ],
        }

        rows.append(
            {
                "id": str(sample.get("id", i)),
                "problem": sample["task"],
                "images": [str(image_path)],
                "answer": json.dumps(answer),
                "data_type": sample.get("data_type"),
                "data_source": sample.get("data_source"),
            }
        )

    rng = random.Random(42)
    rng.shuffle(rows)

    split = int(len(rows) * 0.9)

    train = rows[:split]
    val = rows[split:]

    pd.DataFrame(train).to_parquet(
        OUT_DIR / "train.parquet",
        index=False,
    )

    pd.DataFrame(val).to_parquet(
        OUT_DIR / "validation.parquet",
        index=False,
    )

    manifest = {
        "total": len(rows),
        "train": len(train),
        "validation": len(val),
        "coordinate_system": "[0,1000]",
        "bbox_format": "[x1,y1,x2,y2]",
        "seed": 42,
    }

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
