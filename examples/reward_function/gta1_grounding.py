import json
import math
import re
from typing import Any


REWARD_NAME = "gta1_grounding"
REWARD_TYPE = "batch"


COORD_RE = re.compile(
    r"^\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]\s*$"
)


MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


def parse_coordinate(text: str):
    match = COORD_RE.match(text.strip())
    if not match:
        return None

    return float(match.group(1)), float(match.group(2))


def processed_size(width: int, height: int):
    area = width * height

    if area > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / area)
        width = int(width * scale)
        height = int(height * scale)

    elif area < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / area)
        width = int(width * scale)
        height = int(height * scale)

    return width, height


def compute_score(
    reward_inputs: list[dict[str, Any]],
) -> list[dict[str, float]]:

    scores = []

    for item in reward_inputs:
        response = item.get("response", "")
        ground_truth = item.get("ground_truth", "")

        try:
            gt = json.loads(ground_truth)

            bbox = gt["bbox"]
            original_width, original_height = gt["image_size"]

            coord = parse_coordinate(response)

            if coord is None:
                scores.append(
                    {
                        "overall": 0.0,
                        "accuracy": 0.0,
                        "parse_success": 0.0,
                    }
                )
                continue

            x, y = coord

            proc_width, proc_height = processed_size(
                original_width,
                original_height,
            )

            scale_x = proc_width / original_width
            scale_y = proc_height / original_height

            original_x = x / scale_x
            original_y = y / scale_y

            x1, y1, x2, y2 = bbox

            success = (
                x1 <= original_x <= x2
                and y1 <= original_y <= y2
            )

            score = 1.0 if success else 0.0

            scores.append(
                {
                    "overall": score,
                    "accuracy": score,
                    "parse_success": 1.0,
                }
            )

        except Exception:
            scores.append(
                {
                    "overall": 0.0,
                    "accuracy": 0.0,
                    "parse_success": 0.0,
                }
            )

    return scores
