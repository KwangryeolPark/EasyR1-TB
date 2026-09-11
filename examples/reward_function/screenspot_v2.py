import json
import math
import re
from typing import Any


REWARD_NAME = "screenspot_v2"
REWARD_TYPE = "batch"


TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    flags=re.DOTALL,
)


def extract_coordinate(response: str):
    response = response.strip()

    match = TOOL_CALL_PATTERN.search(response)
    if match is None:
        return None

    try:
        payload = json.loads(match.group(1))
        if payload["name"] != "computer_use":
            return None

        args = payload["arguments"]
        if args["action"] != "left_click":
            return None

        coordinate = args["coordinate"]
        if len(coordinate) != 2:
            return None

        x = float(coordinate[0])
        y = float(coordinate[1])

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        return None

    return x, y


def point_to_bbox_distance(x, y, bbox):
    x1, y1, x2, y2 = bbox

    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)

    return math.sqrt(dx * dx + dy * dy)


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    outputs = []

    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth_raw = reward_input.get("ground_truth", "")

        try:
            gt = json.loads(ground_truth_raw)
            bbox = [float(v) for v in gt["bbox"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            outputs.append(
                {
                    "overall": 0.0,
                    "accuracy": 0.0,
                    "format": 0.0,
                    "proximity": 0.0,
                }
            )
            continue

        coordinate = extract_coordinate(response)

        if coordinate is None:
            outputs.append(
                {
                    "overall": 0.0,
                    "accuracy": 0.0,
                    "format": 0.0,
                    "proximity": 0.0,
                }
            )
            continue

        x, y = coordinate
        x1, y1, x2, y2 = bbox

        inside = x1 <= x <= x2 and y1 <= y <= y2

        if inside:
            accuracy = 1.0
            proximity = 1.0
            overall = 1.0
        else:
            accuracy = 0.0

            distance = point_to_bbox_distance(x, y, bbox)
            normalized_distance = distance / math.sqrt(1000.0**2 + 1000.0**2)

            proximity = math.exp(-8.0 * normalized_distance)
            overall = 0.5 * proximity

        outputs.append(
            {
                "overall": float(overall),
                "accuracy": float(accuracy),
                "format": 1.0,
                "proximity": float(proximity),
            }
        )

    return outputs