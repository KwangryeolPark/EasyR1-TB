"""GroundCUA point-in-box reward, protocol-compatible with ScreenSpot-V2."""

import json
import math
import re
from typing import Any

REWARD_NAME = "groundcua"
REWARD_TYPE = "batch"
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def extract_coordinate(response: str):
    match = TOOL_CALL_PATTERN.search((response or "").strip())
    if match is None:
        return None
    try:
        call = json.loads(match.group(1))
        if call["name"] != "computer_use":
            return None
        arguments = call["arguments"]
        if arguments["action"] != "left_click":
            return None
        coordinate = arguments["coordinate"]
        if len(coordinate) != 2:
            return None
        x, y = float(coordinate[0]), float(coordinate[1])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        return None
    return x, y


def point_to_bbox_distance(x: float, y: float, bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Return the same overall/accuracy/format/proximity metrics as ScreenSpot-V2."""
    outputs = []
    for item in reward_inputs:
        coordinate = extract_coordinate(item.get("response", ""))
        try:
            truth = json.loads(item.get("ground_truth", ""))
            bbox = [float(value) for value in truth["bbox"]]
            if len(bbox) != 4:
                raise ValueError("bbox must contain four coordinates")
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            outputs.append({"overall": 0.0, "accuracy": 0.0, "format": 0.0, "proximity": 0.0})
            continue

        if coordinate is None:
            outputs.append({"overall": 0.0, "accuracy": 0.0, "format": 0.0, "proximity": 0.0})
            continue

        x, y = coordinate
        x1, y1, x2, y2 = bbox
        if x1 <= x <= x2 and y1 <= y <= y2:
            outputs.append({"overall": 1.0, "accuracy": 1.0, "format": 1.0, "proximity": 1.0})
            continue
        distance = point_to_bbox_distance(x, y, bbox)
        proximity = math.exp(-8.0 * distance / math.sqrt(1000.0**2 + 1000.0**2))
        outputs.append({"overall": 0.5 * proximity, "accuracy": 0.0, "format": 1.0, "proximity": proximity})

    return outputs
