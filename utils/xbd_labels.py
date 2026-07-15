from dataclasses import dataclass
import json

from shapely import wkt as shapely_wkt

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


@dataclass
class BuildingLabel:
    uid: str
    polygon: list[tuple[float, float]]
    damage_class: str


def parse_label_file(json_path: str) -> list[BuildingLabel]:
    with open(json_path) as f:
        data = json.load(f)

    labels = []
    for feature in data["features"]["xy"]:
        subtype = feature["properties"]["subtype"]
        if subtype not in DAMAGE_CLASSES:
            continue
        geometry = shapely_wkt.loads(feature["wkt"])
        polygon = list(geometry.exterior.coords)
        labels.append(
            BuildingLabel(
                uid=feature["properties"]["uid"],
                polygon=polygon,
                damage_class=subtype,
            )
        )
    return labels


def compute_class_weights(labels: list[BuildingLabel]) -> dict[str, float]:
    counts = {cls: 0 for cls in DAMAGE_CLASSES}
    for label in labels:
        counts[label.damage_class] += 1

    total = sum(counts.values())
    num_classes = len(DAMAGE_CLASSES)
    weights = {}
    for cls in DAMAGE_CLASSES:
        count = max(counts[cls], 1)
        weights[cls] = total / (num_classes * count)
    return weights
