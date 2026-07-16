from dataclasses import dataclass
import json

from shapely import wkt as shapely_wkt

DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


@dataclass
class BuildingLabel:
    uid: str
    polygon: list[tuple[float, float]]
    damage_class: str | None


def parse_label_file(json_path: str) -> list[BuildingLabel]:
    """Parse an xBD label JSON (pre- or post-disaster) into BuildingLabels.

    Post-disaster label files annotate each building feature with a
    ``subtype`` (one of ``DAMAGE_CLASSES``, or ``"un-classified"`` for
    buildings the annotators could not assess); ``"un-classified"``
    buildings are dropped since they carry no usable training label.

    Pre-disaster label files only contain building footprints — real xBD
    pre-disaster features have no ``subtype`` key at all, since damage is
    inherently undefined before the disaster. Those buildings are kept
    (with ``damage_class=None``) so callers can still match them to their
    post-disaster counterpart by ``uid`` and crop the pre-disaster image.
    """
    with open(json_path) as f:
        data = json.load(f)

    labels = []
    for feature in data["features"]["xy"]:
        properties = feature["properties"]
        subtype = properties.get("subtype")
        if subtype is not None and subtype not in DAMAGE_CLASSES:
            continue
        geometry = shapely_wkt.loads(feature["wkt"])
        polygon = list(geometry.exterior.coords)
        labels.append(
            BuildingLabel(
                uid=properties["uid"],
                polygon=polygon,
                damage_class=subtype,
            )
        )
    return labels
