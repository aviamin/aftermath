import json
import pytest
from utils.xbd_labels import DAMAGE_CLASSES, BuildingLabel, parse_label_file, compute_class_weights


def _write_label_json(tmp_path, features):
    label = {"features": {"xy": features}, "metadata": {}}
    path = tmp_path / "sample_post.json"
    path.write_text(json.dumps(label))
    return str(path)


def test_parse_label_file_extracts_damage_classes(tmp_path):
    features = [
        {
            "properties": {"uid": "a1", "feature_type": "building", "subtype": "no-damage"},
            "wkt": "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
        },
        {
            "properties": {"uid": "a2", "feature_type": "building", "subtype": "destroyed"},
            "wkt": "POLYGON ((20 20, 20 30, 30 30, 30 20, 20 20))",
        },
    ]
    path = _write_label_json(tmp_path, features)

    labels = parse_label_file(path)

    assert len(labels) == 2
    assert labels[0] == BuildingLabel(uid="a1", polygon=[(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0), (0.0, 0.0)], damage_class="no-damage")
    assert labels[1].damage_class == "destroyed"


def test_parse_label_file_drops_unclassified(tmp_path):
    features = [
        {
            "properties": {"uid": "b1", "feature_type": "building", "subtype": "un-classified"},
            "wkt": "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
        },
        {
            "properties": {"uid": "b2", "feature_type": "building", "subtype": "minor-damage"},
            "wkt": "POLYGON ((20 20, 20 30, 30 30, 30 20, 20 20))",
        },
    ]
    path = _write_label_json(tmp_path, features)

    labels = parse_label_file(path)

    assert len(labels) == 1
    assert labels[0].uid == "b2"


def test_parse_label_file_keeps_buildings_missing_subtype(tmp_path):
    # Real xBD pre-disaster label files omit "subtype" entirely (damage is
    # undefined before the disaster) — these buildings must still be
    # returned (with damage_class=None) so they can be matched by uid.
    features = [
        {
            "properties": {"uid": "c1", "feature_type": "building"},
            "wkt": "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))",
        },
    ]
    path = _write_label_json(tmp_path, features)

    labels = parse_label_file(path)

    assert len(labels) == 1
    assert labels[0].uid == "c1"
    assert labels[0].damage_class is None


def test_compute_class_weights_favors_rare_classes():
    labels = (
        [BuildingLabel(uid=f"n{i}", polygon=[], damage_class="no-damage") for i in range(80)]
        + [BuildingLabel(uid=f"d{i}", polygon=[], damage_class="destroyed") for i in range(4)]
    )

    weights = compute_class_weights(labels)

    assert set(weights.keys()) == set(DAMAGE_CLASSES)
    assert weights["destroyed"] > weights["no-damage"]
