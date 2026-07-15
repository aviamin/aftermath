from utils.prepare import split_storms, manifest_row


def test_split_storms_holds_out_test_storm():
    storms = ["hurricane-harvey", "hurricane-florence", "hurricane-matthew", "hurricane-michael"]

    train, test = split_storms(storms, test_storm="hurricane-michael")

    assert train == ["hurricane-harvey", "hurricane-florence", "hurricane-matthew"]
    assert test == ["hurricane-michael"]


def test_manifest_row_contains_expected_fields():
    row = manifest_row(
        pre_path="data/processed/pre/1.png",
        post_path="data/processed/post/1.png",
        damage_class="major-damage",
        storm="hurricane-harvey",
        split="train",
    )

    assert row == {
        "pre_path": "data/processed/pre/1.png",
        "post_path": "data/processed/post/1.png",
        "damage_class": "major-damage",
        "storm": "hurricane-harvey",
        "split": "train",
    }
