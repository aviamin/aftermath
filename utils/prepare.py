def split_storms(storms: list, test_storm: str) -> tuple:
    train = [s for s in storms if s != test_storm]
    test = [s for s in storms if s == test_storm]
    return train, test


def manifest_row(pre_path: str, post_path: str, damage_class: str, storm: str, split: str) -> dict:
    return {
        "pre_path": pre_path,
        "post_path": post_path,
        "damage_class": damage_class,
        "storm": storm,
        "split": split,
    }
