from src.experiment.records import config_fingerprint


def test_config_fingerprint_is_stable_for_key_order():
    left = {"b": 2, "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": 2}

    assert config_fingerprint(left) == config_fingerprint(right)
