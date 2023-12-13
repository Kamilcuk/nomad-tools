import pytest

from nomad_tools.nomad_vardir import human_size


def test_units():
    assert human_size("1") == 1
    assert human_size("1K") == 1024
    assert human_size("1M") == 1024 * 1024
    assert human_size(" 1 M") == 1024 * 1024
    assert human_size(" 0.5 M") == 0.5 * 1024 * 1024
    with pytest.raises(ValueError):
        human_size("1X")
    with pytest.raises(ValueError):
        human_size("1KB")
    with pytest.raises(ValueError):
        human_size("1KM")
