import datetime

from nomad_tools.nomadlib.types import fromisoformat


def test_fromisoformat():
    assert fromisoformat(
        "2023-10-26T13:53:40.577959875Z"
    ) == datetime.datetime.fromisoformat("2023-10-26T13:53:40.577959+00:00")
