from tests.testlib import run_nomadt


def test_go():
    run_nomadt(
        "go --rm busybox:stable sh -c 'echo hello world | md5sum'",
        output=["6f5902ac237024bdd0c176cb93063dc4"],
    )
