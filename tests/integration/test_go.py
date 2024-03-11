from tests.testlib import run_nomadt


def test_go_hello_world():
    run_nomadt(
        "go --rm busybox:stable sh -c 'echo hello world'",
        output=["hello world"],
    )


def test_go_hello_world_md5():
    run_nomadt(
        "go --rm busybox:stable sh -c 'echo hello world | md5sum'",
        output=["6f5902ac237024bdd0c176cb93063dc4"],
    )


def test_go_exit_24():
    run_nomadt("go --rm busybox:stable sh -xc 'exit 24'", check=24)


def test_go_exit_2():
    run_nomadt("go --rm busybox:stable sh -xc 'exit 2'", check=2)
