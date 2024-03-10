from tests.testlib import run_nomadt


def test_nomadt_help():
    run_nomadt("-h")
    run_nomadt("--help")
    run_nomadt("watch --help")
    run_nomadt("port --help")
    run_nomadt("cp --help")
