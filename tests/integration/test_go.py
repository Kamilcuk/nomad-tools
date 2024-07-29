from tests.testlib import run_nomadt

# These tests fail randomly because Nomad is too slow to transfer logs.
# Proper synchronization is somewhere between "can't be done" and TODO.
sync = "echo;echo;echo;echo;echo;echo;sleep 5"


def test_go_hello_world():
    run_nomadt(
        f"go --rm busybox:stable sh -c 'echo hello world;{sync}'",
        output=["hello world"],
    )


def test_go_hello_world_md5():
    run_nomadt(
        f"go --rm busybox:stable sh -c 'echo hello world | md5sum;{sync}'",
        output=["6f5902ac237024bdd0c176cb93063dc4"],
    )


def test_go_exit_24():
    run_nomadt("go --rm busybox:stable sh -xc 'exit 24'", check=24)


def test_go_exit_2():
    run_nomadt("go --rm busybox:stable sh -xc 'exit 2'", check=2)


def test_go_identy():
    run_nomadt(
        """
        go --rm
        --identity '{"name": "example", "aud": ["oidc.example.com"], "file": true, "change_mode": "signal", "change_signal": "SIGHUP", "TTL": "1h"}'
        --driver raw_exec echo hello
        """
    )


def test_go_identities():
    run_nomadt(
        """
        go --rm
        --identity '{"name": "example", "aud": ["oidc.example.com"], "file": true, "change_mode": "signal", "change_signal": "SIGHUP", "TTL": "1h"}'
        --identity '{"name": "example", "aud": ["oidc.example.com"], "file": true, "change_mode": "signal", "change_signal": "SIGHUP", "TTL": "1h"}'
        --identity '{"name": "example", "aud": ["oidc.example.com"], "file": true, "change_mode": "signal", "change_signal": "SIGHUP", "TTL": "1h"}'
        --driver raw_exec echo hello
        """
    )
