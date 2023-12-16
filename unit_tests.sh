#!/bin/bash
r() {
	echo "+ $*" >&2
	time "$@"
}
r pytest -sxv tests/unit "$@"
