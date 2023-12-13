#!/bin/bash
r() {
	echo "+ $*" >&2
	"$@"
}
r pytest -sxv tests/unit "$@"
