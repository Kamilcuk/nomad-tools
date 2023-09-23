#!/bin/bash
# run integration tests
r() {
	echo "+ $*" >&2
	"$@"
}
trap_exit() {
	r nomad status
	r docker ps
}
trap 'trap_exit' exit
r pytest -sxv tests/integration "$@"
