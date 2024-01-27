#!/bin/bash
r() {
	echo "+ $*" >&2
	time "$@"
}
uset NOMAD_TOKEN
if [[ -v NOMAD_ADDR ]]; then
	NOMAD_ADDR=moon
fi
r pytest -sxv tests/unit "$@"
