#!/bin/bash
r() {
	echo "+ $*" >&2
	time "$@"
}
# No connection to Nomad for unit tests is allowed.
unset NOMAD_TOKEN
if [[ -v NOMAD_ADDR ]]; then
	NOMAD_ADDR=moon
fi
# Remove nomad executable from path completely.
tmpd=$(mktemp -d)
trap 'rm -rf "$tmpd"' EXIT
printf "%s\n" "#!/bin/sh" "exit 1" >"$tmpd"/nomad
chmod +x "$tmpd"/nomad
export PATH="$tmpd:$PATH"
r pytest -sxv tests/unit "$@"
