#!/bin/bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
r() {
	echo "+ $*" >&2
	"$@"
}
cleanuplogs() {
	rm -f build/tests/*.log
}
logs_errors() {
	local files gfiles
	files=(build/tests/*.log)
	if ((${#files} == 0)); then
		echo "No tests log files found in build/tests/*.log"
		return
	fi
	gfiles=$(grep -l '=================================== FAILURES ===================================' "${files[@]}")
	if [[ -z "${gfiles}" ]]; then
		echo "No errors in logs found" >&2
		return
	fi
	# shellcheck disable=2086
	tail -n +1 $gfiles || :
}
remove_nomad_from_path() {
	# Remove nomad executable from path completely.
	bind="$PWD"/build/bin
	mkdir -vp "$bind"
	trap 'unit_tests_trap_exit' exit
	printf "%s\n" "#!/bin/sh" "exit 1" >"$bind"/nomad
	chmod +x "$bind"/nomad
	cp "$bind"/nomad "$bind"/docker
	cp "$bind"/nomad "$bind"/consul
	export PATH="$bind:$PATH"
	unset NOMAD_TOKEN
	export NOMAD_ADDR=moon
}
unit_tests_trap_exit() {
	logs_errors || :
}
unit_tests() {
	# No connection to Nomad for unit tests is allowed.
	remove_nomad_from_path
	r pytest -sxv tests/unit "$@"
}
integration_tests() {
	trap 'integration_tests_trap_exit' exit
	r pytest -sxv tests/unit tests/integration "$@"
}
integration_tests_trap_exit() {
	r nomad status || :
	r docker ps || :
	logs_errors || :
}
be_nice() {
	if [[ -v CI ]]; then
		# Ignore in CI/CD
		return
	fi
	r renice -n 40 $BASHPID
	r ionice -c 3 -p $BASHPID
}

###############################################################################

# Executed for both unit test and integration tests
be_nice
cleanuplogs
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
	# If not sourced, execute unit tests
	unit_tests "$@"
fi
