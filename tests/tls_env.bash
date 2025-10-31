#!/bin/bash
_DIR="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/nomad.d"
_r() {
	echo "+ $*" >&2
	"$@"
}
_clear_envs() {
	envs=$(compgen -v NOMAD_)
	if [[ -n "$envs" ]]; then
		# shellcheck disable=2086
		_r unset $envs
	fi
}
while (($#)); do
	case "$1" in
	clear)
		_clear_envs
		;;
	tls)
		_clear_envs
		_r export NOMAD_ADDR=https://localhost:4646
		_r unset NOMAD_TLS_SERVER_NAME
		_r export NOMAD_CACERT="$_DIR"/nomad-agent-ca.pem
		_r unset NOMAD_CAPATH
		_r export NOMAD_CLIENT_CERT="$_DIR"/global-cli-nomad.pem
		_r export NOMAD_CLIENT_KEY="$_DIR"/global-cli-nomad-key.pem
		;;
	capath)
		_clear_envs
		_r export NOMAD_ADDR=https://localhost:4646
		_r unset NOMAD_TLS_SERVER_NAME
		_r unset NOMAD_CACERT
		_r export NOMAD_CAPATH="$_DIR"
		_r export NOMAD_CLIENT_CERT="$_DIR"/global-cli-nomad.pem
		_r export NOMAD_CLIENT_KEY="$_DIR"/global-cli-nomad-key.pem
		;;
	sni)
		_clear_envs
		_r export NOMAD_ADDR=https://127.0.0.1:4646
		_r export NOMAD_TLS_SERVER_NAME=localhost
		_r export NOMAD_CACERT="$_DIR"/nomad-agent-ca.pem
		_r unset NOMAD_CAPATH
		_r export NOMAD_CLIENT_CERT="$_DIR"/global-cli-nomad.pem
		_r export NOMAD_CLIENT_KEY="$_DIR"/global-cli-nomad-key.pem
		;;
	test)
		_r export NOMAD_NAMESPACE=default
		_r nomad status
		_r nomad-tools vardir -j nginx@nginx ls
		_r python -m nomad_tools.taskexec test-forever test-forever echo Hello world
		;;
	testall)
		_r "$0" tls -- nomad-tools watch start "$_DIR"/../../jobs/test-forever.nomad.hcl
		echo
		_r "$0" tls test
		echo
		_r "$0" capath test
		echo
		_r "$0" sni test
		;;
	--)
		shift
		_r exec "$@"
		;;
	*)
		echo "Unknown arguments: $*" >&2
		exit 123
		;;
	esac
	shift
done
