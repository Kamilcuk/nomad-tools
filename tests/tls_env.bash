#!/bin/bash
_DIR="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/nomad.d"
_r() {
	echo "+ $*" >&2
	"$@"
}
case "$1" in
clear)
	_r unset NOMAD_ADDR
	_r unset NOMAD_TLS_SERVER_NAME
	_r unset NOMAD_CACERT
	_r unset NOMAD_CAPATH
	_r unset NOMAD_CLIENT_CERT
	_r unset NOMAD_CLIENT_KEY
	;;
tls)
	_r export NOMAD_ADDR=https://localhost:4646
	_r unset NOMAD_TLS_SERVER_NAME
	_r export NOMAD_CACERT=$_DIR/nomad-agent-ca.pem
	_r unset NOMAD_CAPATH
	_r export NOMAD_CLIENT_CERT=$_DIR/global-cli-nomad.pem
	_r export NOMAD_CLIENT_KEY=$_DIR/global-cli-nomad-key.pem
	;;
capath)
	_r export NOMAD_ADDR=https://localhost:4646
	_r unset NOMAD_TLS_SERVER_NAME
	_r unset NOMAD_CACERT
	_r export NOMAD_CAPATH=$_DIR/capath
	_r export NOMAD_CLIENT_CERT=$_DIR/global-cli-nomad.pem
	_r export NOMAD_CLIENT_KEY=$_DIR/global-cli-nomad-key.pem
	;;
sni)
	_r export NOMAD_ADDR=https://127.0.0.1:4646
	_r export NOMAD_TLS_SERVER_NAME=localhost
	_r export NOMAD_CACERT=$_DIR/nomad-agent-ca.pem
	_r unset NOMAD_CAPATH
	_r export NOMAD_CLIENT_CERT=$_DIR/global-cli-nomad.pem
	_r export NOMAD_CLIENT_KEY=$_DIR/global-cli-nomad-key.pem
	;;
test)
	_r nomad status
	_r nomadtools vardir -j nginx@nginx ls
	;;
*)
	echo "${BASH_SOURCE[0]}: invalid argument: $1" >&2
	echo "${BASH_SOURCE[0]}: must be: clear tls sni" >&2
	;;
esac
unset -f _r
unset _DIR
