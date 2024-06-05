#!/bin/bash
set -xeuo pipefail

log() {
	sed $'s/.*/\x1b[35m'"$1"$': &\x1b[0m/'
}

fatal() {
	echo "ERROR: $*" >&2
	exit 2
}

# shellcheck disable=SC2120
nomad_install() {
	local version
	version="${1:-1.6.3}"
	if ! hash nomad 2>/dev/null; then
		# pin to version 1.6.3
		python -m nomad_tools.entrypoint downloadrelease -p "$version" nomad /usr/local/bin/nomad
		if [[ ! -e /opt/cni/bin && -e /usr/lib/cni/ ]]; then
			mkdir -vp /opt/cni
			ln -vs /usr/lib/cni/ /opt/cni/bin
		fi
	fi
}

cni_install() {
	sudo mkdir -vp /opt/cni/bin
	wget -q https://github.com/containernetworking/plugins/releases/download/v1.5.0/cni-plugins-linux-amd64-v1.5.0.tgz
	sudo tar xafvp cni-plugins*.tgz -C /opt/cni/bin
	sudo rm cni-plugins*.tgz
}

nomad_start() {
	local pid now endtime
	if pid=$(pgrep nomad); then
		echo "nomad already running: $(xargs ps aux "$pid")"
		return
	fi
	sudo nomad agent -dev -config ./tests/nomad.hcl &
	NOMADPID=$!
	# Wait for nomad
	now=$(date +%s)
	endtime=$((now + 30))
	while ! nomad status >/dev/null; do
		sleep 0.5
		now=$(date +%s)
		if ((now > endtime)); then
			fatal "did not start nomad"
		fi
		if ! kill -0 "$NOMADPID"; then
			fatal "nomad exited"
		fi
	done
	nomad status
}

nomad_restart() {
	sudo pkill nomad || :
	sudo "$0" nomad_start
}

_nomad_tools_profile() {
	cd /app
	export PATH="/home/vagrant/.local:$PATH"
}

vagrant() {
	cd /app
	nomad_install
	nomad_start
	nomad -autocomplete-install || :
	sudo -uvagrant nomad -autocomplete-install || :
	apt-get install --no-install-recommends -y python3-pip python3-lib2to3 bash-completion make
	python3 --version
	python3 -m pip install --upgrade pip
	sudo -uvagrant python3 -m pip install --user -e ".[test]"
	sudo tee /etc/profile.d/my-nomad-tools.sh <<<"$(declare -f _nomad_tools_profile); _nomad_tools_profile"
}

case "$1" in
cni_install | nomad_install | nomad_start | nomad_restart | vagrant)
	"$@"
	;;
*)
	fatal ""
	;;
esac
