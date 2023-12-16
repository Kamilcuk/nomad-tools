#!/bin/bash
set -xeuo pipefail

log() {
	sed $'s/.*/\x1b[35m'"$1"$': &\x1b[0m/'
}

fatal() {
	echo "ERROR: $*" >&2
	exit 2
}

nomad_install() {
	if ! hash nomad 2>/dev/null; then
		# pin to version 1.6.3
		nomad-downloadrelease -p 1.6.3 nomad /usr/local/bin/nomad
		if [[ ! -e /opt/cni/bin && -e /usr/lib/cni/ ]]; then
			mkdir -vp /opt/cni
			ln -vs /usr/lib/cni/ /opt/cni/bin
		fi
	fi
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
nomad_install | nomad_start | nomad_restart | vagrant)
	"$@"
	;;
*)
	fatal ""
	;;
esac
