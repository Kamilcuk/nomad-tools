#!/bin/bash
set -xeuo pipefail

if ((UID == 0)) || ! hash sudo 2>/dev/null; then
	# Disable sudo for root
	sudo() {
		"$@"
	}
fi

if ! hash nomad 2>/dev/null; then
	if ! hash wget 2>/dev/null || ! hash unzip 2>/dev/null; then
		sudo apt-get update -y
		sudo apt-get install -y --no-install-recommends wget unzip
	fi
	if [[ ! -e /tmp/nomad.zip ]]; then
		v=1.6.2
		wget -O /tmp/nomad.zip "https://releases.hashicorp.com/nomad/${v}/nomad_${v}_linux_amd64.zip"
	fi
	sudo unzip /tmp/nomad.zip -d /usr/local/bin
	sudo chmod +x /usr/local/bin/nomad
fi
