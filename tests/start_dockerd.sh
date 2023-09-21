#!/bin/bash
set -xeuo pipefail

[[ -e /.dockerenv || -v GITHUB_ACTION ]]
[[ ! -e /var/run/docker.sock ]]

dockerd &

# Wait for docker to be ready
wait_for_docker() {
	while ! docker info; do
		sleep 0.5
	done
}
export -f wait_for_docker
timeout -v 10 bash -c wait_for_docker
