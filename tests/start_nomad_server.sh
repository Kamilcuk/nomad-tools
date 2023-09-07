#!/bin/bash
set -xeuo pipefail

{
	# shellcheck disable=2024
	nomad agent -dev -log-level=WARN > >(sed 's/.*/\x1b[35mNOMAD: &\x1b[0m/') 2>&1
} &

# auto kill nomad after max 10 minutes.
child=$!
(
	sleep $((10 * 60))
	kill "$child"
) &

# Wait for nomad to be ready
wait_for_nomad() {
	while ! nomad status; do
		sleep 0.5
	done
}
export -f wait_for_nomad
timeout -v 10 bash -c wait_for_nomad
nomad status
