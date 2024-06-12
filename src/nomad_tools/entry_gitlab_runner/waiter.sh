#!/bin/sh
set -eu

log() {
	echo "$*"
}

fatal() {
	echo "ERROR: $*" >&2
	exit 2
}

waitfor() {
	desc=$1
	shift
	while ! "$@"; do
		log "Waiting for $desc"
		if [ "$timeout_s" -ge 0 ]; then
			# timeout_s lower than 0 means infinite timeout.
			now=$(date +%s)
			if [ "$now" -gt "$stoptime" ]; then
				fatal "timeout of $timeout_s has expired"
			fi
		fi
		# Polling interval.
		sleep 1
	done
}

container_checker() {
	inspect=$(docker container inspect "$container")
}

ports_checker() {
	for port in $ports; do
		if busybox nc -v -z "$ip" "$port"; then
			log "container $container is open on $ip:$port"
			return
		fi
	done
	return 1
}

main() {
	if [ "$#" -eq 0 ]; then
		cat <<EOF
Usage: NOMAD_ALLOC_ID=allocid $(basename "$0") timeout_s services...

Wait for ports of docker services.
EOF
		exit 1
	fi
	if [ "$#" -eq 1 ]; then
		fatal "wrong number of arguments: $*"
	fi
	timeout_s="$1"
	shift
	now=$(date +%s)
	stoptime=$((now + timeout_s))
	for task in "$@"; do
		container="$task-$NOMAD_ALLOC_ID"

		# Wait for container to exists
		waitfor "container $container" container_checker

		# Get container ip address.
		ip=$(docker container inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container")
		if [ -z "$ip" ]; then
			ip=$(hostname -i)
		fi
		if [ -z "$ip" ]; then
			fatal "cotainer $container has no ip: $inspect"
		fi

		# Extract ports from container
		ports=$(
			docker container inspect -f '{{range $k, $v := .Config.ExposedPorts}}{{$k}}{{"\n"}}{{end}}' "$container" |
				sed -n 's@^\([0-9]*\)/.*@\1@p' | sort -u | paste -sd ' '
		)
		if [ -z "$ports" ]; then
			fatal "container $container has no ports"
		fi

		# Wait for any of the ports to be open.
		waitfor "$container on $ip for any port from $ports" ports_checker
	done
}

main "$@"
