#!/bin/bash
set -xeuo pipefail

fatal() {
	echo "startscript: FATAL: $*" >&2
	exit 123
}

warn() {
	echo "startscript: WARNING: $*" >&2
}

info() {
	echo "startscript: INFO: $*" >&2
}

choosedir() {
	local dir i ret=0
	dir="$1"
	shift
	if ((!$#)); then fatal "missing arguments"; fi
	for ((i = 0; ; i++)); do
		mkdir -vp "$dir/$i"
		{
			if flock -n 249; then
				info "Using $dir/$i directory"
				"$@" "$dir/$i" || ret=$?
				return $ret
			fi
		} 249>"$dir/$i/lockfile"
	done
}

dir_is_empty() {
	[[ -n "$(ls -A "$1")" ]]
}

configure_dockerd() {
	local dir
	dir="$1"
	dir=$(readlink -f "$dir/docker")
	if (($#)); then fatal "too many args"; fi
	if [[ ! -d /var/lib/docker ]]; then
		info "Dockerd using $dir"
		ln -nvfs "$dir" /var/lib/docker/
	else
		warn "Skipping dockerd configuratin: /var/lib/docker already exists"
	fi
}

configure_runner() {
	local dir tmp
	dir="$1"
	#
	tmp=$(readlink -f "$dir/run")
	mkdir -vp "$tmp"
	export RUNNER_WORKDIR="$tmp"
	info "RUNNER_WORKDIR=$RUNNER_WORKDIR"
	#
	tmp=$(readlink -f "$dir/hostedtoolcache")
	if [[ -e /opt/hostedtoolcache/ ]]; then
		rmdir -v /opt/hostedtoolcache/
	fi
	mkdir -vp "$tmp"
	ln -nvfs "$tmp" /opt/hostedtoolcache/
	export AGENT_TOOLSDIRECTORY="$tmp"
	info "AGENT_TOOLSDIRECTORY=$tmp | /opt/hostedtoolcache -> $tmp"
}

run() {
	local dir
	dir="$1"
	configure_dockerd "$dir"
	configure_runner "$dir"
	exec /entrypoint.sh ./bin/Runner.Listener run --startuptype service
}

if [[ ! -e /.dockerenv ]]; then fatal "Not in docker"; fi
run /_work
