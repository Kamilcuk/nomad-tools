#!/bin/bash
set -euo pipefail

if [[ "${DEBUG:-}" ]]; then
	set -x
fi

fatal() {
	echo "startscript: FATAL: $*"
	exit 123
}

warn() {
	echo "startscript: WARNING: $*"
}

info() {
	echo "startscript: INFO: $*"
}

r() {
	echo "+ $*"
	"$@"
}

choosedirin() {
	local dir
	dir="$1"
	shift
	if ((!$#)); then fatal "missing arguments"; fi
	#
	local i ret=0
	for ((i = 0; ; i++)); do
		r mkdir -vp "$dir/$i"
		{
			if flock -n 249; then
				info "Using $dir/$i directory"
				info "+ $* $dir/$i"
				r "$@" "$dir/$i" || ret=$?
				return "$ret"
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
	shift
	if (($#)); then fatal "too many args"; fi
	#
	dir=$(readlink -f "$dir/docker")
	local dockersock
	dockersock=/var/run/docker.sock
	if [[ -e $dockersock ]]; then
		warn "Skipping dockerd configuration: $dockersock already exists"
		if ! r docker info; then
			warn "docker info failed"
		fi
	else
		info "Symlinking docker to $dir"
		r mkdir -vp "$dir"
		r ln -nvfs "$dir" /var/lib/docker
	fi
}

configure_runner() {
	local dir
	dir="$1"
	shift
	if (($#)); then fatal "too many args"; fi
	#
	local tmp
	tmp=$(readlink -f "$dir/configure")
	r mkdir -vp "$tmp"
	r export RUNNER_WORKDIR="$tmp"
	info "RUNNER_WORKDIR=$RUNNER_WORKDIR"
	#
	tmp=$(readlink -f "$dir/hostedtoolcache")
	if [[ -e /opt/hostedtoolcache/ ]]; then
		r rmdir -v /opt/hostedtoolcache/
	fi
	r mkdir -vp "$tmp"
	r ln -nvfs "$tmp" /opt/hostedtoolcache
	r export AGENT_TOOLSDIRECTORY="$tmp"
	info "AGENT_TOOLSDIRECTORY=$tmp | /opt/hostedtoolcache -> $tmp"
}

configure() {
	local dir
	dir="$1"
	shift
	if (($#)); then fatal "too many args"; fi
	#
	configure_dockerd "$dir"
	configure_runner "$dir"
}

save_stdout_in_fd3_and_redirect_stdout_stderr() {
	exec 3>&1 1>&2
}

restore_stdout_from_fd3() {
	exec 1>&3 3>&-
}

save_stdout_in_fd3_and_redirect_stdout_stderr
if (($#)); then fatal "too many args"; fi
if [[ ! -e /.dockerenv ]]; then fatal "Not in docker"; fi
if mountpoint /_work; then
	choosedirin /_work configure
else
	info "/_work is not a mountpoint"
fi
info 'start github runner'
restore_stdout_from_fd3
# synchronize with https://github.com/myoung34/docker-github-actions-runner/blob/master/Dockerfile#L25
r exec /entrypoint.sh ./bin/Runner.Listener run --startuptype service
