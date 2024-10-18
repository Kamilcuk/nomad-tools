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

choose_cachedir() {
	local dir
	dir="$1"
	shift
	if (($#)); then fatal "too many arguments"; fi
	#
	info "Picking cache directory inside $dir"
	declare -g CACHEDIR
	local i
	for ((i = 0; ; i++)); do
		r mkdir -vp "$dir/$i"
		exec 249>"$dir/$i/lockfile"
		if flock --verbose -n 249; then
			info "Using $dir/$i as cache directory"
			# When was the directory last used? See timestamp of lockfile.
			touch "$dir/$i/lockfile" 
			CACHEDIR="$dir/$i"
			break
		fi
		exec 249>&-
	done
}

dockerd_use_dir() {
	local dir
	dir="$1"
	shift
	if (($#)); then fatal "too many args"; fi
	#
	info "Configuring docker daemon to use $dir/docker directory"
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

github_runner_use_dir() {
	local dir
	dir="$1"
	shift
	if (($#)); then fatal "too many args"; fi
	#
	local tmp
	info "Moving Github runner cache to $dir/hostedtoolscache"
	tmp=$(readlink -f "$dir/hostedtoolcache")
	if [[ -e /opt/hostedtoolcache/ ]]; then
		r rmdir -v /opt/hostedtoolcache/
	fi
	r mkdir -vp "$tmp"
	r ln -nvfs "$tmp" /opt/hostedtoolcache
	r export AGENT_TOOLSDIRECTORY="$tmp"
	#
	info "Moving Github runner workdir to $dir/configure"
	tmp=$(readlink -f "$dir/configure")
	r mkdir -vp "$tmp"
	r export RUNNER_WORKDIR="$tmp"
}

###############################################################################
# main

{
	if (($#)); then fatal "too many args"; fi
	if [[ ! -e /.dockerenv ]]; then fatal "Not in docker"; fi
	if mountpoint /_work; then
		choose_cachedir /_work
		dockerd_use_dir "$CACHEDIR"
		github_runner_use_dir "$CACHEDIR"
	else
		info "/_work is not a mountpoint"
	fi
	info 'start github runner'
} 2>&1
# synchronize with https://github.com/myoung34/docker-github-actions-runner/blob/master/Dockerfile#L25
r exec /entrypoint.sh ./bin/Runner.Listener run --startuptype service "$@"
