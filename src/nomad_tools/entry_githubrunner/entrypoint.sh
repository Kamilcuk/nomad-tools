#!/bin/bash
# shellcheck disable=2120
set -euo pipefail

: "${ENTRYPOINT_CLEANUP_FREE_BELOW_GB:=20}"
: "${ENTRYPOINT_CLEANUP_USAGE_PERCENT:=95}"

if [[ -n "${DEBUG:-}" ]]; then
	set -x
fi

_log() {
	echo "nomadtools: $1: ${FUNCNAME[2]:-main}:" "${@:2}"
}

fatal() {
	_log FATAL "$*"
	exit 123
}

error() {
	_log ERROR "$*"
}

warn() {
	_log WARNING "$*"
}

info() {
	_log INFO "$*"
}

r() {
	_log INFO "+ $*"
	"$@"
}

###############################################################################
# cache picking

# Cache structure documentation:
# - cache is created inside /_work directory, if /_work is mountpoint
# - /_work/lockfile serves as a global lockfile
#   - File descriptor 250 is used
# - /_work/c[0-9]* are cache directories
# - /_work/c[0-9]*/lockfile is the lockfile of a specific cache.
#   - If the lockfile is locked, this cache directory is in use.
#   - File descriptor 249 is used

should_cleanup() {
	local df
	if ! df=$(df -k "$dir"); then
		return 1
	fi
	local  freekb usedpercent
	{
		read -r _
		read -r _ _ _ freekb usedpercent _
	} <<<"$df"
	usedpercent=${usedpercent%%%}  # remove percent sign
	local freegb
	freegb=$((freekb / 1024 / 1024))
	info "disc=$dir freegb=${freegb} used=${usedpercent}%"
	# While free space is below threshold or used percent is above used threshold.
	info "Checking disc usage (( $freegb < $ENTRYPOINT_CLEANUP_USAGE_PERCENT || $usedpercent >= $ENTRYPOINT_CLEANUP_USAGE_PERCENT ))"
 	((freegb < ENTRYPOINT_CLEANUP_FREE_BELOW_GB || usedpercent >= ENTRYPOINT_CLEANUP_USAGE_PERCENT))
}

cleanup_locked() {
	local dir
	dir="$1"
	shift
	if (($#)); then fatal "too many arguments"; fi
	#
	while should_cleanup; do
 		local cachedirs
		cachedirs=("$dir"/c[0-9]*)
		if ((${#cachedirs[@]} == 0)); then
			break
		fi
		for ((i=${#cachedirs[@]} - 1; i >= 0; --i)); do
			local cachedir
			cachedir=${cachedirs[$i]}
 			info "Picking cachedir to reduce disc usage, checking $cachedir"
			{
				if flock --verbose -n 249; then
					warn "Removing $cachedir to reduce disc usage"
					r rm --one-file-system -rf "$i"
					break
				fi
			} 249>"$cachedir"/lockfile
		done
	done
	info "Disc usage ok"
}

cleanup() {
	local dir
	dir="$NOMADTOOLS_CACHE"
	if (($#)); then fatal "too many arguments"; fi
	#
	{
		info "Acquiring global cachedir lock $dir/lockfile"
		if ! flock --verbose 250; then
			error "Could not flock $dir/lockfile"
			return
		fi
		if ! cleanup_locked "$dir"; then
			error "There was error executing cleanup on $dir"
			return
		fi
	} 250>"$dir/lockfile"
}

cleanup_mounted() {
	info "Executing"
	local dir
	dir="$NOMADTOOLS_CACHE"
	r killall containerd || true
	r killall dockerd || true
	r killall Runner.listener || true
	r ls -la "$dir"
	if should_cleanup; then
		info "Removing all files except lockfile"
		for i in "$dir"/* "$dir"/.[^.]*; do
			if [[ "$i" != lockfile ]]; then
				r rm --one-file-system -rf "$i"
			fi
		done
	fi
}

choose_cachedir() {
	local dir
	dir="$1"
	shift
	if (($#)); then fatal "too many arguments"; fi
	#
	info "Picking cache directory inside $dir"
	r ls -lah "$dir"
	{
		info "Acquiring global cachedir lock $dir/lockfile"
		if ! flock --verbose 250; then
			error "Could not flock $dir/lockfile"
			return
		fi
		if ! cleanup_locked "$dir"; then
			error "There was error executing cleanup on $dir"
			return
		fi
		local i
		for ((i = 0; ; i++)); do
			r mkdir -vp "$dir/c$i"
			exec 249>"$dir/c$i/lockfile"
			if flock --verbose -n 249; then
				break
			fi
			exec 249>&-
		done
	} 250>"$dir/lockfile"
	#
	# note: flock is held here
	info "Using $dir/c$i as cache directory"
	# note: hide other caches by mount binding.
	declare -g NOMADTOOLS_CACHE
	if r mount -obind "$dir/c$i" "$dir"; then
		NOMADTOOLS_CACHE="$dir"
		trap 'cleanup_mounted 2>&1' EXIT
	else
		NOMADTOOLS_CACHE="$dir/c$i"
		trap 'cleanup 2>&1' EXIT
	fi
	r ls -lah "$NOMADTOOLS_CACHE"
}

dockerd_use_dir() {
	local dir
	dir="$1"
	shift
	if (($#)); then fatal "too many args"; fi
	#
	local dockerdir
	dockerdir="$dir/docker"
	info "Configuring docker daemon to use $dockerdir directory"
	local dockersock
	dockersock=/var/run/docker.sock
	if [[ -e "$dockersock" ]]; then
		warn "Skipping dockerd configuration: $dockersock already exists"
		if ! r docker info; then
			warn "docker info failed"
		fi
	else
		info "Symlinking /var/lib/docker to $dockerdir"
		r mkdir -vp "$dockerdir"
		r ln -nvfs "$dockerdir" /var/lib/docker
	fi
}

github_runner_use_dir() {
	local dir
	dir="$1"
	shift
	if (($#)); then fatal "too many args"; fi
	#
	AGENT_TOOLSDIRECTORY="$dir/hostedtoolcache"
	info "Moving Github runner cache to $AGENT_TOOLSDIRECTORY"
	if [[ -e /opt/hostedtoolcache/ ]]; then
		r rmdir -v /opt/hostedtoolcache/
	fi
	r mkdir -vp "$AGENT_TOOLSDIRECTORY"
	r ln -nvfs "$AGENT_TOOLSDIRECTORY" /opt/hostedtoolcache
	r export AGENT_TOOLSDIRECTORY="$AGENT_TOOLSDIRECTORY"
	#
	RUNNER_WORKDIR="$dir/workdir"
	info "Moving Github runner workdir to $RUNNER_WORKDIR"
	r mkdir -vp "$RUNNER_WORKDIR"
	r export RUNNER_WORKDIR="$RUNNER_WORKDIR"
}

###############################################################################
# main

{
	if (($#)); then fatal "too many args"; fi
	if [[ ! -e /.dockerenv ]]; then fatal "Not in docker"; fi
	r renice -n 39 $BASHPID
	r ionice -c 3 -p $BASHPID
	if mountpoint /_work; then
		choose_cachedir /_work
		if [[ -n "${NOMADTOOLS_CACHE:-}" ]]; then
			dockerd_use_dir "$NOMADTOOLS_CACHE"
			github_runner_use_dir "$NOMADTOOLS_CACHE"
		fi
	else
		info "/_work is not a mountpoint"
	fi
	r export "NOMADTOOLS_CACHE=${NOMADTOOLS_CACHE:-/var/tmp}"
	info 'start github runner'
} 2>&1
# synchronize with https://github.com/myoung34/docker-github-actions-runner/blob/master/Dockerfile#L25
r /entrypoint.sh ./bin/Runner.Listener run --startuptype service "$@"

