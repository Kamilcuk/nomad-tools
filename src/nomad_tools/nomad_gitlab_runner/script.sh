#!/bin/sh
fatal() {
	echo "nomad-gitlab-runner:script.sh: ERROR: $*" >&2
	exit 1
}
# Reduce niceness
if hash ionice 2>/dev/null; then ionice -c 3 -p $$; fi
if hash renice 2>/dev/null; then renice -n 40 $$ >/dev/null; fi
if hash chrt 2>/dev/null; then chrt -i -p 0 $$; fi
# Check if all variables are fine.
for i in \
	NOMAD_META_CI_DRIVER \
	NOMAD_META_CI_RUNUSER \
	NOMAD_META_CI_JOB_URL \
	NOMAD_META_CI_PROJECT_URL \
	NOMAD_META_CI_OOM_SCORE_ADJUST \
	NOMAD_META_CI_CPUSET_CPUS; do
	if eval "[ -z \"\${$i+x}\" ]"; then
		fatal "variable is not set: $i"
	fi
done
# Adjust oom score.
if [ "$NOMAD_META_CI_OOM_SCORE_ADJUST" -ne 0 ] &&
	[ -e /proc/self/oom_score_adj ] &&
	i=$(cat /proc/self/oom_score_adj) &&
	[ "$i" != "$NOMAD_META_CI_OOM_SCORE_ADJUST" ]; then
	echo "$NOMAD_META_CI_OOM_SCORE_ADJUST" >/proc/self/oom_score_adj
fi
if hash bash 2>/dev/null; then set -- bash -s "$@"; else set -- sh -s "$@"; fi
case "$NOMAD_META_CI_DRIVER" in
raw_exec | exec)
	# Set taskset if available.
	if [ -n "$NOMAD_META_CI_CPUSET_CPUS" ]; then
		taskset -pc "$NOMAD_META_CI_CPUSET_CPUS" "$$" >/dev/null || true
	fi
	if [ -n "$NOMAD_META_CI_RUNUSER" ]; then
		set -- runuser -u "$NOMAD_META_CI_RUNUSER" -- "$@"
	fi
	;;
esac
"$@" || exit 76
