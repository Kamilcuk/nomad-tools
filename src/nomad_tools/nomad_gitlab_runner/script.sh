#!/bin/sh
# nomad-gitlab-runner entrypoint script for adjusting stuff for gitlab runner.
# This script is passed two arguments: "script_to_execute" "one_arg"
fatal() {
	echo "nomad-gitlab-runner:script.sh: ERROR: $*" >&2
	exit 2
}
# Reduce niceness
if hash ionice 2>/dev/null; then ionice -c 3 -p $$; fi
if hash renice 2>/dev/null; then renice -n 40 $$ >/dev/null; fi
if hash chrt 2>/dev/null; then chrt -i -p 0 $$; fi
# Check if all variables are fine passed from Nomad meta.
for i in \
	NOMAD_META_CI_DRIVER \
	NOMAD_META_CI_RUNUSER \
	NOMAD_META_CI_JOB_URL \
	NOMAD_META_CI_PROJECT_URL \
	NOMAD_META_CI_OOM_SCORE_ADJUST \
	NOMAD_META_CI_CPUSET_CPUS \
	NOMAD_META_CI_EXIT_FAILURE; do
	if eval "[ -z \"\${$i+x}\" ]"; then
		fatal "variable is not set: $i"
	fi
done
if [ "$#" -ne 2 ]; then
	fatal "wrong number of arguments: $#"
fi
# Adjust oom score.
if [ "$NOMAD_META_CI_OOM_SCORE_ADJUST" -ne 0 ] &&
	[ -w /proc/self/oom_score_adj ] &&
	i=$(cat /proc/self/oom_score_adj) &&
	[ "$i" != "$NOMAD_META_CI_OOM_SCORE_ADJUST" ]; then
	echo "$NOMAD_META_CI_OOM_SCORE_ADJUST" >/proc/self/oom_score_adj
fi
# Command is accumulated in script positional arguments.
set -- -c "$@"
# If we are executing with -x, pass it also to the subshell.
case "$-" in *x*) set -- -x "$@" ;; esac
# Choose the shell if available.
if hash bash 2>/dev/null; then
	set -- bash "$@"
else
	set -- sh "$@"
fi
# Execute additional options depending on the driver.
case "$NOMAD_META_CI_DRIVER" in
raw_exec | exec)
	if [ -n "$NOMAD_META_CI_CPUSET_CPUS" ]; then
		taskset -pc "$NOMAD_META_CI_CPUSET_CPUS" "$$" >/dev/null || true
	fi
	if [ -n "$NOMAD_META_CI_RUNUSER" ]; then
		if ! hash runuser 2>/dev/null; then
			fatal "command runuser not found but requested"
		fi
		set -- runuser -u "$NOMAD_META_CI_RUNUSER" -- "$@"
	fi
	;;
docker)
	if [ -n "$NOMAD_META_CI_CPUSET_CPUS" ] && hash taskset 2>/dev/null; then
		taskset -pc "$NOMAD_META_CI_CPUSET_CPUS" "$$" >/dev/null || true
	fi
	;;
esac
# Finally execute the commands script. Exit with proper exit status.
"$@" || exit "$NOMAD_META_CI_EXIT_FAILURE"
