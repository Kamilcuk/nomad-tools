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
	NOMAD_TASK_DIR \
	NOMAD_META_CI_DRIVER \
	NOMAD_META_CI_RUNUSER \
	NOMAD_META_CI_JOB_URL \
	NOMAD_META_CI_PROJECT_URL \
	NOMAD_META_CI_OOM_SCORE_ADJUST \
	NOMAD_META_CI_CPUSET_CPUS \
; do
	if eval "[ -z \"\${$i+x}\" ]"; then
		fatal "variable is not set: $i. This is an internal error from nomadtools project. Please report it to it. The variables should be set from the job definition when running the job."
	fi
done
if [ "$#" -lt 4 ]; then
	fatal "wrong number of arguments: \$#=$#. This is an internal error from nomadtools project. The variables are passed to this script from the statement in nomadtools source code. Please report it to nomadtools project github issues page."
fi
# Adjust oom score.
if [ "$NOMAD_META_CI_OOM_SCORE_ADJUST" -ne 0 ] &&
	[ -w /proc/self/oom_score_adj ] &&
	i=$(cat /proc/self/oom_score_adj) &&
	[ "$i" != "$NOMAD_META_CI_OOM_SCORE_ADJUST" ]; then
	echo "$NOMAD_META_CI_OOM_SCORE_ADJUST" >/proc/self/oom_score_adj
fi
# Command is accumulated in script positional arguments.
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
	if [ $(id -u) -eq 0 ] && [ -n "$NOMAD_META_CI_RUNUSER" ]; then
		if ! hash runuser 2>/dev/null; then
			fatal "Command runuser not found but requested. It was requested to run the command inside a runuser call. But the executable runuser was not found. There is no way I can run the command. Giving up."
		fi
		set -- runuser -u "$NOMAD_META_CI_RUNUSER" -- "$@"
	fi
	;;
docker)
	if [ -n "$NOMAD_META_CI_CPUSET_CPUS" ] && hash taskset 2>/dev/null; then
		taskset -pc "$NOMAD_META_CI_CPUSET_CPUS" "$$" >/dev/null || true
	fi
	# Fix permissions between ci-task and ci-help images.
	# ci-help runs gitlab-runner-helper image that runs as user root:root
	# But ci-task runs the user specified image with the user specified user.
	# When ci-help creates files in /builds directory they cannot be overwritten by ci-task.
	# This is a temporary fix until I can find proper solution.
	if [ "$NOMAD_TASK_NAME" = "ci-help" ]; then
		umask 000
	fi
	;;
esac
# Finally execute the commands script. Exit with proper exit status.
"$@" || { echo $? > "$NOMAD_TASK_DIR"/code.txt && exit 155; }
