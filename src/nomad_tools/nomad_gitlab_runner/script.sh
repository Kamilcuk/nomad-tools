#!/bin/sh
fatal() {
	echo "nomad-gitlab-runner: ERROR: $*" >&2
	exit 1
}
# Reduce niceness
if hash ionice 2>/dev/null; then ionice -c 3 -p $$; fi
if hash renice 2>/dev/null; then renice -n 40 $$ >/dev/null; fi
if hash chrt 2>/dev/null; then chrt -i -p 0 $$; fi
if [ -e /proc/self/oom_score_adj ] && i=$(cat /proc/self/oom_score_adj) && [ "$i" = 0 ]; then
	{ echo 10 >/proc/self/oom_score_adj; } 2>/dev/null
fi
# Check if all variables are fine.
for i in DRIVER RUNUSER JOB_URL PROJECT_URL; do
	i=NOMAD_META_CI_$i
	if eval "[ -z \"\${$i+x}\" ]"; then
		fatal "variable is not set: $i"
	fi
done
if hash bash 2>/dev/null; then set -- bash -s "$@"; else set -- sh -s "$@"; fi
case "$NOMAD_META_CI_DRIVER" in
raw_exec | exec)
	if [ -n "${NOMAD_META_CI_RUNUSER}" ]; then
		set -- runuser -u "$NOMAD_META_CI_RUNUSER" -- "$@"
	fi
	;;
esac
"$@" || exit 76
