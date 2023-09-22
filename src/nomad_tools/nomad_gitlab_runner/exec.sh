#!/bin/sh
if hash ionice 2>/dev/null; then ionice -c 3 -p $$; fi
if hash renice 2>/dev/null; then renice -n 40 $$ >/dev/null; fi
if hash bash 2>/dev/null; then sh=bash; else sh=sh; fi
if hash id runuser 2>/dev/null && id gitlab-runner 2>/dev/null >&2 && test "$(id -u)" = 0; then
	runuser="runuser -u gitlab-runner --"
fi
exec $runuser "$sh" "$@" -s
