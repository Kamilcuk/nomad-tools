#!/bin/sh
if hash ionice 2>/dev/null; then ionice -c 3 -p $$; fi
if hash renice 2>/dev/null; then renice -n 40 $$ >/dev/null; fi
cmd=""
if hash id runuser 2>/dev/null && id gitlab-runner 2>/dev/null && test "$(id -u)" = 0; then
	cmd="runuser -u gitlab-runner"
fi
if hash bash 2>/dev/null; then exec $cmd bash -s; else exec $cmd sh -s; fi
