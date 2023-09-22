#!/bin/sh
if ! test -e /.dockerenv; then echo "ERROR: /.dockerenv does not exists" >&2; exit 2; fi
if hash ionice 2>/dev/null; then ionice -c 3 -p $$; fi
if hash renice 2>/dev/null; then renice -n 40 $$ >/dev/null; fi
if hash bash 2>/dev/null; then sh=bash; else sh=sh; fi
exec "$sh" "$@" -s
