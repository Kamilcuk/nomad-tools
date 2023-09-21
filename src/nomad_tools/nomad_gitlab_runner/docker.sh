#!/bin/sh
if hash ionice 2>/dev/null; then ionice -c 3 -p $$; fi
if hash renice 2>/dev/null; then renice -n 40 $$ >/dev/null; fi
if hash bash 2>/dev/null; then exec bash -s; else exec sh -s; fi
