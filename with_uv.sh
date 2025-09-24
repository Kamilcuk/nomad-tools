#!/bin/bash
set -x
exec uv run --with-requirements requirements-test.txt --with-requirements requirements.txt "$@"
