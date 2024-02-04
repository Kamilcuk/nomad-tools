#!/bin/bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
. ./unit_tests.sh
integration_tests "$@"
