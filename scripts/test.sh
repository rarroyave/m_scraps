#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/scripts/mpi-env.sh"
set_scraps_mpi_environment

python_bin="${SCRAPS_PYTHON:-$root/.venv/bin/python}"
if [ ! -x "$python_bin" ]; then
    echo "SCRAPs is not installed. Run $root/install.sh first." >&2
    exit 1
fi

exec "$python_bin" -m pytest "$root/tools/tests" "$@"
