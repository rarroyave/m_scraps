#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="$root/.venv"
requirements="$root/tools/requirements-lock.txt"

source "$root/scripts/mpi-env.sh"
set_scraps_mpi_environment
"$root/build.sh"

uv_bin="${SCRAPS_UV:-}"
if [ -z "$uv_bin" ] && command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [ -z "$uv_bin" ] && [ -x "$HOME/.local/bin/uv" ]; then
    uv_bin="$HOME/.local/bin/uv"
fi

if [ -n "$uv_bin" ]; then
    "$uv_bin" venv --clear --python 3.12 "$venv"
    "$uv_bin" pip install --python "$venv/bin/python" -r "$requirements"
else
    python_bin="${SCRAPS_BOOTSTRAP_PYTHON:-python3}"
    "$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "SCRAPs requires Python 3.11+"'
    "$python_bin" -m venv --clear "$venv"
    "$venv/bin/python" -m pip install --upgrade pip
    "$venv/bin/python" -m pip install -r "$requirements"
fi

mkdir -p "$HOME/.local/bin"
for command in scraps-wizard scraps-run scraps-verify scraps-htp; do
    ln -sfn "$root/bin/$command" "$HOME/.local/bin/$command"
done

echo "Installed scraps-perpair at $root"
echo "MPI compiler: $MPICXX"
echo "MPI launcher: $MPIRUN"
echo "Commands: scraps-wizard, scraps-run, scraps-verify, scraps-htp"
