#!/usr/bin/env bash

set_scraps_mpi_environment() {
    if [ -z "${SCRAPS_BREW:-}" ]; then
        if command -v brew >/dev/null 2>&1; then
            SCRAPS_BREW="$(command -v brew)"
        elif [ -x /opt/homebrew/bin/brew ]; then
            SCRAPS_BREW=/opt/homebrew/bin/brew
        elif [ -x /usr/local/bin/brew ]; then
            SCRAPS_BREW=/usr/local/bin/brew
        else
            SCRAPS_BREW=""
        fi
    fi

    if [ -z "${MPICXX:-}" ]; then
        if [ -n "$SCRAPS_BREW" ] && brew_prefix="$("$SCRAPS_BREW" --prefix open-mpi 2>/dev/null)"; then
            MPICXX="$brew_prefix/bin/mpic++"
        elif command -v mpic++ >/dev/null 2>&1; then
            MPICXX="$(command -v mpic++)"
        else
            echo "ERROR: no MPI C++ compiler found." >&2
            echo "Install Open MPI or set MPICXX=/path/to/mpic++." >&2
            return 1
        fi
    fi

    if [ ! -x "$MPICXX" ]; then
        echo "ERROR: MPI compiler is not executable: $MPICXX" >&2
        return 1
    fi

    if [ -z "${MPIRUN:-}" ]; then
        mpi_bin="$(cd -P "$(dirname "$MPICXX")" && pwd)"
        if [ -x "$mpi_bin/mpirun" ]; then
            MPIRUN="$mpi_bin/mpirun"
        elif command -v mpirun >/dev/null 2>&1; then
            MPIRUN="$(command -v mpirun)"
        else
            echo "ERROR: no MPI launcher found next to $MPICXX or on PATH." >&2
            return 1
        fi
    fi

    if [ ! -x "$MPIRUN" ]; then
        echo "ERROR: MPI launcher is not executable: $MPIRUN" >&2
        return 1
    fi

    export SCRAPS_BREW MPICXX MPIRUN
}
