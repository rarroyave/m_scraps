#!/usr/bin/env bash
# Build SCRAPS with the per-pair SRO extension.
#
# Tested with Homebrew Open MPI on macOS and system Open MPI on Linux.

set -euo pipefail

root="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$root/scripts/mpi-env.sh"
set_scraps_mpi_environment

if [ -z "${EIGEN_INC:-}" ]; then
    if command -v brew >/dev/null 2>&1 && brew_prefix="$(brew --prefix eigen 2>/dev/null)"; then
        EIGEN_INC="$brew_prefix/include/eigen3"
    elif command -v pkg-config >/dev/null 2>&1 && pkg_eigen="$(pkg-config --cflags-only-I eigen3 2>/dev/null)"; then
        EIGEN_INC="${pkg_eigen#-I}"
    elif [ -d /usr/include/eigen3 ]; then
        EIGEN_INC="/usr/include/eigen3"
    else
        EIGEN_INC=""
    fi
fi

if [ ! -x "$MPICXX" ]; then
    echo "ERROR: MPI compiler not found at $MPICXX"
    echo "Install with: brew install open-mpi"
    echo "Or set MPICXX=/path/to/your/mpic++"
    exit 1
fi

if [ ! -d "$EIGEN_INC" ]; then
    echo "ERROR: Eigen headers not found at $EIGEN_INC"
    echo "Install with: brew install eigen"
    echo "Or set EIGEN_INC=/path/to/eigen3"
    exit 1
fi

cd "$root/SCRAPs"

echo "Building with $MPICXX (Eigen at $EIGEN_INC)..."
"$MPICXX" -std=c++17 -O2 \
    main.cpp Lattice.cpp readInput.cpp \
    -I "$EIGEN_INC" \
    -o scraps

echo "Built: SCRAPs/scraps"
