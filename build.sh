#!/bin/bash
# Build SCRAPS with the per-pair SRO extension.
#
# Tested on macOS arm64 with Homebrew Eigen + Open MPI.
# On Linux or Intel Mac, replace /opt/homebrew/... paths with your system
# MPI and Eigen locations.

set -e

MPICXX="${MPICXX:-/opt/homebrew/bin/mpic++}"
EIGEN_INC="${EIGEN_INC:-/opt/homebrew/opt/eigen/include/eigen3}"

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

cd "$(dirname "$0")/SCRAPs"

echo "Building with $MPICXX (Eigen at $EIGEN_INC)..."
"$MPICXX" -std=c++17 -O2 \
    main.cpp Lattice.cpp readInput.cpp \
    -I "$EIGEN_INC" \
    -o scraps

echo "Built: SCRAPs/scraps"
