//SCRAPs: A Multicomponent Alloy Structure Design Tool
//A novel Hybrid CUCKOO SEARCH determines combinatorial global optimum
//SuperCell Random APproximates(SCRAPs)for specified point and pair
//correlations in high-entropy alloys having proper distributions.
//Copyright (c) 2020. Duane D. Johnson. All rights reserved.
//Developed by: Duane D. Johnson, Rahul Singh, Prashant Singh, Aayush Sharma, Ganesh Balasubramanian.
//Iowa State University
//www.iastate.edu

//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the &quot;Software&quot;), to deal
//with the Software without restriction, including without limitation the
//rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
//sell copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//* Redistributions of source code must retain the above copyright notice,
//this list of conditions and the following disclaimers.
//* Redistributions in binary form must reproduce the above copyright notice,
//this list of conditions and the following disclaimers in the documentation
//and/or other materials provided with the distribution.
//* Neither the names of Duane D. Johnson, Iowa State University, nor the names
//of its contributors may be used to endorse or promote products derived from
//this Software without specific prior written permission.
//* Citation of the original archival journal should be made as is proper, i.e.
//R. Singh, et al., Nature Computational Science VOL, PAGE (2021) DOI:

//THE SOFTWARE IS PROVIDED &quot;AS IS&quot;, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
//THE SOFTWARE.


#ifndef readInput_hpp
#define readInput_hpp

#include <stdio.h>
#include <vector>
#include <string>
#include <ostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <utility>
#include "Lattice.hpp"

using namespace std;
class readInput{
public:
 // Input Variables
    int typeofatoms;
    int globalIter;
    vector<int> maxAtomNum;
    vector<int> atomNum;
    vector<int> eachElement;
    vector<int> cell_dim;
    vector<double> weights;
    int num_of_nests;
    int MCiter;
    int iterTotal;
    int num_of_shells;
    // Heuristic 1: plateau-based early stopping. If patience > 0, the main
    // optimization loop breaks once the global-best fitness has not
    // improved by more than `earlyStopTol` for `earlyStopPatience`
    // consecutive outer iterations. Default 0 disables (run to iterTotal).
    int earlyStopPatience;
    double earlyStopTol;
    // Heuristic 2: auto-scale MCSTEP. If autoMcstepFactor > 0, MCiter is
    // overridden by max(50, autoMcstepFactor * total_atoms) once the
    // supercell size is known. Default 0 leaves MCiter as parsed.
    int autoMcstepFactor;
    // Reproducibility: seed for std::mt19937. 0 = use std::random_device
    // (default, non-reproducible). Non-zero gives deterministic runs per
    // rank: each rank uses (rngSeed XOR rank_salt) so different ranks
    // still explore different streams.
    unsigned long rngSeed;
    // Baseline / ablation: if 1, skip the entire MC optimization loop
    // and return the best of the initial random pool. Used to measure
    // how much the cuckoo-search MC is improving over random search.
    int randomOnly;
    // Convergence telemetry: per-iteration fitness log path. Empty
    // disables logging. Written by rank 0 only; CSV with columns
    // iteration, best_fitness_rank0, wall_seconds.
    std::string convergenceLog;
    // Warm-start: path to a plain-text file containing the initial
    // nest as one species index per line (0-based, matching ELEMENT
    // order). When set, replaces nests[0] on every rank with the
    // file's contents -- the MC search then explores around that
    // seed instead of from a uniform random init. Useful for steep
    // targets like L2_1 where uniform random plateaus far from the
    // global minimum. Default empty = pure random init.
    std::string initNestFile;
    // MC kernel selector: "cuckoo" (default, the upstream-derived
    // greedy + epsilon-band acceptance inside a cuckoo-search
    // population), "sa" (simulated annealing with exponential
    // temperature schedule across outer iterations), or "pt"
    // (parallel tempering: each MPI rank runs SA at a fixed
    // log-spaced temperature, with periodic state exchange between
    // adjacent ranks via Metropolis criterion). Same swap proposal
    // mechanism; different acceptance rule and ensemble structure.
    std::string mcKernel;
    // SA temperature schedule. For mcKernel="sa": T(outer_iter)
    // decays exponentially from saTInitial to saTFinal across
    // iterTotal. For mcKernel="pt": each rank gets a fixed
    // log-spaced T in [saTFinal, saTInitial], rank 0 is hottest
    // (saTInitial), rank nproc-1 is coldest (saTFinal).
    double saTInitial;
    double saTFinal;
    // PT-only: every ptExchangeInterval outer iterations, paired
    // adjacent ranks (0-1, 2-3, ...) attempt to swap their nests[0]
    // state via the standard PT Metropolis criterion
    //   min(1, exp((1/T_i - 1/T_j) * (E_j - E_i))).
    // Default 1 = exchange every iter.
    int ptExchangeInterval;
    // Cluster swap: each MC step proposes clusterSwapK simultaneous
    // atom-pair swaps and accepts/rejects the joint move. K = 1
    // (default) reduces to the standard single-swap MC. K > 1
    // explores larger moves that can escape local minima single-swap
    // MC can't cross -- e.g., the L2_1 cold-start plateau where the
    // ground state requires multiple correlated swaps. Tradeoff: each
    // cluster move costs ~K * O(z) to apply and an O(N*z) rebuild on
    // reject, so the per-step cost grows roughly linearly in K. Sweet
    // spot is usually K in {2, 3, 4} for hard targets.
    int clusterSwapK;
    // Delta-fitness MC: instead of recomputing the full fitness on
    // every swap proposal (O(N*z)), maintain pairGr as state and apply
    // a delta from each swap (O(z)). Yields 50-250x more swap attempts
    // per wall-second on typical cells with no change to the underlying
    // optimization. Default 1 (on); set 0 to use the legacy
    // recompute-each-swap path for direct A/B comparison.
    int fastSwaps;
    // Targeted swap proposals: with probability targetedSwapProb, propose
    // swaps biased toward the species pair (i, j, shell k) with the
    // largest residual^2, instead of fully uniform-random atom pairs.
    // The remaining fraction stays uniform-random for exploration.
    // Default off (0) -- preserves the existing optimization trajectory
    // and lets you A/B benchmark the win on your workload. Set
    // targetedSwaps=1 to enable; targetedSwapProb tunes the bias level
    // (0.0 = pure random, 1.0 = pure greedy).
    int targetedSwaps;
    double targetedSwapProb;
    // Per-pair, per-shell Warren-Cowley SRO targets: targetSROpair[i][j][shell].
    // Defaults to all zeros (random / SQS) when no SRO lines appear in INPUT.
    // Populated by readInput::input() after TYP and MAXSHELLNUM are known.
    vector<vector<vector<double>>> targetSROpair;

    // Spectator sublattice handling for structures like AlB2-type diborides
    // (high-entropy diborides), where one sublattice has no compositional
    // disorder and just acts as a fixed structural scaffold. Each entry is
    // a (basis_position, species_index) pair declaring that the given basis
    // position is always occupied by the given species and is not swapped
    // during optimization. Empty when no SPECTATOR_BASIS_SPECIES lines
    // appear in INPUT (single-sublattice / standard substitutional alloy).
    vector<pair<int,int>> spectatorBasisSpecies;
    // Derived: fast lookups built once after parsing.
    set<int> spectatorBasisPositions;  // basis positions in the spectator set
    set<int> spectatorSpecies;         // species marked as spectator
// Lattice Variables
    int totalatoms;
    point3d a, b, c;
    vector<point3d> atoms;

// Functions
    void input();
    readInput();
    void readPOSCAR();
    
    
};


#endif /* readInput_hpp */
