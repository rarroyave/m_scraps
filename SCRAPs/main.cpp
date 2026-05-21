/*
SCRAPs: A Multicomponent Alloy Structure Design Tool
A novel Hybrid CUCKOO SEARCH determines combinatorial global optimum
SuperCell Random APproximates(SCRAPs)for specified point and pair
correlations in high-entropy alloys having proper distributions.
Copyright (c) 2020. Duane D. Johnson. All rights reserved.
Developed by: Duane D. Johnson, Rahul Singh, Prashant Singh, Aayush Sharma, Ganesh Balasubramanian.
Iowa State University
www.iastate.edu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the &quot;Software&quot;), to deal
with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimers.
* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimers in the documentation
and/or other materials provided with the distribution.
* Neither the names of Duane D. Johnson, Iowa State University, nor the names
of its contributors may be used to endorse or promote products derived from
this Software without specific prior written permission.
* Citation of the original archival journal should be made as is proper, i.e.
R. Singh, et al., Nature Computational Science VOL, PAGE (2021) DOI:

THE SOFTWARE IS PROVIDED &quot;AS IS&quot;, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
THE SOFTWARE.
*/

#include <iostream>
#include <fstream>
#include "Lattice.hpp"
#include <unordered_map>
#include <chrono>
#include <random>
#include "readInput.hpp"
#include <mpi.h>
#include <math.h>
#include <unistd.h>
#include <cstdlib>

// Process-level RNG seed base for reproducibility. Set in main()
// from inputfile.rngSeed (xor'd with rank). All thread-local RNGs
// in empty_nests / randomizeLattice seed from this on first use.
// Value 0 falls back to std::random_device for non-reproducible runs.
static unsigned long g_rng_seed_base = 0;
// Process-level switch for the delta-fitness MC path. Set in main()
// from inputfile.fastSwaps. When true, empty_nests dispatches to
// empty_nests_fast at the top.
static bool g_fast_swaps = true;
// Process-level controls for targeted (residual-biased) swap proposals.
// When g_targeted_swaps is true, empty_nests_fast proposes
// g_targeted_swap_prob of swaps biased toward the species pair (i,j,k)
// with the largest residual^2; the rest stay uniform-random.
static bool g_targeted_swaps = false;
static double g_targeted_swap_prob = 0.7;

// MC kernel selector and SA schedule parameters. Set in main() from
// inputfile.mcKernel / saTInitial / saTFinal. g_sa_current_T is
// recomputed at the start of each outer iteration based on a global
// counter (g_outer_iter / g_outer_iter_total) so the temperature
// decays exponentially across the optimization. PT reuses the SA
// inner kernel but holds T fixed per rank and adds an exchange step
// between iters.
static bool g_use_sa = false;          // true if MC_KERNEL == "sa" or "pt"
static bool g_use_pt = false;          // true if MC_KERNEL == "pt"
static double g_sa_t_initial = 2.0;
static double g_sa_t_final = 1e-3;
static double g_sa_current_T = 1.0;
static int g_pt_exchange_interval = 1;

// Forward decls so empty_nests can dispatch to them.
int empty_nests_fast(vector<vector<int>> &_nests, vector<vector<int>> &_newnests, int iter, vector<double> &origfitness, const Lattice &lat, const int _typesatoms, const vector<int> eachElement, int &max_num_neighs, vector<double> &weights, const vector<vector<vector<double>>> &targetSROpair, const vector<int> &initialPlacement);
int empty_nests_sa(vector<vector<int>> &_nests, vector<vector<int>> &_newnests, int iter, vector<double> &origfitness, const Lattice &lat, const int _typesatoms, const vector<int> eachElement, int &max_num_neighs, vector<double> &weights, const vector<vector<vector<double>>> &targetSROpair, const vector<int> &initialPlacement);


void sortthenests(vector<vector<int>> &_nests, vector<double> &_fitness, int &max_num_neighs)
{
    int numofnests = _nests.size();
    vector<int> temp;
    double tempfitn;
    for(int i = 0; i < numofnests; i++)
    {
        for (int j = i + 1; j < numofnests; j++) {
            if(_fitness[j] < _fitness[i])
            {
                swap(_nests[i], _nests[j]);
                swap(_fitness[i], _fitness[j]);
            }
        }
        
    }
}

void printtoscreen(vector<int> nest, const Lattice &lat, int _typesatoms, vector<int> eachElement, const vector<int> &initialPlacement)
{
    int numofmaxneighs = lat.maxneigh;
    int numofatoms = nest.size();

    // Per-sublattice counts for the M-only reference. For a pure
    // (non-HEB) run with initialPlacement all -1, these reduce to
    // numofatoms and eachElement.
    int numofatoms_M = 0;
    vector<int> eachElement_M(_typesatoms, 0);
    for (int i = 0; i < numofatoms; ++i) {
        if (i < (int)initialPlacement.size() && initialPlacement[i] >= 0) continue;
        numofatoms_M += 1;
        eachElement_M[nest[i]] += 1;
    }

    vector<vector<vector<double> > > pairGr(_typesatoms, std::vector<vector <double>> (_typesatoms, std::vector<double> (numofmaxneighs, 0.0)));
    // Accumulate pair counts over non-spectator centers and non-spectator
    // neighbors only — this is the M-M-only neighbor census.
    for (int i = 0; i < numofatoms; ++i) {
        if (i < (int)initialPlacement.size() && initialPlacement[i] >= 0) continue;
        for (int j = 0; j < numofatoms; ++j) {
            if (j < (int)initialPlacement.size() && initialPlacement[j] >= 0) continue;
            if (lat.neighbours[i][j] > 0) {
                pairGr[nest[i]][nest[j]][lat.neighbours[i][j] - 1] += 1;
            }
        }
    }

    // Warren-Cowley parameter from pair counts:
    //   alpha_ij = 1 - P(j|i) / c_j
    //            = 1 - count_ij * N_total / (z_shell * N_i * N_j)
    //
    // For spectator-aware (HEB) runs, N_total -> N^M (non-spectator atoms),
    // N_i, N_j -> N_i^M, N_j^M (per-sublattice species counts), and z_shell
    // is the *M-M only* coordination (count of non-spectator neighbors of
    // a non-spectator center). We compute it directly from atoms_shell_atom
    // here rather than via Lattice::coordination_per_shell(), because that
    // method derives from maxneighbourspershell which is truncated by
    // numneighofeachshell()'s early-exit-on-zero behavior (an M atom
    // whose first global shell is the B-B distance has zero entries
    // there, so its maxneighbourspershell vector terminates after one
    // element and we lose the M-M shells that come later).
    //
    // Rows/columns of the alpha matrix are suppressed for spectator
    // species since they carry no SRO.
    int ref_idx = 0;
    while (ref_idx < numofatoms &&
           ref_idx < (int)initialPlacement.size() &&
           initialPlacement[ref_idx] >= 0) {
        ref_idx += 1;
    }
    if (ref_idx >= numofatoms) return;  // all atoms are spectator; nothing to report

    for (int k = 0; k < numofmaxneighs; k++) {
        // Compute z_M for shell k: non-spectator neighbors of ref atom.
        int z_M_shell = 0;
        if (k < (int)lat.atoms_shell_atom[ref_idx].size()) {
            for (int neigh_idx : lat.atoms_shell_atom[ref_idx][k]) {
                if (neigh_idx < (int)initialPlacement.size() &&
                    initialPlacement[neigh_idx] >= 0) continue;
                z_M_shell += 1;
            }
        }
        if (z_M_shell <= 0) continue;  // no M-M pairs in this shell; skip

        cout << " SRO FOR THE SHELL NO - > " << k << endl;
        for (int i = 0; i < _typesatoms; i++) {
            if (eachElement_M[i] == 0) continue;  // suppress spectator-species rows
            for (int j = 0; j < _typesatoms; j++) {
                if (i == j) continue;
                if (eachElement_M[j] == 0) continue;  // suppress spectator-species cols
                double alpha = 1.0 - (double)pairGr[i][j][k] * numofatoms_M
                             / ((double)z_M_shell * eachElement_M[i] * eachElement_M[j]);
                cout << "The pair SRO[" << i << "][" << j << "] - > is " << alpha << endl;
            }
        }
    }
    return;
}

//Test for the goodness of fit
//
// Per-pair, per-shell Warren-Cowley targets are passed in via targetSROpair
// (shape [_typesatoms][_typesatoms][>=max_num_neighs]). An empty/zero matrix
// reproduces the original SQS-style "all-random" behavior.
//
// Also fixes an upstream bug: the original gated avgGr accumulation on
// `nest[center] != nest[surr]`, which indexed `nest` by a species index
// rather than an atom index. The correct test for "unlike pair" is
// `center != surr` (compare species directly).
double fitnessvalue(vector<int> nest, const Lattice &lat, int _typesatoms, vector<int> eachElement, int &max_num_neighs, vector<double> &weights, const vector<vector<vector<double>>> &targetSROpair, const vector<int> &initialPlacement)
{
    // Spectator-aware Warren-Cowley fitness:
    //   - center atoms with initialPlacement[i] >= 0 are skipped entirely
    //     (no SRO is defined for fixed-sublattice positions);
    //   - spectator neighbors are excluded from the per-shell sum, so the
    //     effective coordination number used in the expected-count formula
    //     becomes "number of non-spectator neighbors in this shell" rather
    //     than the full geometric coordination;
    //   - the reference concentration c_j becomes the per-sublattice
    //     concentration N_j^M / N^M, where M denotes the set of non-
    //     spectator atoms. For HEB-type structures (one designed sublattice
    //     plus one all-spectator sublattice) this is exactly the right
    //     reference for "M-M-only" SRO; for general partially-occupied
    //     multi-sublattice systems it assumes every species is either
    //     fully spectator or fully non-spectator.
    int doublepairs = (int)_typesatoms + (int)_typesatoms*((int)_typesatoms - 1)/2;
    double targetErr = 0.0, Error = 0.0, sqrError = 0.0;
    double totalGr = 0.0;
    int numofmaxneighs = lat.maxneigh;
    int numofatoms = nest.size();

    // Per-sublattice atom counts (computed from the current nest; preserved
    // across swaps because empty_nests rejects swaps involving spectators).
    int numofatoms_M = 0;
    vector<int> eachElement_M(_typesatoms, 0);
    for (int i = 0; i < numofatoms; ++i) {
        if (initialPlacement[i] < 0) {
            numofatoms_M += 1;
            eachElement_M[nest[i]] += 1;
        }
    }
    // If no atoms are spectator, numofatoms_M == numofatoms and
    // eachElement_M == eachElement, so this reduces to the prior behavior.

    // Per-shell error accumulator. Was previously a hardcoded
    // double[5], which silently overflowed the stack for any
    // MAXSHELLNUM > 5 (HEB-type structures hit 10+ global shells
    // before reaching the M-M shells, so this bound bit immediately
    // when spectator support landed).
    vector<double> errorArray(max_num_neighs > 0 ? max_num_neighs : 1, 0.0);
    vector<vector<vector<double> > > pairGr(_typesatoms, std::vector<vector <double>> (_typesatoms, std::vector<double> (numofmaxneighs, 0.0)));
    vector<vector<vector<double> > > avgGr(_typesatoms, std::vector<vector <double>> (_typesatoms, std::vector<double> (numofmaxneighs, 0.0)));
    vector<vector<vector<double> > > countgr(_typesatoms, std::vector<vector <double>> (_typesatoms, std::vector<double> (numofmaxneighs, 0.0)));
    int center, surr, shellnum;
    for(int i = 0; i < (int)nest.size(); i++)
    {
        if (initialPlacement[i] >= 0) continue;  // skip spectator centers
        for(int j = 0; j < max_num_neighs; j++)
        {
            // Effective coordination = non-spectator neighbors in this shell.
            shellnum = lat.atoms_shell_atom[i][j].size();
            int shellnum_M = 0;
            for (int k = 0; k < shellnum; ++k) {
                if (initialPlacement[lat.atoms_shell_atom[i][j][k]] < 0) {
                    shellnum_M += 1;
                }
            }
            for(int k = 0; k < shellnum; k++)
            {
                int neigh_idx = lat.atoms_shell_atom[i][j][k];
                if (initialPlacement[neigh_idx] >= 0) continue;  // skip spectator neighbor
                center = nest[i];
                surr = nest[neigh_idx];
                pairGr[center][surr][j] += 1;
                if(center != surr)
                {
                    double alphaTarget = targetSROpair[center][surr][j];
                    avgGr[center][surr][j] += (double)eachElement_M[center]*(1 - alphaTarget)*shellnum_M*((double)eachElement_M[surr]/(double)numofatoms_M);
                    countgr[center][surr][j] += 1;
                }
            }
        }
    }
    
   // cout << "   pair GER stsrts" << endl;
    for(int k = 0; k < max_num_neighs; k++)
    {
        for(int i = 0; i < _typesatoms; i++)
        {
            for(int j = 0; j < _typesatoms; j++)
            {
                if(i != j )
                {
                    if(countgr[i][j][k] == 0)
                        avgGr[i][j][k] = 0;
                    else
                        avgGr[i][j][k] /= countgr[i][j][k];
                    errorArray[k] += pow(pairGr[i][j][k] - avgGr[i][j][k], 2);
                }
            }
        }
        Error += weights[k]*errorArray[k];
    }
    sqrError = sqrt(Error);
//    cout << Error << endl;
    return sqrError;
}

/*
 
 The function based on fitness randomize and sorts the best nests for global optimization.
 
 */
int get_best_nest(vector<vector<int>> &_nests, vector<vector<int>> &_newnests, vector<double> &_fitness, const Lattice &lat, int _typeatoms, vector<int> eachElement, int &max_num_neighs, vector<double> &weights, const vector<vector<vector<double>>> &targetSROpair, const vector<int> &initialPlacement)
{
    int n, indexOfBest;
    n = _nests.size();
    double keepCuckoos = 0.4;
    sortthenests(_nests, _fitness, max_num_neighs);
    if(n == 1)
    {
        int fitness_new = fitnessvalue(_newnests[0], lat, _typeatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
        if (fitness_new < _fitness[0])
        {
            _nests = _newnests;
            _fitness[0] = fitness_new;
        }
    }
    else
    {
        for(int i = (int)(keepCuckoos*(double)n); i < n; i++)
        {
            _nests[i] = _newnests[i - (int)(keepCuckoos*(double)n)];
            _fitness[i] = fitnessvalue(_nests[i], lat, _typeatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
        }
        sortthenests(_nests, _fitness, max_num_neighs);
    }
    cout << "===================  BEST NEST  =========================" << endl;
    cout << "            	Fitness ::" << _fitness[0] << endl;
    cout << endl;
    indexOfBest = 0;
    return indexOfBest;
}
/*
 
 Based on goodness of "Fitness", function below discards the old and add new ones
during global optimization.

 */

int empty_nests(vector<vector<int>> &_nests, vector<vector<int>> &_newnests, int iter, vector<double> &origfitness , const Lattice &lat, const int _typesatoms, const vector<int> eachElement, int &max_num_neighs, vector<double> &weights, const vector<vector<vector<double>>> &targetSROpair, const vector<int> &initialPlacement)
{
    // Dispatch by MC_KERNEL then FAST_SWAPS:
    //   MC_KERNEL = sa     -> SA acceptance with exponential temperature
    //                         schedule (delta-fitness inside).
    //   MC_KERNEL = cuckoo + FAST_SWAPS = 1 (default)
    //                      -> greedy + epsilon-band, delta-fitness path.
    //   MC_KERNEL = cuckoo + FAST_SWAPS = 0
    //                      -> greedy + epsilon-band, legacy full-recompute.
    if (g_use_sa) {
        return empty_nests_sa(_nests, _newnests, iter, origfitness, lat,
                              _typesatoms, eachElement, max_num_neighs,
                              weights, targetSROpair, initialPlacement);
    }
    if (g_fast_swaps) {
        return empty_nests_fast(_nests, _newnests, iter, origfitness, lat,
                                _typesatoms, eachElement, max_num_neighs,
                                weights, targetSROpair, initialPlacement);
    }
    // initialPlacement preserves spectator-sublattice occupancy through
    // the Cuckoo Search MCMC: swaps that would move a fixed atom are
    // rejected, so the spectator pattern (e.g. boron sites in an HEB)
    // stays put for the full optimization.
    double len = _nests[0].size();
    double eps = 0.001;
    // Reproducible-seed-aware RNGs. First call seeds from g_rng_seed_base
    // (set in main from RNG_SEED, xor'd with rank); subsequent calls reuse
    // the advanced state so we don't keep restarting the stream and waste
    // entropy.
    static thread_local std::mt19937 gen, gen1;
    static thread_local bool seeded = false;
    if (!seeded) {
        if (g_rng_seed_base != 0) {
            gen.seed(g_rng_seed_base);
            gen1.seed(g_rng_seed_base ^ 0xA5A5A5A5UL);
        } else {
            std::random_device rd, rd1;
            gen.seed(rd());
            gen1.seed(rd1());
        }
        seeded = true;
    }
    std::uniform_int_distribution<> dis(0, len - 1);
    std::uniform_int_distribution<> dis1(1, 100);
    double oldFitness, newFitness;

    int atom1, atom2, temphold, prob;
    vector<int> prevnest, currnest;
    for (int i = 0; i < _nests.size(); i++) {
        prevnest = _nests[i];
        currnest = _nests[i];
        oldFitness = fitnessvalue(_nests[i], lat, _typesatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
        origfitness[i] = oldFitness;
        for(int j = 0; j < iter; j++)
        {
            atom1 = (int) dis(gen);
            atom2 = (int) dis(gen);
            // Reject swaps involving spectator atoms (preserve sublattice).
            if (initialPlacement[atom1] >= 0 || initialPlacement[atom2] >= 0)
                continue;
            if(currnest[atom1] == currnest[atom2])
                continue;
            temphold = currnest[atom1];
            currnest[atom1] = currnest[atom2];
            currnest[atom2] = temphold;
            newFitness = fitnessvalue(currnest, lat, _typesatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
            if(newFitness < oldFitness)
            {
                prevnest = currnest;
                origfitness[i] = newFitness;
                oldFitness = newFitness;
            }
            else
            {
                if (abs(newFitness - oldFitness) < eps) {
                    prob = dis1(gen1);
                    if(prob > 50)
                    {
                        prevnest = currnest;
                        origfitness[i] = newFitness;
                        oldFitness = newFitness;
                    }
                }
                else
                    currnest = prevnest;
            }
        }
        _nests[i] = currnest;
    }
    return 0;
}
// Delta-fitness MC swap loop. Maintains pairGr as state across swaps and
// applies an O(z) delta per proposal instead of recomputing the full
// O(N*z) fitness. Otherwise identical to empty_nests: greedy accept with a
// small epsilon-band stochastic acceptance to escape micro-plateaus, and
// the same spectator-rejection rule for HEB-type sublattices.
//
// Correctness: expected[i][j][k] = N_i^M * (1 - alpha_ij^k) * z_M^k *
// N_j^M / N^M is constant for a given nest topology (N_i^M, z_M, alpha
// are all invariant under within-sublattice swaps). The fitness equals
// sum_k w_k * sum_{i != j} (pairGr[i][j][k] - expected[i][j][k])^2. We
// precompute expected, maintain pairGr, and update it incrementally by
// touching only the O(z) neighbors of the two swapped atoms.
//
// Edge case: when countgr would be 0 in the legacy formulation (no i-j
// pairs in the nest at all for some i!=j), the legacy code zeros out
// avgGr while the constant-expected here would still penalize. In
// practice countgr > 0 for normal compositions, so results match to FP
// precision on real test cases.
int empty_nests_fast(vector<vector<int>> &_nests, vector<vector<int>> &_newnests, int iter, vector<double> &origfitness, const Lattice &lat, const int _typesatoms, const vector<int> eachElement, int &max_num_neighs, vector<double> &weights, const vector<vector<vector<double>>> &targetSROpair, const vector<int> &initialPlacement)
{
    const int n_atoms = (int) _nests[0].size();
    const double eps = 0.001;

    // Same salts as the legacy empty_nests() path so that, with the same
    // RNG_SEED, FAST_SWAPS=0 and FAST_SWAPS=1 propose the same swap
    // sequence and reach the same final state (modulo the constant-vs-
    // countgr-zero edge case noted above).
    static thread_local std::mt19937 gen_f, gen1_f;
    static thread_local bool seeded_f = false;
    if (!seeded_f) {
        if (g_rng_seed_base != 0) {
            gen_f.seed(g_rng_seed_base);
            gen1_f.seed(g_rng_seed_base ^ 0xA5A5A5A5UL);
        } else {
            std::random_device rd_f, rd1_f;
            gen_f.seed(rd_f());
            gen1_f.seed(rd1_f());
        }
        seeded_f = true;
    }
    std::uniform_int_distribution<> dis(0, n_atoms - 1);
    std::uniform_int_distribution<> dis1(1, 100);

    // ---- Sublattice statistics: N^M and N_i^M (constant during MC) ----
    int N_M = 0;
    vector<int> eachElement_M(_typesatoms, 0);
    for (int i = 0; i < n_atoms; i++) {
        if (initialPlacement[i] < 0) {
            N_M += 1;
            eachElement_M[_nests[0][i]] += 1;
        }
    }
    if (N_M == 0) return 0;

    // ---- z_M^k: avg non-spectator neighbors per non-spectator center per shell ----
    int ref = 0;
    while (ref < n_atoms && initialPlacement[ref] >= 0) ref++;
    vector<int> z_M(max_num_neighs, 0);
    for (int k = 0; k < max_num_neighs; k++) {
        if (k < (int) lat.atoms_shell_atom[ref].size()) {
            for (int j : lat.atoms_shell_atom[ref][k]) {
                if (initialPlacement[j] < 0) z_M[k] += 1;
            }
        }
    }

    // ---- Precompute expected[i][j][k] (zero for pure-spectator species) ----
    vector<vector<vector<double>>> expected(_typesatoms,
        vector<vector<double>>(_typesatoms, vector<double>(max_num_neighs, 0.0)));
    for (int k = 0; k < max_num_neighs; k++) {
        if (z_M[k] == 0) continue;
        for (int i = 0; i < _typesatoms; i++) {
            if (eachElement_M[i] == 0) continue;
            for (int j = 0; j < _typesatoms; j++) {
                if (i == j) continue;
                if (eachElement_M[j] == 0) continue;
                double alpha = targetSROpair[i][j][k];
                expected[i][j][k] = (double) eachElement_M[i] * (1.0 - alpha)
                                  * (double) z_M[k] * (double) eachElement_M[j]
                                  / (double) N_M;
            }
        }
    }

    auto fitness_from_pairGr = [&](const vector<vector<vector<double>>>& pgr) -> double {
        // Match legacy fitnessvalue semantics: when countgr is 0 (no
        // pairs of this type in the current nest), legacy sets avgGr = 0
        // and the contribution becomes pgr^2 = 0. The principled
        // delta-fitness would use the constant `expected` here -- giving
        // expected^2 contribution -- which DISAGREES with legacy and
        // breaks the L2_1 warm-start (perfect L2_1 has zero X-Y and X-Z
        // at 2NN; legacy says fit=0, principled delta says fit > 0).
        // Match legacy by skipping pairs with pgr==0 (equivalent to
        // setting the per-pair contribution to 0 in that case).
        double err = 0.0;
        for (int k = 0; k < max_num_neighs; k++) {
            double e_k = 0.0;
            for (int i = 0; i < _typesatoms; i++) {
                for (int j = 0; j < _typesatoms; j++) {
                    if (i == j) continue;
                    if (pgr[i][j][k] == 0.0) continue;  // legacy countgr==0 contract
                    double d = pgr[i][j][k] - expected[i][j][k];
                    e_k += d * d;
                }
            }
            err += weights[k] * e_k;
        }
        return std::sqrt(err);
    };

    auto build_pairGr_full = [&](const vector<int>& nest,
                                  vector<vector<vector<double>>>& pgr) {
        for (auto& a : pgr) for (auto& b : a) std::fill(b.begin(), b.end(), 0.0);
        for (int i = 0; i < n_atoms; i++) {
            if (initialPlacement[i] >= 0) continue;
            for (int k = 0; k < max_num_neighs; k++) {
                for (int j : lat.atoms_shell_atom[i][k]) {
                    if (initialPlacement[j] >= 0) continue;
                    pgr[nest[i]][nest[j]][k] += 1;
                }
            }
        }
    };

    vector<vector<vector<double>>> pairGr(_typesatoms,
        vector<vector<double>>(_typesatoms, vector<double>(max_num_neighs, 0.0)));

    // For targeted swap proposals: maintain per-species atom lists with
    // O(1) swap-remove (atom_pos[idx] points into species_atoms[species]).
    vector<vector<int>> species_atoms(_typesatoms);
    vector<int> atom_pos(n_atoms, -1);
    std::uniform_real_distribution<double> dis_real(0.0, 1.0);

    for (int i = 0; i < (int) _nests.size(); i++) {
        vector<int> currnest = _nests[i];
        build_pairGr_full(currnest, pairGr);
        double oldFitness = fitness_from_pairGr(pairGr);
        origfitness[i] = oldFitness;

        // (Re)build species_atoms for this nest. Cheap O(N) once per cuckoo.
        for (auto &v : species_atoms) v.clear();
        for (int idx = 0; idx < n_atoms; idx++) {
            if (initialPlacement[idx] >= 0) continue;
            atom_pos[idx] = (int) species_atoms[currnest[idx]].size();
            species_atoms[currnest[idx]].push_back(idx);
        }

        for (int step = 0; step < iter; step++) {
            int a, b;
            bool use_targeted = g_targeted_swaps
                && dis_real(gen_f) < g_targeted_swap_prob;
            if (use_targeted) {
                // Pick (i, j, k) with prob ~ w_k * (pairGr - expected)^2.
                // Falls back to uniform if no non-zero residual exists yet.
                double total_score = 0.0;
                for (int k = 0; k < max_num_neighs; k++) {
                    double w = (k < (int) weights.size()) ? weights[k] : 1.0;
                    if (w == 0.0) continue;
                    for (int p = 0; p < _typesatoms; p++) {
                        if (species_atoms[p].empty()) continue;
                        for (int q = 0; q < _typesatoms; q++) {
                            if (p == q) continue;
                            if (species_atoms[q].empty()) continue;
                            double r = pairGr[p][q][k] - expected[p][q][k];
                            total_score += w * r * r;
                        }
                    }
                }
                if (total_score > 0.0) {
                    double pick = dis_real(gen_f) * total_score;
                    double acc = 0.0;
                    int chosen_p = -1, chosen_q = -1;
                    for (int k = 0; k < max_num_neighs && chosen_p < 0; k++) {
                        double w = (k < (int) weights.size()) ? weights[k] : 1.0;
                        if (w == 0.0) continue;
                        for (int p = 0; p < _typesatoms && chosen_p < 0; p++) {
                            if (species_atoms[p].empty()) continue;
                            for (int q = 0; q < _typesatoms; q++) {
                                if (p == q) continue;
                                if (species_atoms[q].empty()) continue;
                                double r = pairGr[p][q][k] - expected[p][q][k];
                                acc += w * r * r;
                                if (acc >= pick) {
                                    chosen_p = p; chosen_q = q;
                                    break;
                                }
                            }
                        }
                    }
                    if (chosen_p >= 0 && chosen_q >= 0) {
                        a = species_atoms[chosen_p][
                            std::uniform_int_distribution<>(0, (int)species_atoms[chosen_p].size() - 1)(gen_f)];
                        b = species_atoms[chosen_q][
                            std::uniform_int_distribution<>(0, (int)species_atoms[chosen_q].size() - 1)(gen_f)];
                    } else {
                        a = dis(gen_f); b = dis(gen_f);
                    }
                } else {
                    // Already at zero residual everywhere; nothing to bias.
                    a = dis(gen_f); b = dis(gen_f);
                }
            } else {
                a = dis(gen_f);
                b = dis(gen_f);
            }
            if (initialPlacement[a] >= 0 || initialPlacement[b] >= 0) continue;
            int sa = currnest[a], sb = currnest[b];
            if (sa == sb) continue;

            // Apply swap delta to pairGr in-place (will undo on reject)
            for (int k = 0; k < max_num_neighs; k++) {
                for (int jj : lat.atoms_shell_atom[a][k]) {
                    if (initialPlacement[jj] >= 0) continue;
                    if (jj == b) continue;
                    int sj = currnest[jj];
                    pairGr[sa][sj][k] -= 1;
                    pairGr[sj][sa][k] -= 1;
                    pairGr[sb][sj][k] += 1;
                    pairGr[sj][sb][k] += 1;
                }
                for (int jj : lat.atoms_shell_atom[b][k]) {
                    if (initialPlacement[jj] >= 0) continue;
                    if (jj == a) continue;
                    int sj = currnest[jj];
                    pairGr[sb][sj][k] -= 1;
                    pairGr[sj][sb][k] -= 1;
                    pairGr[sa][sj][k] += 1;
                    pairGr[sj][sa][k] += 1;
                }
            }

            double newFitness = fitness_from_pairGr(pairGr);

            bool accept = false;
            if (newFitness < oldFitness) {
                accept = true;
            } else if (std::abs(newFitness - oldFitness) < eps) {
                int prob = dis1(gen1_f);
                if (prob > 50) accept = true;
            }

            if (accept) {
                // Update species_atoms via swap-remove. After the swap,
                // atom a has species sb and atom b has species sa.
                if (g_targeted_swaps) {
                    auto remove_from = [&](int s, int idx) {
                        int pos = atom_pos[idx];
                        int last = (int) species_atoms[s].size() - 1;
                        if (pos != last) {
                            species_atoms[s][pos] = species_atoms[s][last];
                            atom_pos[species_atoms[s][pos]] = pos;
                        }
                        species_atoms[s].pop_back();
                        atom_pos[idx] = -1;
                    };
                    auto add_to = [&](int s, int idx) {
                        atom_pos[idx] = (int) species_atoms[s].size();
                        species_atoms[s].push_back(idx);
                    };
                    remove_from(sa, a);
                    remove_from(sb, b);
                    add_to(sb, a);
                    add_to(sa, b);
                }
                currnest[a] = sb;
                currnest[b] = sa;
                oldFitness = newFitness;
                origfitness[i] = newFitness;
            } else {
                // Undo the swap delta in pairGr (a, b have not been swapped
                // in currnest yet, so sa, sb still refer to original species)
                for (int k = 0; k < max_num_neighs; k++) {
                    for (int jj : lat.atoms_shell_atom[a][k]) {
                        if (initialPlacement[jj] >= 0) continue;
                        if (jj == b) continue;
                        int sj = currnest[jj];
                        pairGr[sa][sj][k] += 1;
                        pairGr[sj][sa][k] += 1;
                        pairGr[sb][sj][k] -= 1;
                        pairGr[sj][sb][k] -= 1;
                    }
                    for (int jj : lat.atoms_shell_atom[b][k]) {
                        if (initialPlacement[jj] >= 0) continue;
                        if (jj == a) continue;
                        int sj = currnest[jj];
                        pairGr[sb][sj][k] += 1;
                        pairGr[sj][sb][k] += 1;
                        pairGr[sa][sj][k] -= 1;
                        pairGr[sj][sa][k] -= 1;
                    }
                }
            }
        }
        _nests[i] = currnest;
    }
    return 0;
}

// Simulated-annealing MC kernel. Same swap-proposal mechanism and
// delta-fitness machinery as empty_nests_fast, but the acceptance
// rule is Metropolis: accept downhill always; accept uphill with
// probability exp(-(newFitness - oldFitness) / T). T is the
// process-level g_sa_current_T, set by main() at the start of each
// outer iteration via an exponential schedule between SA_T_INITIAL
// and SA_T_FINAL.
//
// SA is intended as an alternative MC kernel to compare against the
// cuckoo-style greedy+epsilon-band path. The cuckoo-search outer
// loop (get_cuckoos, get_best_nest) still runs around this kernel,
// so calling this is really "SA inside a cuckoo population" -- not
// pure SA, but the inner acceptance is Metropolis. For pure SA, set
// NESTS=1 to disable the cuckoo population dynamics.
int empty_nests_sa(vector<vector<int>> &_nests, vector<vector<int>> &_newnests, int iter, vector<double> &origfitness, const Lattice &lat, const int _typesatoms, const vector<int> eachElement, int &max_num_neighs, vector<double> &weights, const vector<vector<vector<double>>> &targetSROpair, const vector<int> &initialPlacement)
{
    const int n_atoms = (int) _nests[0].size();
    const double T = g_sa_current_T;
    const double inv_T = (T > 0.0) ? (1.0 / T) : 0.0;

    static thread_local std::mt19937 gen_s, gen1_s;
    static thread_local bool seeded_s = false;
    if (!seeded_s) {
        if (g_rng_seed_base != 0) {
            gen_s.seed(g_rng_seed_base ^ 0x5C5C5C5CUL);
            gen1_s.seed(g_rng_seed_base ^ 0xC5C5C5C5UL);
        } else {
            std::random_device rd, rd1;
            gen_s.seed(rd());
            gen1_s.seed(rd1());
        }
        seeded_s = true;
    }
    std::uniform_int_distribution<> dis(0, n_atoms - 1);
    std::uniform_real_distribution<double> dis_real(0.0, 1.0);

    // Sublattice statistics + expected matrix (same as empty_nests_fast).
    int N_M = 0;
    vector<int> eachElement_M(_typesatoms, 0);
    for (int i = 0; i < n_atoms; i++) {
        if (initialPlacement[i] < 0) {
            N_M += 1;
            eachElement_M[_nests[0][i]] += 1;
        }
    }
    if (N_M == 0) return 0;

    int ref = 0;
    while (ref < n_atoms && initialPlacement[ref] >= 0) ref++;
    vector<int> z_M(max_num_neighs, 0);
    for (int k = 0; k < max_num_neighs; k++) {
        if (k < (int) lat.atoms_shell_atom[ref].size()) {
            for (int j : lat.atoms_shell_atom[ref][k]) {
                if (initialPlacement[j] < 0) z_M[k] += 1;
            }
        }
    }

    vector<vector<vector<double>>> expected(_typesatoms,
        vector<vector<double>>(_typesatoms, vector<double>(max_num_neighs, 0.0)));
    for (int k = 0; k < max_num_neighs; k++) {
        if (z_M[k] == 0) continue;
        for (int i = 0; i < _typesatoms; i++) {
            if (eachElement_M[i] == 0) continue;
            for (int j = 0; j < _typesatoms; j++) {
                if (i == j) continue;
                if (eachElement_M[j] == 0) continue;
                double alpha = targetSROpair[i][j][k];
                expected[i][j][k] = (double) eachElement_M[i] * (1.0 - alpha)
                                  * (double) z_M[k] * (double) eachElement_M[j]
                                  / (double) N_M;
            }
        }
    }

    auto fitness_from_pairGr = [&](const vector<vector<vector<double>>>& pgr) -> double {
        double err = 0.0;
        for (int k = 0; k < max_num_neighs; k++) {
            double e_k = 0.0;
            for (int i = 0; i < _typesatoms; i++) {
                for (int j = 0; j < _typesatoms; j++) {
                    if (i == j) continue;
                    if (pgr[i][j][k] == 0.0) continue;  // legacy contract
                    double d = pgr[i][j][k] - expected[i][j][k];
                    e_k += d * d;
                }
            }
            err += weights[k] * e_k;
        }
        return std::sqrt(err);
    };

    auto build_pairGr_full = [&](const vector<int>& nest,
                                  vector<vector<vector<double>>>& pgr) {
        for (auto& a : pgr) for (auto& b : a) std::fill(b.begin(), b.end(), 0.0);
        for (int i = 0; i < n_atoms; i++) {
            if (initialPlacement[i] >= 0) continue;
            for (int k = 0; k < max_num_neighs; k++) {
                for (int j : lat.atoms_shell_atom[i][k]) {
                    if (initialPlacement[j] >= 0) continue;
                    pgr[nest[i]][nest[j]][k] += 1;
                }
            }
        }
    };

    vector<vector<vector<double>>> pairGr(_typesatoms,
        vector<vector<double>>(_typesatoms, vector<double>(max_num_neighs, 0.0)));

    for (int i = 0; i < (int) _nests.size(); i++) {
        vector<int> currnest = _nests[i];
        build_pairGr_full(currnest, pairGr);
        double oldFitness = fitness_from_pairGr(pairGr);
        origfitness[i] = oldFitness;

        for (int step = 0; step < iter; step++) {
            int a = dis(gen_s);
            int b = dis(gen_s);
            if (initialPlacement[a] >= 0 || initialPlacement[b] >= 0) continue;
            int sa = currnest[a], sb = currnest[b];
            if (sa == sb) continue;

            // Apply swap delta (same as empty_nests_fast).
            for (int k = 0; k < max_num_neighs; k++) {
                for (int jj : lat.atoms_shell_atom[a][k]) {
                    if (initialPlacement[jj] >= 0) continue;
                    if (jj == b) continue;
                    int sj = currnest[jj];
                    pairGr[sa][sj][k] -= 1;
                    pairGr[sj][sa][k] -= 1;
                    pairGr[sb][sj][k] += 1;
                    pairGr[sj][sb][k] += 1;
                }
                for (int jj : lat.atoms_shell_atom[b][k]) {
                    if (initialPlacement[jj] >= 0) continue;
                    if (jj == a) continue;
                    int sj = currnest[jj];
                    pairGr[sb][sj][k] -= 1;
                    pairGr[sj][sb][k] -= 1;
                    pairGr[sa][sj][k] += 1;
                    pairGr[sj][sa][k] += 1;
                }
            }

            double newFitness = fitness_from_pairGr(pairGr);

            // Metropolis acceptance. Downhill always accepted; uphill
            // accepted with probability exp(-delta / T). When T <= 0
            // this degenerates to pure greedy.
            bool accept = false;
            double delta = newFitness - oldFitness;
            if (delta <= 0.0) {
                accept = true;
            } else if (inv_T > 0.0) {
                double p = std::exp(-delta * inv_T);
                if (dis_real(gen1_s) < p) accept = true;
            }

            if (accept) {
                currnest[a] = sb;
                currnest[b] = sa;
                oldFitness = newFitness;
                origfitness[i] = newFitness;
            } else {
                // Undo the delta in pairGr (currnest unchanged so sa/sb
                // and the captured sj values still refer to original).
                for (int k = 0; k < max_num_neighs; k++) {
                    for (int jj : lat.atoms_shell_atom[a][k]) {
                        if (initialPlacement[jj] >= 0) continue;
                        if (jj == b) continue;
                        int sj = currnest[jj];
                        pairGr[sa][sj][k] += 1;
                        pairGr[sj][sa][k] += 1;
                        pairGr[sb][sj][k] -= 1;
                        pairGr[sj][sb][k] -= 1;
                    }
                    for (int jj : lat.atoms_shell_atom[b][k]) {
                        if (initialPlacement[jj] >= 0) continue;
                        if (jj == a) continue;
                        int sj = currnest[jj];
                        pairGr[sb][sj][k] += 1;
                        pairGr[sj][sb][k] += 1;
                        pairGr[sa][sj][k] -= 1;
                        pairGr[sj][sa][k] -= 1;
                    }
                }
            }
        }
        _nests[i] = currnest;
    }
    return 0;
}

vector<double> fitnessofnests(vector<vector<int>> nests, const Lattice &lat, int _typesatoms, vector<int> eachElement, int &max_num_neighs, vector<double> &weights, const vector<vector<vector<double>>> &targetSROpair, const vector<int> &initialPlacement)
{
    vector<double> fitness(nests.size(), INT_MAX);
    for(std::size_t i = 0; i < nests.size(); i++)
    {
        fitness[i] = fitnessvalue(nests[i], lat, _typesatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
    }
    return fitness;
}

vector<vector<int>> randomizeLattice(int numofnests, int _typeofatoms, int _size, vector<int> _maxAtomNum, int numofatoms, const vector<int> &initialPlacement)
{
    // initialPlacement[i] = -1 for atoms whose species should be randomized,
    // or a species index (>=0) for atoms whose species is fixed (spectator
    // sublattice positions). For atoms with fixed species, _maxAtomNum is
    // consumed upfront in the spectator pass; the random pass then only
    // touches the free positions and may not draw spectator species.
    // Reproducible-seed-aware RNG, same pattern as in empty_nests.
    static thread_local std::mt19937 gen;
    static thread_local bool seeded = false;
    if (!seeded) {
        if (g_rng_seed_base != 0) {
            gen.seed(g_rng_seed_base ^ 0x5A5A5A5AUL);
        } else {
            std::random_device rd;
            gen.seed(rd());
        }
        seeded = true;
    }
    std::uniform_int_distribution<> dis(0, _typeofatoms - 1);
    vector<vector<int>> nests(numofnests, vector<int>(numofatoms, 0));
    vector<int> _numofatoms(_size, 0);
    vector<int> atomNum(_typeofatoms, 0);
    int atom;
    for (int j = 0; j < numofnests; j++) {
        // Spectator pass: place fixed atoms at their declared species.
        for (int k = 0; k < _size; k++) {
            if (initialPlacement[k] >= 0) {
                _numofatoms[k] = initialPlacement[k];
                atomNum[initialPlacement[k]] += 1;
            }
        }
        // Random pass: fill free positions with non-spectator species.
        // A draw that hits an already-maxed species (which includes any
        // spectator species, since the spectator pass exhausted their
        // budget) is rejected; loop until a feasible draw lands.
        int i = 0;
        while (i < _size) {
            if (initialPlacement[i] >= 0) {
                i++;  // already placed by spectator pass
                continue;
            }
            atom = (int)dis(gen);
            if (atomNum[atom] < _maxAtomNum[atom]) {
                _numofatoms[i] = atom;
                atomNum[atom] += 1;
                i++;
            }
        }
        nests[j] = _numofatoms;
        std::fill(_numofatoms.begin(), _numofatoms.end(), 0);
        std::fill(atomNum.begin(), atomNum.end(), 0);
    }
    return nests;
}

/*
 
 This par below oldbased on fitness randomize and sorts the best nests for global optimization.
.
 
 */

int get_cuckoos(vector<vector<int>> &_nests, vector<vector<int>> &_newnests, int iter, int MCiter, const Lattice &lat, int _typesatoms, const vector<int> &eachElement, const vector<int> &maxAtomNum, int &max_num_neighs, vector<double> &weights, int nproc, const vector<vector<vector<double>>> &targetSROpair, const vector<int> &initialPlacement)
{
    int numofatoms = _nests[0].size();
    int numofnests = _nests.size();
    int localiter = (36 + numofnests)/(int)nproc; // This decides the number of MC steps for local optimization
    vector<vector<int> > localnests(randomizeLattice(localiter, _typesatoms, numofatoms, maxAtomNum, numofatoms, initialPlacement));
    vector<double> localfitness(localiter, INT_MAX);
    vector<double> fitness(numofnests, INT_MAX);
    empty_nests(localnests, localnests, 8*MCiter, localfitness, lat, _typesatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
    sortthenests(localnests, localfitness, max_num_neighs);
    std::copy ( localnests.begin(), localnests.begin() + numofnests, _newnests.begin());
    empty_nests(_newnests, _newnests, iter, fitness, lat, _typesatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
    sortthenests(_newnests, fitness, max_num_neighs);
    cout << " NEW NESTS FITNESS =<----1----> " << fitness[0] << " <----2----> " << fitness[1] << " <----3----> " << fitness[2] << endl;
    return 0;
}
void printFinalCoords2File(vector<int> nest, const Lattice &lat, int _typesofatoms, vector<int> _eachElements)
{
    ofstream finalcoords;
    vector<string> coords(_typesofatoms, "");
    int numofatoms = nest.size();
    finalcoords.open("./SCRAPS.vasp", ios::out); //Calculates and writes the final SCRAPS supercell using generalized Hybrid Cuckoo program
    if(finalcoords.is_open()){
        finalcoords << "Generated-Hybrid-Cuckoo-Search" << "\n" << "1.0" << "\n" ;
        finalcoords << "\t" << lat.celldim[0]*lat.basisset.a.x << "\t" << lat.celldim[0]*lat.basisset.a.y << "\t" << lat.celldim[0]*lat.basisset.a.z << "\n" ;
        finalcoords << "\t" << lat.celldim[1]*lat.basisset.b.x << "\t" << lat.celldim[1]*lat.basisset.b.y << "\t" << lat.celldim[1]*lat.basisset.b.z << "\n" ;
        finalcoords << "\t" << lat.celldim[2]*lat.basisset.c.x << "\t" << lat.celldim[2]*lat.basisset.c.y << "\t" << lat.celldim[2]*lat.basisset.c.z << "\n" ;
        for(int i = 0; i < _typesofatoms; i++)
        {
            finalcoords << _eachElements[i] << "   ";
        }
        finalcoords << "\n" << "Cartesian" << "\n";
        for(int i = 0; i < numofatoms; i++){
            coords[nest[i]] += to_string(lat.supercell[i].x) + " \t"  +  to_string(lat.supercell[i].y) + "\t" + to_string(lat.supercell[i].z) + "\n";
            cout << "   " << lat.supercell[i].x << "  " << lat.supercell[i].y << "   " << lat.supercell[i].z << endl;
        }
        for(int i = 0; i < _typesofatoms; i++)
        {
            finalcoords << coords[i];
        }
        
    }
    else
        cout << "Unable to open file";
    
}
/*
Parallaization is done over numbers of nests during global optimization to 
speed up the SCRAPs generation
*/
int main(int argc, char ** argv)
{
    auto started = std::chrono::high_resolution_clock::now();
    readInput inputfile;
    int id, nproc;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);  // get number of total nodes
    MPI_Comm_rank(MPI_COMM_WORLD, &id); // get id of mynode

    // Singleton-mode detection: if this process reports nproc == 1 but
    // the OMPI bootstrap env vars say more were spawned, we're running
    // as one of N silently-uncoordinated singletons (typical when the
    // user launched via a mpirun from a *different* MPI than the one
    // SCRAPS was linked against -- e.g. anaconda's MPICH Hydra mpirun
    // with a SCRAPS built against Homebrew Open MPI). Each "rank" then
    // runs an independent search, the final SCRAPS.vasp is whichever
    // process writes last, and all but one rank's compute is wasted.
    // Print a loud, one-shot warning so the user can fix the launcher.
    if (nproc == 1) {
        const char *ompi_size = getenv("OMPI_COMM_WORLD_SIZE");
        const char *pmi_size = getenv("PMI_SIZE");
        const char *expected = ompi_size ? ompi_size : pmi_size;
        if (expected && std::atoi(expected) > 1) {
            std::cout << "==================================================\n"
                      << "WARNING: SCRAPS is running in SINGLETON MODE.\n"
                      << "  This process reports nproc=1 but the launcher\n"
                      << "  spawned " << expected << " processes. They cannot\n"
                      << "  communicate via MPI -- each is doing an\n"
                      << "  independent search; only one's output is kept.\n"
                      << "  Likely cause: SCRAPS was built against one MPI\n"
                      << "  (e.g. Homebrew Open MPI) and launched with a\n"
                      << "  different mpirun (e.g. anaconda's MPICH Hydra).\n"
                      << "  Fix: use the mpirun shipped with the same MPI\n"
                      << "  as the mpic++ in build.sh. On Homebrew macOS\n"
                      << "  that is /opt/homebrew/bin/mpirun.\n"
                      << "==================================================" << std::endl;
        }
    }

    // Reproducibility: if RNG_SEED was set in INPUT, derive a per-rank
    // seed and stash it in the file-scope `g_rng_seed_base` that
    // empty_nests / randomizeLattice consult on first use. The xor with
    // a multiplier of `id` keeps ranks decorrelated under the same
    // base seed.
    if (inputfile.rngSeed != 0) {
        g_rng_seed_base = inputfile.rngSeed ^ (static_cast<unsigned long>(id) * 0x9E3779B97F4A7C15UL);
        if (id == 0) {
            std::cout << "RNG_SEED=" << inputfile.rngSeed
                      << " (per-rank seed = base XOR rank*phi64)" << std::endl;
        }
    }
    g_fast_swaps = (inputfile.fastSwaps != 0);
    if (id == 0 && !g_fast_swaps) {
        std::cout << "FAST_SWAPS=0: using legacy full-recompute MC path" << std::endl;
    }
    g_targeted_swaps = (inputfile.targetedSwaps != 0);
    g_targeted_swap_prob = inputfile.targetedSwapProb;
    if (id == 0 && g_targeted_swaps) {
        std::cout << "TARGETED_SWAPS=1 (bias prob = " << g_targeted_swap_prob
                  << "): proposing biased swaps for the worst-residual species pair"
                  << std::endl;
        std::cout << "  NOTE: uniform-random swap proposal already converges fast on most"
                     "\n  test cases; targeted swaps may add overhead without speedup."
                     "\n  Recommended only for very large supercells or complex multi-pair"
                     "\n  SRO targets where uniform random struggles to find improving swaps."
                  << std::endl;
    }
    g_use_pt = (inputfile.mcKernel == "pt");
    g_use_sa = g_use_pt || (inputfile.mcKernel == "sa");
    g_sa_t_initial = inputfile.saTInitial;
    g_sa_t_final = inputfile.saTFinal;
    g_pt_exchange_interval = inputfile.ptExchangeInterval;
    if (g_use_pt) {
        // Each rank gets its own fixed log-spaced T in
        // [saTFinal, saTInitial]. Rank 0 = hottest, rank nproc-1 = coldest.
        // With one rank, falls through to plain SA at saTInitial.
        if (nproc <= 1) {
            g_sa_current_T = g_sa_t_initial;
        } else {
            double ratio = g_sa_t_final / g_sa_t_initial;
            double frac = (double) id / (double) (nproc - 1);
            g_sa_current_T = g_sa_t_initial * std::pow(ratio, frac);
        }
        std::cout << "MC_KERNEL=pt: parallel tempering. rank " << id
                  << " runs at fixed T = " << g_sa_current_T
                  << "; exchange every " << g_pt_exchange_interval
                  << " iter(s)." << std::endl;
    } else if (g_use_sa) {
        g_sa_current_T = g_sa_t_initial;
        if (id == 0) {
            std::cout << "MC_KERNEL=sa: simulated annealing inner loop. "
                      << "T schedule = " << g_sa_t_initial
                      << " -> " << g_sa_t_final << " (exponential)"
                      << std::endl;
        }
    }

//  *****READ INPUT FROM "INPUT" and "POSCAR" FILES******
    
    //  ###### INITIALIZE THE SCRAPS GENERATION######
    int iters = 0;
    double error = INT_MAX, epsi = 0.001;
    int globalIter = inputfile.globalIter;
    int num_of_nests = inputfile.num_of_nests;
    int MCiter = inputfile.MCiter;
    int iterTotal = inputfile.iterTotal;

    //####    LATTICE INPUT #######
    
    point3d a = inputfile.a;
    point3d b = inputfile.b;
    point3d c = inputfile.c;
    latstruct basis(a, b, c);
    vector<int>dim = inputfile.cell_dim;
    vector<point3d> atoms = inputfile.atoms;
    int typeofatoms = inputfile.typeofatoms;
    vector<int> eachElement = inputfile.eachElement;
    vector<double> weights = inputfile.weights; // Weights can be defined later on:
    int max_num_neighs = inputfile.num_of_shells; // Maximum number of neighbours can be changed here
    const vector<vector<vector<double>>> &targetSROpair = inputfile.targetSROpair;
    
    //point3d atom1(-0.000001613, 1.838152961, 1.312359929), atom2(1.591887967 , 0.919075084, 3.937079787);
    //vector<point3d> atoms  = inputfile.atoms;
    

    cout << "Creating Lattice....." << endl;
    Lattice templat(basis, dim, atoms);
    cout << "Lattice Built....." << endl;

    // Validate lattice supports the Warren-Cowley formulation before
    // running optimization. coordination_per_shell() aborts with a
    // sublattice-guidance diagnostic if any non-spectator atom's per-shell
    // coordination differs from the reference (sigma, Laves, fluorite-
    // without-SPECTATOR_BASIS_SPECIES, layered, etc.). Running this here
    // lets us fail fast rather than after MCMC compute. The check is
    // delayed until after we build initialPlacement below, so spectator
    // atoms can be excluded.

// Switch to maximum shell number if exceeds MAXIMUM allowed in given lattice
    if(templat.maxneigh < max_num_neighs)
        max_num_neighs = templat.maxneigh;
    if((int)weights.size() < max_num_neighs)
    {
        cout << "WEIGHTS SIZE (" << weights.size()
             << ") LESS THAN SHELL NUMBER (" << max_num_neighs
             << "). STOPPING......." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    vector<vector<double>> distances = templat.distanceMatrix();
    vector<vector<int>> neighs = templat.neighbours;
    unordered_map<int, int> numneighs;
    int total_atoms = atoms.size()*dim[0]*dim[1]*dim[2];

    // Heuristic 2: auto-scale MCSTEP from supercell size if AUTO_MCSTEP is set.
    // MCiter (= INPUT keyword MCSTEP) is the number of MC swap attempts per
    // outer iteration per nest. The right scaling for "fixed sampling per
    // atom" is k * N with k ~ 5-10; default factor 7 in our cell-size sweep.
    // We override only when autoMcstepFactor > 0 so the explicit MCSTEP
    // keyword remains the source of truth otherwise.
    if (inputfile.autoMcstepFactor > 0) {
        int auto_mciter = std::max(50, inputfile.autoMcstepFactor * total_atoms);
        if (id == 0) {
            cout << "AUTO_MCSTEP=" << inputfile.autoMcstepFactor
                 << " -> MCSTEP = max(50, " << inputfile.autoMcstepFactor
                 << " * " << total_atoms << ") = " << auto_mciter
                 << " (overriding INPUT MCSTEP=" << MCiter << ")" << endl;
        }
        MCiter = auto_mciter;
    }

    // Sanity check: ELEMENT counts must sum to total supercell atoms.
    // Otherwise randomizeLattice() falls into an infinite loop trying to
    // place atoms into species buckets whose capacity is already full
    // (or never reaches the cell size if sums fall short). Hangs with
    // no error were repeatedly traced to this user-input mismatch.
    int sum_element = 0;
    for (int v : eachElement) sum_element += v;
    if (sum_element != total_atoms) {
        cout << "ERROR: ELEMENT counts sum to " << sum_element
             << " but supercell has " << total_atoms << " atoms"
             << " (" << atoms.size() << " basis atoms x "
             << dim[0] << "x" << dim[1] << "x" << dim[2] << " cells)."
             << " Aborting." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Build initialPlacement: -1 for atoms whose species is free to be
    // randomized, or a species index for atoms whose basis position is
    // marked as a spectator-sublattice site. The Lattice supercell is
    // unrolled basis-OUTERMOST in Lattice.cpp's initializer (all atoms
    // of basis 0 first, then all of basis 1, ...), so basis position
    // bi covers atom indices [bi*n_cells, (bi+1)*n_cells).
    const int basis_size = (int) atoms.size();
    const int n_cells = dim[0] * dim[1] * dim[2];
    vector<int> initialPlacement(total_atoms, -1);
    for (const auto &spec : inputfile.spectatorBasisSpecies) {
        const int basis_idx = spec.first;
        const int species_idx = spec.second;
        if (basis_idx < 0 || basis_idx >= basis_size) {
            cout << "ERROR: SPECTATOR_BASIS_SPECIES basis index "
                 << basis_idx << " out of range [0, " << basis_size
                 << "). Aborting." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        if (species_idx < 0 || species_idx >= typeofatoms) {
            cout << "ERROR: SPECTATOR_BASIS_SPECIES species index "
                 << species_idx << " out of range [0, " << typeofatoms
                 << "). Aborting." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        const int start = basis_idx * n_cells;
        const int end = start + n_cells;
        for (int atom_idx = start; atom_idx < end; ++atom_idx) {
            initialPlacement[atom_idx] = species_idx;
        }
    }
    // Cross-check: per-species ELEMENT counts must accommodate the
    // spectator placements (each spectator species needs at least
    // n_cells * num_spectator_basis_positions_for_that_species atoms).
    {
        vector<int> spectatorCount(typeofatoms, 0);
        for (int v : initialPlacement) {
            if (v >= 0) spectatorCount[v] += 1;
        }
        for (int s = 0; s < typeofatoms; ++s) {
            if (spectatorCount[s] > eachElement[s]) {
                cout << "ERROR: spectator placement requires "
                     << spectatorCount[s] << " atoms of species " << s
                     << " but ELEMENT only allocates " << eachElement[s]
                     << ". Aborting." << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }

    // Validate lattice supports the Warren-Cowley formulation on the
    // *designed* (non-spectator) sublattice. Aborts with a sublattice-
    // guidance diagnostic if non-spectator atoms have non-uniform per-shell
    // coordination. Runs after initialPlacement is built so HEB-type cases
    // can declare their spectator sublattice and pass.
    vector<int> z_per_shell = templat.coordination_per_shell(initialPlacement);

    // Feasibility check (rank 0 only): warn the user if the configured
    // ELEMENT counts make any weighted target pair count non-integer, in
    // which case the SCRAPS fitness has a strictly positive lower bound
    // and the optimizer will plateau above zero no matter how long it
    // runs. This is a number-theoretic property of (N_i, z_k, N_j, N),
    // not a search-time issue.
    //
    // For the default SQS target (no SRO keyword, all alpha = 0), the
    // expected pair count is T_ij^k = N_i * z_k * N_j / N. If T_ij^k is
    // not integer, the best achievable error^2 for that pair is
    // (round(T) - T)^2 * 2 (factor of 2: pair[i][j] = pair[j][i]).
    //
    // Per-pair SRO targets shift T by (1 - alpha_ij) but the integer-
    // divisibility constraint still applies; the warning is printed
    // assuming alpha=0 since that's the common case for which the
    // floor is most surprising to users.
    if (id == 0) {
        // M = non-spectator subset: start from ELEMENT and subtract spectator allocations.
        vector<int> eachElement_M = eachElement;
        for (int v : initialPlacement) {
            if (v >= 0) eachElement_M[v] -= 1;
        }
        int numofatoms_M = 0;
        for (int s = 0; s < typeofatoms; ++s) numofatoms_M += eachElement_M[s];

        double fit2_floor = 0.0;
        int n_nonint = 0;
        int n_shells_check = std::min(max_num_neighs, (int) z_per_shell.size());
        for (int k = 0; k < n_shells_check; ++k) {
            double w = (k < (int) weights.size()) ? weights[k] : 1.0;
            if (w == 0.0) continue;
            const double z = (double) z_per_shell[k];
            if (z <= 0) continue;
            for (int i = 0; i < typeofatoms; ++i) {
                if (eachElement_M[i] == 0) continue;
                for (int j = 0; j < typeofatoms; ++j) {
                    if (i == j) continue;
                    if (eachElement_M[j] == 0) continue;
                    double target = (double) eachElement_M[i] * z
                                    * (double) eachElement_M[j] / (double) numofatoms_M;
                    double gap = target - std::round(target);
                    if (std::fabs(gap) > 1e-9) {
                        fit2_floor += w * gap * gap;
                        if (n_nonint < 6) {  // show up to first 6 to avoid log noise
                            cout << "  shell " << k << " pair[" << i << "][" << j
                                 << "]: target=" << target
                                 << " (nearest integer " << std::round(target)
                                 << ", off by " << gap << ")" << endl;
                        }
                        n_nonint += 1;
                    }
                }
            }
        }
        if (n_nonint > 0) {
            cout << "WARNING: " << n_nonint << " weighted target pair count(s) "
                 << "are non-integer for this (ELEMENT, Cell_DIM, MAXSHELLNUM)." << endl
                 << "  Minimum achievable fitness >= sqrt(" << fit2_floor
                 << ") = " << std::sqrt(fit2_floor) << "." << endl
                 << "  Exact SQS (fitness = 0) is mathematically unreachable on this cell."
                 << endl
                 << "  Use tools/suggest_cell.py to find the nearest feasible Cell_DIM"
                 << endl
                 << "  (no extra options needed; pass the same POSCAR + composition)."
                 << endl;
        }
    }

    vector<int> maxAtomNum(typeofatoms, 0);
    vector<int> atomNum(typeofatoms, 0);

    std::uniform_int_distribution<> dis(0, typeofatoms - 1);
    for(int i = 0; i < typeofatoms; i++)
    {
        atomNum[i] = 0;
        maxAtomNum[i] = eachElement[i];
    }
    
// Global Optimization - The Cuckoo search algorithm starts here
// Set the number of nests
     cout << "Starting Optimization ....." << endl;
    double fmin, fmax, fnew, pa;
    int best_max, best_new;
    int startval = num_of_nests*id/nproc + 1;
    int endval = num_of_nests*(id + 1)/nproc;
    int nestsperproc = endval - startval + 1;
    vector<vector<int>> nests(randomizeLattice(nestsperproc, typeofatoms, total_atoms, maxAtomNum, total_atoms, initialPlacement));
    vector<vector<int >> new_nests(randomizeLattice(nestsperproc, typeofatoms, total_atoms, maxAtomNum, total_atoms, initialPlacement));

    // Warm-start: if INIT_NEST is set, read a per-atom species vector from
    // the file and use it to seed nests[0] on every rank. The MC then
    // explores around that nest instead of from a uniform random init.
    //
    // File format: total_atoms whitespace-separated nonneg integers in
    // [0, typeofatoms), in the same atom order SCRAPS uses internally
    // (basis-OUTERMOST: all basis-0 atoms across the supercell, then all
    // basis-1, ...). Blank lines and '#' comment lines are tolerated.
    //
    // Validation: total count must match total_atoms; per-species counts
    // must match ELEMENT; spectator atoms (initialPlacement[i] >= 0)
    // must keep their fixed species. Any mismatch aborts via MPI_Abort
    // with a clear message rather than silently optimizing a wrong nest.
    if (!inputfile.initNestFile.empty()) {
        std::ifstream nf(inputfile.initNestFile.c_str());
        if (!nf.is_open()) {
            if (id == 0) {
                cout << "ERROR: INIT_NEST file not found: "
                     << inputfile.initNestFile << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        vector<int> init_nest;
        init_nest.reserve(total_atoms);
        std::string line;
        while (std::getline(nf, line)) {
            std::size_t first_ns = line.find_first_not_of(" \t");
            if (first_ns == std::string::npos) continue;        // blank line
            if (line[first_ns] == '#') continue;                // comment
            std::stringstream ssn(line);
            int v;
            while (ssn >> v) init_nest.push_back(v);
        }
        if ((int) init_nest.size() != total_atoms) {
            if (id == 0) {
                cout << "ERROR: INIT_NEST has " << init_nest.size()
                     << " entries but supercell has " << total_atoms
                     << " atoms." << endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        vector<int> per_species(typeofatoms, 0);
        for (int i = 0; i < total_atoms; ++i) {
            int s = init_nest[i];
            if (s < 0 || s >= typeofatoms) {
                if (id == 0) {
                    cout << "ERROR: INIT_NEST atom " << i << " has species "
                         << s << ", out of range [0, " << typeofatoms << ")."
                         << endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            per_species[s] += 1;
            if (initialPlacement[i] >= 0 && initialPlacement[i] != s) {
                if (id == 0) {
                    cout << "ERROR: INIT_NEST atom " << i << " has species "
                         << s << " but the spectator sublattice fixes it to "
                         << initialPlacement[i] << "." << endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        for (int s = 0; s < typeofatoms; ++s) {
            if (per_species[s] != eachElement[s]) {
                if (id == 0) {
                    cout << "ERROR: INIT_NEST species " << s << " count = "
                         << per_species[s] << " but ELEMENT says "
                         << eachElement[s] << "." << endl;
                }
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        if (!nests.empty()) nests[0] = init_nest;
        if (!new_nests.empty()) new_nests[0] = init_nest;
        if (id == 0) {
            cout << "INIT_NEST: seeded nests[0] on every rank from "
                 << inputfile.initNestFile << " (" << total_atoms
                 << " atoms)." << endl;
        }
    }

    vector<double> fitness(nestsperproc, INT_MAX);
    fitness = fitnessofnests(nests, templat, typeofatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
    // Diagnostic: report initial (pre-MCMC) fitness so the user can see
    // whether optimization is doing meaningful work vs. the random init
    // already landing close to the target. The fitness vector here is
    // the fitness of each per-rank nest after randomization.
    double init_min = fitness[0], init_max = fitness[0];
    for (double f : fitness) { init_min = std::min(init_min, f); init_max = std::max(init_max, f); }
    cout << " INITIAL FITNESS (post-randomize, pre-MCMC) on rank " << id
         << ": min=" << init_min << " max=" << init_max
         << " across " << fitness.size() << " nests" << endl;
    vector<int> best(total_atoms, 0), bestnest(total_atoms, 0);
    best_max = get_best_nest(nests, new_nests, fitness, templat, typeofatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);

//    empty_nests(nests, new_nests, MCiter, fitness, templat, typeofatoms, eachElement, max_num_neighs, weights, targetSROpair);
    // Heuristic 1: plateau-based early stopping. Each rank tracks its own
    // per-rank best fitness; if no improvement larger than earlyStopTol
    // for earlyStopPatience consecutive outer iterations, that rank breaks
    // out of the loop. Ranks may exit at different iterations — that's
    // safe because the post-loop MPI merge is one-shot (each rank sends
    // its nests exactly once after exiting).
    const int early_stop_patience = inputfile.earlyStopPatience;  // 0 disables
    const double early_stop_tol = inputfile.earlyStopTol;
    double best_seen = INT_MAX;
    int no_improve_count = 0;

    // Convergence telemetry: rank-0 writes per-iteration best fitness to
    // a CSV file (path from CONVERGENCE_LOG INPUT keyword). Lets users
    // plot the optimization trajectory and check whether the MC loop is
    // genuinely descending vs sitting at the initial-pool best.
    std::ofstream conv_log;
    auto t_loop_start = std::chrono::high_resolution_clock::now();
    if (id == 0 && !inputfile.convergenceLog.empty()) {
        // On systems where MPI launches multiple singletons (each
        // believing it is the lone rank 0 -- common with Open MPI on
        // macOS when PMIx isn't available), opening a single shared
        // log path would race-and-interleave across processes. When
        // we detect singleton mode (nproc_world == 1) we suffix the
        // log filename with the OS PID so each process writes its
        // own clean trajectory. In real multi-rank MPI mode, only
        // rank 0 reaches this block and writes the canonical path.
        int nproc_world = 0;
        MPI_Comm_size(MPI_COMM_WORLD, &nproc_world);
        std::string log_path = inputfile.convergenceLog;
        if (nproc_world == 1) {
            log_path += "." + std::to_string(getpid());
        }
        conv_log.open(log_path);
        conv_log << "iteration,best_fitness,wall_seconds" << std::endl;
        // Record the pre-MC starting point (iteration -1) for the random
        // baseline comparison. Use fitness[0] (best of rank-0's random
        // pool after the initial fitnessofnests+get_best_nest sort) --
        // not `error`, which is still INT_MAX at this point (only
        // updated inside the MC loop).
        conv_log << -1 << "," << fitness[0] << "," << 0.0 << std::endl;
    }

    // Ablation / baseline mode: skip the entire MC loop and return the
    // best of the initial random pool. Used to measure how much the
    // cuckoo-search MC is improving over pure random search.
    if (inputfile.randomOnly) {
        if (id == 0) {
            cout << "RANDOM_ONLY mode: skipping MC loop, returning best "
                 << "of the initial random nest pool." << endl;
        }
        iterTotal = 0;  // make the while() condition fall through
    }
    // PT requires all ranks to march in lockstep through the periodic
    // MPI_Sendrecv exchange step. The legacy `error > epsi` per-rank
    // early-exit would let a cold rank that found fitness=0 leave the
    // loop while warm ranks keep iterating into an exchange call that
    // deadlocks waiting for the missing partner. In PT mode, ignore
    // the per-rank fitness check and rely on TOTALIter alone for
    // termination; in cuckoo/SA mode the original behavior is kept.
    auto loop_condition = [&]() -> bool {
        if (iters >= iterTotal) return false;
        if (g_use_pt) return true;  // PT: only TOTALIter terminates
        return error > epsi;
    };
    while (loop_condition())
    {
        // SA temperature schedule: exponential interpolation between
        // SA_T_INITIAL (iter 0) and SA_T_FINAL (iter iterTotal-1).
        // Recomputed each outer iter so empty_nests_sa picks it up via
        // the g_sa_current_T global. No-op when MC_KERNEL=cuckoo.
        // PT keeps T fixed per rank (set at startup); only update when
        // running plain SA.
        if (g_use_sa && !g_use_pt) {
            if (iterTotal <= 1) {
                g_sa_current_T = g_sa_t_initial;
            } else {
                double frac = (double) iters / (double) (iterTotal - 1);
                if (g_sa_t_initial > 0.0 && g_sa_t_final > 0.0) {
                    double ratio = g_sa_t_final / g_sa_t_initial;
                    g_sa_current_T = g_sa_t_initial * std::pow(ratio, frac);
                } else {
                    g_sa_current_T = g_sa_t_initial * (1.0 - frac)
                                   + g_sa_t_final * frac;  // linear fallback
                }
            }
            if (id == 0) {
                cout << " SA T = " << g_sa_current_T
                     << " (iter " << iters << "/" << iterTotal << ")" << endl;
            }
        }
// The Cuckoo search algorithm: set up cuckoo search, finds best nests, do fitness test until fins the best nest
        get_cuckoos(nests, new_nests, globalIter, MCiter, templat, typeofatoms, eachElement, maxAtomNum, max_num_neighs, weights, nproc, targetSROpair, initialPlacement);
        fitness = fitnessofnests(nests, templat, typeofatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
        best_new = get_best_nest(nests, new_nests, fitness, templat, typeofatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
        fitness = fitnessofnests(nests, templat, typeofatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
        empty_nests(nests, new_nests, MCiter, fitness, templat, typeofatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
        fitness = fitnessofnests(nests, templat, typeofatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
        globalIter = globalIter + (iters+1)*MCiter;
        sortthenests(nests, fitness, max_num_neighs);
        error = fitness[0];
        cout << " FITNESS - > " << fitness[0] << "<-- iter -->" << iters << endl;
        if (id == 0 && conv_log.is_open()) {
            auto now = std::chrono::high_resolution_clock::now();
            double secs = std::chrono::duration<double>(now - t_loop_start).count();
            conv_log << iters << "," << fitness[0] << "," << secs << std::endl;
        }
        iters += 1;

        // Parallel tempering exchange step. Every PT_EXCHANGE_INTERVAL
        // outer iters, pair adjacent ranks (0-1, 2-3, ...) and attempt
        // to swap their best nest (nests[0]) using the standard PT
        // Metropolis criterion
        //   P_exchange = min(1, exp((1/T_i - 1/T_j) * (E_j - E_i))).
        // The lower-id rank of each pair decides (via its own RNG)
        // and broadcasts the decision so both ranks act consistently.
        // No-op when MC_KERNEL != "pt", when nproc < 2, or when this
        // rank has no nest (can happen when NESTS < nproc -- the
        // per-rank nest partition can produce zero-nest ranks).
        if (g_use_pt && nproc >= 2 && (iters % g_pt_exchange_interval) == 0
            && !nests.empty()) {
            int partner = id ^ 1;  // pair (0,1), (2,3), ...
            if (partner < nproc) {
                MPI_Status st;
                double my_fit = fitness[0];
                double partner_fit;
                double my_T = g_sa_current_T;
                double partner_T;
                MPI_Sendrecv(&my_fit, 1, MPI_DOUBLE, partner, 200,
                             &partner_fit, 1, MPI_DOUBLE, partner, 200,
                             MPI_COMM_WORLD, &st);
                MPI_Sendrecv(&my_T, 1, MPI_DOUBLE, partner, 201,
                             &partner_T, 1, MPI_DOUBLE, partner, 201,
                             MPI_COMM_WORLD, &st);

                int do_exchange = 0;  // 0/1 to broadcast as int
                if (id < partner) {
                    // Lower-id rank decides via PT Metropolis. Reuse a
                    // small thread_local mt19937 for the coin flip,
                    // seeded from g_rng_seed_base for reproducibility.
                    static thread_local std::mt19937 pt_gen;
                    static thread_local bool pt_seeded = false;
                    if (!pt_seeded) {
                        unsigned long s = (g_rng_seed_base != 0)
                            ? (g_rng_seed_base ^ 0xD0D0D0D0UL)
                            : std::random_device{}();
                        pt_gen.seed(s);
                        pt_seeded = true;
                    }
                    double delta_inv_T = 1.0 / my_T - 1.0 / partner_T;
                    double delta_E = partner_fit - my_fit;
                    double arg = delta_inv_T * delta_E;
                    bool accept = (arg >= 0.0);
                    if (!accept) {
                        double p = std::exp(arg);
                        std::uniform_real_distribution<double> ud(0.0, 1.0);
                        accept = (ud(pt_gen) < p);
                    }
                    do_exchange = accept ? 1 : 0;
                    MPI_Send(&do_exchange, 1, MPI_INT, partner, 202, MPI_COMM_WORLD);
                } else {
                    MPI_Recv(&do_exchange, 1, MPI_INT, partner, 202,
                             MPI_COMM_WORLD, &st);
                }

                if (do_exchange) {
                    int n = (int) nests[0].size();
                    std::vector<int> partner_nest(n, 0);
                    MPI_Sendrecv(nests[0].data(), n, MPI_INT, partner, 203,
                                 partner_nest.data(), n, MPI_INT, partner, 203,
                                 MPI_COMM_WORLD, &st);
                    nests[0] = std::move(partner_nest);
                    // Recompute this rank's fitness[0] after swap;
                    // fitness[1..] still belong to other (unswapped) nests.
                    fitness[0] = fitnessvalue(nests[0], templat, typeofatoms,
                                               eachElement, max_num_neighs,
                                               weights, targetSROpair,
                                               initialPlacement);
                    error = fitness[0];  // refresh early-stop tracker too
                    if (id == 0) {
                        cout << " PT exchange: rank " << id
                             << " <-> " << partner << " accepted." << endl;
                    }
                }
            }
        }

        // Per-rank early stop is incompatible with PT (would deadlock the
        // MPI exchange step), so we disable it when PT is enabled.
        if (early_stop_patience > 0 && !g_use_pt) {
            if (error < best_seen - early_stop_tol) {
                best_seen = error;
                no_improve_count = 0;
            } else {
                no_improve_count += 1;
            }
            if (no_improve_count >= early_stop_patience) {
                cout << "Early stop on rank " << id
                     << ": no improvement > " << early_stop_tol
                     << " for " << early_stop_patience
                     << " iterations (best = " << best_seen
                     << ", iter " << iters << "/" << iterTotal << ")." << endl;
                break;
            }
        }
    }
    vector<int> nest1D;
    if (id != 0) {
        for (int i = 0; i < nestsperproc; i++)
        {
            for(int j = 0; j < total_atoms; j++)
                nest1D.push_back(nests[i][j]);
        }
        MPI_Send(&nest1D[0], total_atoms*nestsperproc, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        for (int j = 1; j < nproc; j++) {
            startval = num_of_nests*j/nproc + 1;
            endval = num_of_nests*(j + 1)/nproc;
            nestsperproc = endval - startval + 1;
            nest1D.resize(total_atoms*nestsperproc);
            MPI_Recv(&nest1D[0], total_atoms*nestsperproc, MPI_INT, j, 0, MPI_COMM_WORLD, &status);
            for (int i = 0; i < nestsperproc; i++) {
                vector<int> temp(nest1D.begin() + i*total_atoms, nest1D.begin() + (i + 1)*total_atoms);
                nests.push_back(temp);
            }
        }
        fitness = fitnessofnests(nests, templat, typeofatoms, eachElement, max_num_neighs, weights, targetSROpair, initialPlacement);
        sortthenests(nests, fitness, max_num_neighs);
        cout << "FINAL fitness (best across all " << nests.size() << " merged nests) = "
             << fitness[0] << endl;
        printtoscreen(nests[0], templat, typeofatoms, eachElement, initialPlacement);
        printFinalCoords2File(nests[0], templat, typeofatoms, eachElement);
    //cout << "HELOO" << endl;
        auto done = std::chrono::high_resolution_clock::now();
        std::cout << "TIME TAKEN ->  ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
        std::cout << "ms \n" << endl;
    }
    MPI_Finalize();
}
/**
 Generate new random structures
 **/
