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
    // initialPlacement preserves spectator-sublattice occupancy through
    // the Cuckoo Search MCMC: swaps that would move a fixed atom are
    // rejected, so the spectator pattern (e.g. boron sites in an HEB)
    // stays put for the full optimization.
    double len = _nests[0].size();
    double eps = 0.001;
    std::random_device rd, rd1;
    std::mt19937 gen(rd());
    std::mt19937 gen1(rd1());
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
    std::random_device rd;
    std::mt19937 gen(rd());
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
    (void) templat.coordination_per_shell(initialPlacement);

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
    while (iters < iterTotal && error > epsi)
    {
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
        iters += 1;

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
