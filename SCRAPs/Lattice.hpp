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

#ifndef Lattice_hpp
#define Lattice_hpp

#include <Eigen/Dense>
#include<vector>
#include<vector>
#include<string>
#include<iostream>
#include<math.h>
#include <unordered_map>

using namespace std;
struct point3d{
public:
    double x;
    double y;
    double z;
    
    point3d():x(0.0), y(0.0), z(0.0){}
    point3d(double _x, double _y, double _z)
    : x(_x), y(_y), z(_z)
    {
    }
    point3d& operator=(const point3d &vec2);
    friend point3d operator+(const point3d &vec1,const point3d &vec2);
    friend point3d operator*(const point3d &vec1,const point3d &vec2);
    friend point3d operator*(int m,const point3d &vec2);
    friend point3d operator*(const point3d &vec1,int m);
    // Operator " - " is used to calculate distance between two points: vec1, and vec2.
    friend point3d operator-(const point3d &vec1,const point3d &vec2);
    friend double dot(const point3d &vec1, const point3d &vec2);
    friend double distance(const point3d &vec1,const point3d &vec2);
    friend point3d cross(const point3d &vec1,const point3d &vec2);
    friend double mod(const point3d &vec1);
};
struct atom{
    double x;
    double y;
    double z;
    
    string name;
    int id;
    atom():x(0.0), y(0.0), z(0.0), name("NA"), id(0)
    {
    }
    atom(double _x, double _y, double _z, string _name, int _id)
    : x(_x), y(_y), z(_z), name(_name), id(_id)
    {
    }
};

struct latstruct{
    struct point3d a;
    struct point3d b;
    struct point3d c;
    
    latstruct():a(0.0, 0.0, 0.0), b(0.0, 0.0, 0.0), c(0.0, 0.0, 0.0)
    {}
    
    latstruct(struct point3d _a, struct point3d _b, struct point3d _c):a(_a), b(_b), c(_c)
    {}
};


class Lattice{
public:
    latstruct basisset;
    vector<int> celldim;
    vector<struct point3d> atoms;
    vector<struct point3d> supercell;
    double maxneighdis;
    int maxneigh;
    Eigen::Matrix3d real;
    Eigen::Matrix3d frac;

//   Lattice():basisset(latstruct()), celldim({1, 1, 1});
    Lattice(latstruct _basisset, vector<int> dim, vector<struct point3d> _atoms,
            int requested_shells = 0);
    vector<vector<int>> neighbours;
    vector<vector<double>> distances;
    vector<vector<vector<int>>> atoms_shell_atom;
    void initialize();
    vector<vector<double>> distanceMatrix();
    vector<vector <int>> neighbourMatrix();
    void periodicNeighbourGraph(int requested_shells);
    double mindist();
    int neighmax();
    double imageDistance(point3d latpoint1, point3d latpoint2);
    vector<vector<int>> maxneighbourspershell;  // [atom_index][shell]
    vector<vector<int>> numneighofeachshell();
    // Per-shell coordination number, derived from maxneighbourspershell.
    // Asserts that every non-spectator atom has identical coordination
    // in each shell (true for any single-Bravais-lattice-with-basis
    // crystal: FCC, BCC, HCP, B2, L1_2, D0_3, etc.). Aborts via
    // MPI_Abort if a non-uniform lattice is detected.
    //
    // For HEB-type structures with a spectator sublattice (B atoms in
    // AlB_2 diborides), pass initialPlacement so spectator atoms are
    // excluded from the uniformity check; they legitimately have a
    // different coordination structure from the metal sublattice.
    //
    // For multi-sublattice structures with per-Wyckoff-orbit declared
    // SUBLATTICE blocks (v0.4.2+), pass atomToSublattice so the
    // uniformity check is applied *within* each sublattice rather
    // than across all non-spectator atoms. Different sublattices may
    // legitimately have different coordination (fluorite: cation z=8,
    // anion z=4; sigma phase: z=12-15 per Wyckoff orbit); the
    // per-sublattice fitness machinery (g_z_sub_sub_shell) handles
    // these correctly as long as each sublattice is uniform within
    // itself. atomToSublattice[i] = -1 marks atom i as spectator (or
    // outside any SUBLATTICE block); pass empty for the legacy
    // single-sublattice check.
    //
    // Return value: per-shell coordination from the reference (first
    // non-spectator) atom. Same return for single-sublattice and
    // multi-sublattice paths so callers that consume z for legacy
    // feasibility math don't break; multi-sublattice callers should
    // prefer the global g_z_sub_sub_shell tensor built in main().
    vector<int> coordination_per_shell(
        const vector<int> &initialPlacement = {},
        const vector<int> &atomToSublattice = {}) const;
    friend Lattice operator+(const Lattice &lat1, Lattice &lat2);
    void maxneighshell();
};


#endif /* Lattice_hpp */
