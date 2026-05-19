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

#include "readInput.hpp"
#include  <iostream>

using namespace std;

readInput::readInput()
{
    input();
    readPOSCAR();
}
void readInput::readPOSCAR()
{
    totalatoms = 0;
    stringstream ss;
    int linnumb = 1;
    string line, key;
    std::ifstream inp ("./POSCAR");
    if (inp.is_open()) {
        while (inp.good()) {
            getline(inp, line);
            ss.str(line);
            if (linnumb == 1)
            {
                linnumb += 1;
                ss.clear();
                continue;
            }
            if (linnumb == 2) {
                double scale;
                ss >> scale;
                if(ss.fail())
                {
                    cout << "POSCAR not in correct format, Provide Correct POSCAR";
                    break;
                
                }
            }
            if (linnumb == 3)
            {
                double x, y, z;
                ss >> x >> y >> z;
                if(ss.fail())
                {
                    cout << "POSCAR not in correct format, Provide Correct POSCAR";
                    break;
                    
                }
                a.x = x;
                a.y = y;
                a.z = z;
            }
            if (linnumb == 4)
            {
                double x, y, z;
                ss >> x >> y >> z;
                if(ss.fail())
                {
                    cout << "POSCAR not in correct format, Provide Correct POSCAR";
                    break;
                    
                }
                b.x = x;
                b.y = y;
                b.z = z;
            }
            if (linnumb == 5)
            {
                double x, y, z;
                ss >> x >> y >> z;
                if(ss.fail())
                {
                    cout << "POSCAR not in correct format, Provide Correct POSCAR";
                    break;
                    
                }
                c.x = x;
                c.y = y;
                c.z = z;
            }
            if (linnumb == 6)
                ss >> totalatoms;
            
            if (linnumb > 7 && linnumb <= 7 + totalatoms) {
                point3d latatom;
                double x, y, z;
                ss >> x >> y >> z;
                if(ss.fail())
                {
                    cout << "POSCAR not in correct format, Provide Correct POSCAR";
                    break;
                    
                }
                latatom.x = x;
                latatom.y = y;
                latatom.z = z;
                atoms.push_back(latatom);
                
            }
            if (linnumb > 7 + totalatoms)
                break;
            ss.clear();
            linnumb += 1;
        }
    }
    else
    {
        
        std::cout << " POSCAR FILE DOES NOT EXIST " << endl;
        
    }
}


void readInput::input() {
    stringstream ss;
    string line, key;
    // Collect SRO directives during the main pass; resolve into targetSROpair
    // afterwards, once TYP and MAXSHELLNUM are known.
    struct SROEntry { int i; int j; int shell; double alpha; };
    vector<SROEntry> sroEntries;
    std::ifstream inp ("./INPUT");
    if (inp.is_open()) {
        while (inp.good()) {
            getline(inp, line);
            ss.str(line);
            // Clear `key` before each extraction. Without this, an empty
            // line at end-of-file (where getline returns "" but inp.good()
            // is still true until the next read fails) would leave key
            // holding its previous value, causing the LAST keyword's branch
            // to fire spuriously on the trailing empty line.
            key.clear();
            ss >> key;
            transform(key.begin(), key.end(), key.begin(), ::toupper);
            if(key.compare("TYP") == 0)
            {
                ss >> typeofatoms;

            }
            if(key.compare("GITER") == 0)
            {
                ss >> globalIter;
            }
            if(key.compare("ELEMENT") == 0)
            {
                int elem;
                while (ss >> elem) {
                    eachElement.push_back(elem);
                }
            }
            if(key.compare("CELL_DIM") == 0)
            {
                int dim;
                while (ss >> dim) {
                    cell_dim.push_back(dim);
                }

            }
            if(key.compare("NESTS") == 0)
            {
                ss >> num_of_nests;
            }
            if(key.compare("MCSTEP") == 0)
            {
                ss >> MCiter;
            }
            if(key.compare("TOTALITER") == 0)
            {
                ss >> iterTotal;
            }
            if(key.compare("MAXSHELLNUM") == 0)
            {
                ss >> num_of_shells;
            }
            if(key.compare("WEIGHT") == 0)
            {
                double w;
                while (ss >> w) {
                    weights.push_back(w);
                }
            }
            if(key.compare("SRO") == 0)
            {
                SROEntry e;
                if(ss >> e.i >> e.j >> e.shell >> e.alpha)
                {
                    sroEntries.push_back(e);
                }
                else
                {
                    cout << "Warning: malformed SRO line ignored: " << line << endl;
                }
            }
            if(key.compare("SPECTATOR_BASIS_SPECIES") == 0)
            {
                int basis_idx, species_idx;
                if(ss >> basis_idx >> species_idx)
                {
                    spectatorBasisSpecies.emplace_back(basis_idx, species_idx);
                }
                else
                {
                    cout << "Warning: malformed SPECTATOR_BASIS_SPECIES line ignored: "
                         << line << endl;
                }
            }
            ss.clear();
        }
    }
    else
    {

        std::cout << " INPUT FILE DOES NOT EXIST " << endl;
        return;
    }

    // Resolve spectator-sublattice configuration. Build fast-lookup sets for
    // basis positions and species. Validation against atoms.size() (basis
    // size from POSCAR) and ELEMENT counts happens in main() once both are
    // known; here we just structure the data.
    for (const auto &p : spectatorBasisSpecies) {
        spectatorBasisPositions.insert(p.first);
        spectatorSpecies.insert(p.second);
    }

    // Resolve per-pair, per-shell SRO targets. Default to zeros; symmetrize
    // user-supplied entries (alpha_ij = alpha_ji).
    int nTyp = typeofatoms;
    int nShell = num_of_shells > 0 ? num_of_shells : 5;
    targetSROpair.assign(
        nTyp,
        vector<vector<double>>(nTyp, vector<double>(nShell, 0.0))
    );
    for (const auto &e : sroEntries) {
        if (e.i < 0 || e.i >= nTyp || e.j < 0 || e.j >= nTyp) {
            cout << "Warning: SRO species index out of range (TYP=" << nTyp
                 << "): " << e.i << " " << e.j << endl;
            continue;
        }
        if (e.shell < 0 || e.shell >= nShell) {
            cout << "Warning: SRO shell index out of range (MAXSHELLNUM="
                 << nShell << "): " << e.shell << endl;
            continue;
        }
        targetSROpair[e.i][e.j][e.shell] = e.alpha;
        targetSROpair[e.j][e.i][e.shell] = e.alpha;
    }
}
