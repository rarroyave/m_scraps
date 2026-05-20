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
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

using namespace std;

readInput::readInput()
    : typeofatoms(0), globalIter(0), num_of_nests(0), MCiter(0),
      iterTotal(0), num_of_shells(0),
      earlyStopPatience(0), earlyStopTol(1e-6), autoMcstepFactor(0),
      rngSeed(0), randomOnly(0), convergenceLog(""),
      fastSwaps(1), targetedSwaps(0), targetedSwapProb(0.7),
      totalatoms(0)
{
    input();
    readPOSCAR();
}
// Defensive VASP POSCAR parser. The upstream readPOSCAR honored the
// SCRAPS-internal POSCAR dialect (unit lattice, scale=1, single count
// line, no species labels, "direct" keyword treated as Cartesian) but
// silently mis-read any of the standard VASP variants. This rewrite
// handles the full POSCAR spec we care about:
//
//   line 1: comment (ignored)
//   line 2: scale factor (positive multiplier; negative = target volume)
//   line 3-5: three lattice vectors
//   line 6: OPTIONAL species labels (e.g. "Ni Co Cr") -- detected by
//           failure of int-parse on the first token
//   line 6 or 7: per-species counts ("36" or "36 36 36" -- summed)
//   OPTIONAL line: "Selective dynamics" (presence detected by leading
//           S/s); skipped along with the per-atom T/F flags
//   next line: "Direct"/"d" or "Cartesian"/"C" coordinate-type keyword
//   next totalatoms lines: per-atom positions (x y z [T T T])
//
// Aborts via MPI_Abort + clear message on any malformed file rather
// than silently producing wrong basis positions. Stores `atoms` as
// Cartesian-Angstrom regardless of input mode.
namespace {

[[noreturn]] void poscar_abort(const std::string &msg) {
    std::cout << "ERROR: POSCAR: " << msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);  // unreachable, but quiets compilers
}

bool starts_with_letter(const std::string &line, char up, char lo) {
    for (char c : line) {
        if (c == ' ' || c == '\t') continue;
        return c == up || c == lo;
    }
    return false;
}

bool first_token_is_integer(const std::string &line) {
    std::stringstream ss(line);
    std::string token;
    if (!(ss >> token)) return false;
    if (token.empty()) return false;
    std::size_t i = (token[0] == '+' || token[0] == '-') ? 1 : 0;
    if (i >= token.size()) return false;
    for (; i < token.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(token[i]))) return false;
    }
    return true;
}

}  // namespace

void readInput::readPOSCAR()
{
    totalatoms = 0;
    atoms.clear();

    std::ifstream inp("./POSCAR");
    if (!inp.is_open()) {
        std::cout << " POSCAR FILE DOES NOT EXIST " << std::endl;
        return;
    }
    std::vector<std::string> lines;
    std::string buf;
    while (std::getline(inp, buf)) lines.push_back(buf);

    if (lines.size() < 8) {
        poscar_abort("file has fewer than 8 lines; need comment + scale + 3 vectors "
                     "+ counts + coord-type + at least one atom.");
    }

    // ---- Line 2: scale factor ----
    // Positive value = uniform multiplier on lattice vectors and (if used)
    // Cartesian coordinates. Negative value = target |unit-cell volume|.
    double raw_scale = 0.0;
    {
        std::stringstream ss(lines[1]);
        if (!(ss >> raw_scale)) poscar_abort("line 2 (scale factor) not a number.");
    }
    const bool scale_by_volume = (raw_scale < 0.0);

    // ---- Lines 3-5: lattice vectors (pre-scale) ----
    auto read_vec = [&](std::size_t li, const char *name) -> point3d {
        std::stringstream ss(lines[li]);
        double x, y, z;
        if (!(ss >> x >> y >> z))
            poscar_abort(std::string("line ") + std::to_string(li + 1)
                         + " (" + name + " vector) not 3 numbers.");
        return point3d(x, y, z);
    };
    point3d va = read_vec(2, "first lattice");
    point3d vb = read_vec(3, "second lattice");
    point3d vc = read_vec(4, "third lattice");

    // Apply scale factor uniformly. Volume mode: solve linear_scale so the
    // resulting lattice has |det| == |raw_scale|.
    double linear_scale = 0.0;
    if (scale_by_volume) {
        const double current_vol = std::abs(
            va.x * (vb.y * vc.z - vb.z * vc.y)
          - va.y * (vb.x * vc.z - vb.z * vc.x)
          + va.z * (vb.x * vc.y - vb.y * vc.x));
        if (current_vol <= 0.0) poscar_abort("degenerate lattice (zero volume).");
        linear_scale = std::cbrt(-raw_scale / current_vol);
    } else {
        linear_scale = raw_scale;
    }
    auto scale_vec = [linear_scale](const point3d &p) {
        return point3d(p.x * linear_scale, p.y * linear_scale, p.z * linear_scale);
    };
    a = scale_vec(va);
    b = scale_vec(vb);
    c = scale_vec(vc);

    // ---- Line 6: either species labels (string tokens) or counts (ints) ----
    std::size_t counts_line = 5;
    if (!first_token_is_integer(lines[5])) {
        counts_line = 6;
        if (counts_line >= lines.size())
            poscar_abort("species label line present but counts line missing.");
    }
    {
        std::stringstream ss(lines[counts_line]);
        int n;
        bool any = false;
        while (ss >> n) {
            if (n < 0) poscar_abort("negative atom count.");
            totalatoms += n;
            any = true;
        }
        if (!any) poscar_abort("counts line had no integers.");
    }
    std::size_t cursor = counts_line + 1;

    // ---- Optional "Selective dynamics" line ----
    if (cursor >= lines.size()) poscar_abort("file truncated before coord-type line.");
    if (starts_with_letter(lines[cursor], 'S', 's')) {
        cursor += 1;
        if (cursor >= lines.size())
            poscar_abort("file truncated after 'Selective dynamics'.");
    }

    // ---- Coordinate-type keyword (Direct / Cartesian) ----
    const bool is_direct    = starts_with_letter(lines[cursor], 'D', 'd');
    const bool is_cartesian = starts_with_letter(lines[cursor], 'C', 'c');
    if (!is_direct && !is_cartesian)
        poscar_abort("line " + std::to_string(cursor + 1)
                     + " must start with 'D'/'d' (Direct) or 'C'/'c' (Cartesian); got: \""
                     + lines[cursor] + "\".");
    cursor += 1;

    // ---- totalatoms position lines ----
    for (int i = 0; i < totalatoms; ++i) {
        if (cursor + i >= lines.size())
            poscar_abort("file ended after " + std::to_string(i)
                         + " of " + std::to_string(totalatoms) + " atoms.");
        std::stringstream ss(lines[cursor + i]);
        double x, y, z;
        if (!(ss >> x >> y >> z))
            poscar_abort("position line " + std::to_string(cursor + i + 1)
                         + " not 3 numbers.");
        // Trailing Selective-dynamics T/F flags after the coords are ignored.
        point3d pos;
        if (is_direct) {
            // Fractional -> Cartesian using the post-scale lattice.
            pos.x = x * a.x + y * b.x + z * c.x;
            pos.y = x * a.y + y * b.y + z * c.y;
            pos.z = x * a.z + y * b.z + z * c.z;
        } else {
            // VASP-Cartesian coordinates: also multiplied by the scale factor.
            pos.x = x * linear_scale;
            pos.y = y * linear_scale;
            pos.z = z * linear_scale;
        }
        atoms.push_back(pos);
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
            if(key.compare("AUTO_MCSTEP") == 0)
            {
                ss >> autoMcstepFactor;
            }
            if(key.compare("RNG_SEED") == 0)
            {
                ss >> rngSeed;
            }
            if(key.compare("RANDOM_ONLY") == 0)
            {
                ss >> randomOnly;
            }
            if(key.compare("CONVERGENCE_LOG") == 0)
            {
                ss >> convergenceLog;
            }
            if(key.compare("FAST_SWAPS") == 0)
            {
                ss >> fastSwaps;
            }
            if(key.compare("TARGETED_SWAPS") == 0)
            {
                ss >> targetedSwaps;
            }
            if(key.compare("TARGETED_SWAP_PROB") == 0)
            {
                ss >> targetedSwapProb;
            }
            if(key.compare("EARLY_STOP_PATIENCE") == 0)
            {
                ss >> earlyStopPatience;
            }
            if(key.compare("EARLY_STOP_TOL") == 0)
            {
                ss >> earlyStopTol;
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
