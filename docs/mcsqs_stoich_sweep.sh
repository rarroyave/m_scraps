#!/bin/bash
# Single-sublattice stoichiometry sweep: scraps-perpair vs ATAT mcsqs.
# Six cases: FCC and BCC at 50:50, 75:25 binary, and one ternary each.
# For each case, run both tools, then verify with the same multi-shell
# alpha calculator (verify_sro.py) and report max |alpha| across shells.
set -e

ROOT=/tmp/mcsqs_vs_scraps_stoich
REPO=$(cd "$(dirname "$0")/.." && pwd)
SCRAPS_BIN=$REPO/SCRAPs/scraps
VERIFY=$REPO/tools/verify_sro.py
MPIRUN=/opt/homebrew/bin/mpirun
MCSQS_BUDGET=${MCSQS_BUDGET:-60}  # seconds per mcsqs run

mkdir -p "$ROOT"

# ---------- helpers ----------
max_alpha_multishell() {
    local poscar="$1" species="$2" shells="$3"
    python3 $VERIFY "$poscar" --species "$species" --shells "$shells" --json 2>/dev/null \
        | python3 -c "
import json,sys
d=json.load(sys.stdin); mx=0.0
for sh in d['shells']:
    for r in sh['alpha_matrix']:
        for v in r:
            if v is not None and v==v: mx=max(mx,abs(v))
print(f'{mx:.4f}')"
}

run_scraps() {
    local case_name="$1" src_test_dir="$2"
    local dir="$ROOT/${case_name}_scraps"; mkdir -p "$dir"; cd "$dir"
    cp "$SCRAPS_BIN" ./scraps
    cp "$src_test_dir/POSCAR" "$src_test_dir/INPUT" .
    local t0=$(date +%s)
    "$MPIRUN" -np 4 ./scraps > scraps.log 2>&1
    local t1=$(date +%s)
    echo "$dir $((t1-t0))"
}

run_mcsqs() {
    local case_name="$1" natom="$2" rndstr="$3" cluster_cutoff="$4"
    local dir="$ROOT/${case_name}_mcsqs"; mkdir -p "$dir"; cd "$dir"
    printf '%s' "$rndstr" > rndstr.in
    rm -f bestsqs.out bestcorr.out stopsqs mcsqs.log clusters.out
    mcsqs -2="$cluster_cutoff" > clus.log 2>&1
    local t0=$(date +%s)
    mcsqs -n="$natom" > mcsqs.log 2>&1 &
    local pid=$!
    local elapsed=0
    while kill -0 $pid 2>/dev/null && [ $elapsed -lt $MCSQS_BUDGET ]; do
        sleep 1; elapsed=$((elapsed+1))
    done
    touch stopsqs
    sleep 3  # mcsqs polls periodically
    if kill -0 $pid 2>/dev/null; then kill $pid 2>/dev/null; fi
    wait $pid 2>/dev/null || true
    local t1=$(date +%s)
    if [ -s bestsqs.out ]; then
        str2cif < bestsqs.out > bestsqs.cif 2>/dev/null
        python3 -c "from pymatgen.core import Structure; Structure.from_file('bestsqs.cif').to(filename='bestsqs.poscar', fmt='poscar')" 2>/dev/null
    fi
    echo "$dir $((t1-t0))"
}

# ---------- shared rndstr templates (FCC primitive, BCC primitive) ----------
fcc_rndstr() {  # args: comp_spec_string
    cat <<EOF
3.56 3.56 3.56 90 90 90
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5
0.0 0.0 0.0 $1
EOF
}

bcc_rndstr() {  # args: comp_spec_string
    cat <<EOF
3.20 3.20 3.20 90 90 90
-0.5  0.5  0.5
 0.5 -0.5  0.5
 0.5  0.5 -0.5
0.0 0.0 0.0 $1
EOF
}

# 2-shell cluster cutoffs.
# FCC a=3.56: NN=2.517, 2NN=3.560. Cutoff 3.6 covers both. 3NN=4.359 excluded.
# BCC a=3.20: NN=2.771, 2NN=3.200. Cutoff 3.3 covers both. 3NN=4.525 excluded.
FCC_CUT=3.6
BCC_CUT=3.3

# ---------- cases ----------
declare -a results

run_case() {
    local name="$1" lattice="$2" natom="$3" species="$4" comp_spec="$5" shells="$6"
    local src_test="$REPO/tests/${name}"  # tests/stoich_*

    echo "=========================================================="
    echo "Case: $name ($lattice, $natom atoms, $comp_spec)"
    echo "=========================================================="

    # SCRAPS
    local sc=$(run_scraps "$name" "$src_test")
    local sc_dir=$(echo "$sc" | awk '{print $1}')
    local sc_wall=$(echo "$sc" | awk '{print $2}')
    local sc_final=$(grep "FINAL fitness" "$sc_dir/scraps.log" | sed -E 's/.*= //' | head -1)
    local sc_a=$(max_alpha_multishell "$sc_dir/SCRAPS.vasp" "$species" "$shells")

    # mcsqs
    local rndstr
    if [ "$lattice" = "fcc" ]; then rndstr=$(fcc_rndstr "$comp_spec"); CUT=$FCC_CUT
    else                            rndstr=$(bcc_rndstr "$comp_spec"); CUT=$BCC_CUT
    fi
    local mc=$(run_mcsqs "$name" "$natom" "$rndstr" "$CUT")
    local mc_dir=$(echo "$mc" | awk '{print $1}')
    local mc_wall=$(echo "$mc" | awk '{print $2}')
    local mc_a="(no output)"
    if [ -s "$mc_dir/bestsqs.poscar" ]; then
        mc_a=$(max_alpha_multishell "$mc_dir/bestsqs.poscar" "$species" "$shells")
    fi

    echo "  scraps  | wall ${sc_wall}s | FINAL fit $sc_final | max|alpha| $sc_a"
    echo "  mcsqs   | wall ${mc_wall}s | (no fitness)     | max|alpha| $mc_a"
    results+=("| $name | $natom | $comp_spec | $sc_wall | $sc_final | $sc_a | $mc_wall | $mc_a |")
}

# FCC cases (3x3x3 conventional = 108 atoms)
run_case stoich_fcc_50_50    fcc 108 "A,B"   "A=0.5,B=0.5"             2
run_case stoich_fcc_75_25    fcc 108 "A,B"   "A=0.75,B=0.25"           2
run_case stoich_fcc_50_25_25 fcc 108 "A,B,C" "A=0.5,B=0.25,C=0.25"     2

# BCC cases
# 50:50 + ternary equi use 3x3x3 = 54 atoms (compositions divide cleanly).
# 75:25 needs at least 4x4x4 = 128 (54 doesn't split into 3:1 evenly).
run_case stoich_bcc_50_50_54 bcc 54  "A,B"   "A=0.5,B=0.5"             2
run_case stoich_bcc_75_25    bcc 128 "A,B"   "A=0.75,B=0.25"           2
run_case stoich_bcc_equi3_54 bcc 54  "A,B,C" "A=0.3333333,B=0.3333333,C=0.3333334" 2

echo ""
echo "=========================================================="
echo "Summary (max |alpha| across 2 shells, lower is better)"
echo "=========================================================="
echo "| case | atoms | composition | sc_wall(s) | scraps fit | scraps max|a| | mc_wall(s) | mcsqs max|a| |"
echo "|---|---|---|---|---|---|---|---|"
for r in "${results[@]}"; do echo "$r"; done
