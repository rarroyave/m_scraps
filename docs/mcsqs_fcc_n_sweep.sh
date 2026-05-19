#!/bin/bash
# FCC 25:50:25 ternary, cell-size sweep: scraps-perpair vs ATAT mcsqs.
# Four sizes: n = 32, 108, 256, 500 (Cell_DIM 2..5 conventional FCC).
# SCRAPS uses MCSTEP scaled ~7 MC steps/atom; mcsqs uses -rc with an
# explicit sqscell.out for the larger cells to skip the supercell-
# enumeration phase that otherwise dominates wall time.
set -e

ROOT=/tmp/mcsqs_fcc_n
REPO=$(cd "$(dirname "$0")/.." && pwd)
SCRAPS_BIN=$REPO/SCRAPs/scraps
VERIFY=$REPO/tools/verify_sro.py
MPIRUN=/opt/homebrew/bin/mpirun

mkdir -p "$ROOT"

max_alpha_2shell() {
    local poscar="$1" species="$2"
    python3 $VERIFY "$poscar" --species "$species" --shells 2 --json 2>/dev/null \
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

# FCC primitive rndstr template with 25:50:25 composition
fcc_rndstr() {
    cat <<EOF
3.56 3.56 3.56 90 90 90
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5
0.0 0.0 0.0 A=0.25,B=0.5,C=0.25
EOF
}

# sqscell.out for FCC conventional k×k×k expressed in primitive units.
# det = 4 k^3 = total atoms (one primitive cell has 1 atom).
fcc_sqscell() {
    local k="$1"
    cat <<EOF
1

$k $k -$k
$k -$k $k
-$k $k $k
EOF
}

run_mcsqs() {
    local case_name="$1" k="$2" natom="$3" budget="$4"
    local dir="$ROOT/${case_name}_mcsqs"; mkdir -p "$dir"; cd "$dir"
    fcc_rndstr > rndstr.in
    fcc_sqscell "$k" > sqscell.out
    rm -f bestsqs.out bestcorr.out stopsqs mcsqs.log clusters.out

    # 2-shell cluster cutoff for FCC a=3.56: NN=2.517, 2NN=3.560
    mcsqs -2=3.6 > clus.log 2>&1
    local t0=$(date +%s)
    # -rc skips supercell enumeration; uses sqscell.out
    mcsqs -rc > mcsqs.log 2>&1 &
    local pid=$!
    local elapsed=0
    while kill -0 $pid 2>/dev/null && [ $elapsed -lt $budget ]; do
        sleep 2; elapsed=$((elapsed+2))
    done
    touch stopsqs
    sleep 3
    if kill -0 $pid 2>/dev/null; then kill $pid 2>/dev/null; fi
    wait $pid 2>/dev/null || true
    local t1=$(date +%s)
    if [ -s bestsqs.out ]; then
        str2cif < bestsqs.out > bestsqs.cif 2>/dev/null
        python3 -c "from pymatgen.core import Structure; Structure.from_file('bestsqs.cif').to(filename='bestsqs.poscar', fmt='poscar')" 2>/dev/null
    fi
    echo "$dir $((t1-t0))"
}

declare -a results

run_case() {
    local case_name="$1" k="$2" natom="$3" budget="$4"
    local src_test="$REPO/tests/${case_name}"

    echo "=========================================================="
    echo "Case: $case_name  (FCC conv ${k}x${k}x${k} = $natom atoms, mcsqs budget ${budget}s)"
    echo "=========================================================="

    local sc=$(run_scraps "$case_name" "$src_test")
    local sc_dir=$(echo "$sc" | awk '{print $1}')
    local sc_wall=$(echo "$sc" | awk '{print $2}')
    local sc_fit=$(grep "FINAL fitness" "$sc_dir/scraps.log" | sed -E 's/.*= //' | head -1)
    local sc_a=$(max_alpha_2shell "$sc_dir/SCRAPS.vasp" "A,B,C")

    local mc=$(run_mcsqs "$case_name" "$k" "$natom" "$budget")
    local mc_dir=$(echo "$mc" | awk '{print $1}')
    local mc_wall=$(echo "$mc" | awk '{print $2}')
    local mc_a="(no output)"
    if [ -s "$mc_dir/bestsqs.poscar" ]; then
        mc_a=$(max_alpha_2shell "$mc_dir/bestsqs.poscar" "A,B,C")
    fi

    echo "  scraps  | wall ${sc_wall}s | FINAL fit ${sc_fit:-?} | max|alpha| $sc_a"
    echo "  mcsqs   | wall ${mc_wall}s | (no fitness)        | max|alpha| $mc_a"
    results+=("| $natom | $sc_wall | ${sc_fit:-?} | $sc_a | $mc_wall | $mc_a |")
}

# Sweep: (case_dir, k, n_atoms, mcsqs_budget_s)
run_case stoich_fcc_25_50_25_n32   2  32    30
run_case stoich_fcc_25_50_25_n108  3  108   60
run_case stoich_fcc_25_50_25_n256  4  256   180
run_case stoich_fcc_25_50_25_n500  5  500   300

echo ""
echo "=========================================================="
echo "Summary: FCC 25:50:25 across cell sizes"
echo "=========================================================="
echo "| N (atoms) | sc wall(s) | sc fitness | sc max|a| | mc wall(s) | mc max|a| |"
echo "|---|---|---|---|---|---|"
for r in "${results[@]}"; do echo "$r"; done
