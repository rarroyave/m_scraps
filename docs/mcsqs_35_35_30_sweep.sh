#!/bin/bash
# 35:35:30 ternary on FCC: compare scraps vs mcsqs on two cells.
# Case A: 2x5x5 conventional = 200 atoms (non-cubic, fully feasible).
# Case B: 5x5x5 conventional = 500 atoms (cubic, shell-1 infeasible).
# 5 trials each, scraps wall + mcsqs wall + max|alpha| from verify_sro.
set -e

ROOT=/tmp/mcsqs_35_35_30
REPO=$(cd "$(dirname "$0")/.." && pwd)
SCRAPS_BIN=$REPO/SCRAPs/scraps
VERIFY=$REPO/tools/verify_sro.py
MPIRUN=/opt/homebrew/bin/mpirun
TRIALS=${TRIALS:-5}

mkdir -p "$ROOT"
RESULTS_CSV="$ROOT/results.csv"
echo "case,n,tool,trial,wall_s,fitness,max_alpha" > "$RESULTS_CSV"

max_alpha_2shell() {
    local poscar="$1" species="$2"
    python3 $VERIFY "$poscar" --species "$species" --shells 2 --json 2>/dev/null \
        | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); mx=0.0
    for sh in d['shells']:
        for r in sh['alpha_matrix']:
            for v in r:
                if v is not None and v==v: mx=max(mx,abs(v))
    print(f'{mx:.4f}')
except Exception:
    print('nan')"
}

run_scraps_trial() {
    local case_name="$1" src_test_dir="$2" trial="$3" natom="$4"
    local dir="$ROOT/${case_name}_scraps/trial_${trial}"; mkdir -p "$dir"; cd "$dir"
    cp "$SCRAPS_BIN" ./scraps
    cp "$src_test_dir/POSCAR" "$src_test_dir/INPUT" .
    local t0=$(date +%s)
    "$MPIRUN" -np 4 ./scraps > scraps.log 2>&1
    local t1=$(date +%s)
    local wall=$((t1-t0))
    local fit=$(grep "FINAL fitness" scraps.log | sed -E 's/.*= //' | head -1)
    local a=$(max_alpha_2shell SCRAPS.vasp "A,B,C")
    echo "${case_name},${natom},scraps,${trial},${wall},${fit:-nan},${a}" >> "$RESULTS_CSV"
    echo "  scraps trial ${trial}: wall=${wall}s fit=${fit:-?} max|a|=${a}"
}

# FCC primitive rndstr with 35:35:30 composition
fcc_rndstr() {
    cat <<EOF
3.56 3.56 3.56 90 90 90
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5
0.0 0.0 0.0 A=0.35,B=0.35,C=0.30
EOF
}

# sqscell.out for FCC conventional k1xk2xk3 in primitive units.
# det = 4 k1 k2 k3.
fcc_sqscell() {
    local k1="$1" k2="$2" k3="$3"
    cat <<EOF
1

$k1 $k1 -$k1
$k2 -$k2 $k2
-$k3 $k3 $k3
EOF
}

run_mcsqs_trial() {
    local case_name="$1" k1="$2" k2="$3" k3="$4" budget="$5" trial="$6" natom="$7"
    local dir="$ROOT/${case_name}_mcsqs/trial_${trial}"; mkdir -p "$dir"; cd "$dir"
    fcc_rndstr > rndstr.in
    fcc_sqscell "$k1" "$k2" "$k3" > sqscell.out
    rm -f bestsqs.out bestcorr.out stopsqs mcsqs.log clusters.out
    mcsqs -2=3.6 > clus.log 2>&1
    local t0=$(date +%s)
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
    local wall=$((t1-t0))
    local a="nan"
    if [ -s bestsqs.out ]; then
        str2cif < bestsqs.out > bestsqs.cif 2>/dev/null
        python3 -c "from pymatgen.core import Structure; Structure.from_file('bestsqs.cif').to(filename='bestsqs.poscar', fmt='poscar')" 2>/dev/null
        a=$(max_alpha_2shell bestsqs.poscar "A,B,C")
    fi
    echo "${case_name},${natom},mcsqs,${trial},${wall},nan,${a}" >> "$RESULTS_CSV"
    echo "  mcsqs  trial ${trial}: wall=${wall}s max|a|=${a}"
}

run_case() {
    local case_name="$1" k1="$2" k2="$3" k3="$4" natom="$5" budget="$6"
    local src_test="$REPO/tests/${case_name}"

    echo "=========================================================="
    echo "Case: $case_name  (Cell_DIM ${k1}x${k2}x${k3} = $natom atoms, $TRIALS trials)"
    echo "=========================================================="
    for trial in $(seq 1 $TRIALS); do
        run_scraps_trial "$case_name" "$src_test" "$trial" "$natom"
        run_mcsqs_trial  "$case_name" "$k1" "$k2" "$k3" "$budget" "$trial" "$natom"
    done
}

# (case_dir, k1, k2, k3, natoms, mcsqs_budget_s)
run_case stoich_fcc_35_35_30_n200        2 5 5  200  120
run_case stoich_fcc_35_35_30_n500_cubic  5 5 5  500  240

echo ""
echo "=========================================================="
echo "Aggregated results: 35:35:30 ternary on FCC"
echo "=========================================================="
python3 <<EOF
import csv, statistics as st
rows = list(csv.DictReader(open("$RESULTS_CSV")))
cases = []
for r in rows:
    if r['case'] not in cases: cases.append(r['case'])

def agg(vals):
    vs = [float(v) for v in vals if v not in ('nan','')]
    if not vs: return None,None,None
    if len(vs) == 1: return vs[0], 0.0, vs[0]
    return st.mean(vs), st.stdev(vs), min(vs)

print("| case | n | tool | trials | wall_s mean±sd | max|α| mean±sd | best max|α| | scraps hit-0 |")
print("|---|---|---|---|---|---|---|---|")
for case in cases:
    n = next(r['n'] for r in rows if r['case']==case)
    for tool in ['scraps','mcsqs']:
        trs = [r for r in rows if r['case']==case and r['tool']==tool]
        if not trs: continue
        m_w, s_w, _ = agg(r['wall_s'] for r in trs)
        m_a, s_a, best_a = agg(r['max_alpha'] for r in trs)
        hit0 = ""
        if tool == 'scraps':
            zero_trials = sum(1 for r in trs if r['fitness'] not in ('nan','') and float(r['fitness']) == 0.0)
            hit0 = f"{zero_trials}/{len(trs)}"
        print(f"| {case} | {n} | {tool} | {len(trs)} | "
              f"{m_w:.0f}±{s_w:.0f} | "
              f"{m_a:.4f}±{s_a:.4f} | "
              f"{best_a:.4f} | {hit0} |")
EOF
