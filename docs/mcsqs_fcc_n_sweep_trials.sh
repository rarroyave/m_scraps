#!/bin/bash
# FCC 25:50:25 ternary, cell-size sweep with multiple trials per case.
# For each (n, tool) pair: run TRIALS independent trials and aggregate
# wall time, fitness, and max|alpha|. Each trial uses a fresh directory
# so SCRAPS.vasp doesn't get overwritten across trials.
set -e

ROOT=/tmp/mcsqs_fcc_n_trials
REPO=$(cd "$(dirname "$0")/.." && pwd)
SCRAPS_BIN=$REPO/SCRAPs/scraps
VERIFY=$REPO/tools/verify_sro.py
MPIRUN=/opt/homebrew/bin/mpirun

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
    local case_name="$1" src_test_dir="$2" trial="$3"
    local dir="$ROOT/${case_name}_scraps/trial_${trial}"; mkdir -p "$dir"; cd "$dir"
    cp "$SCRAPS_BIN" ./scraps
    cp "$src_test_dir/POSCAR" "$src_test_dir/INPUT" .
    local t0=$(date +%s)
    "$MPIRUN" -np 4 ./scraps > scraps.log 2>&1
    local t1=$(date +%s)
    local wall=$((t1-t0))
    local fit=$(grep "FINAL fitness" scraps.log | sed -E 's/.*= //' | head -1)
    local a=$(max_alpha_2shell SCRAPS.vasp "A,B,C")
    echo "${case_name},${4},scraps,${trial},${wall},${fit:-nan},${a}" >> "$RESULTS_CSV"
    echo "  scraps trial ${trial}: wall=${wall}s fit=${fit:-?} max|a|=${a}"
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

# sqscell.out for FCC conventional kxkxk (in primitive units; det = 4k^3)
fcc_sqscell() {
    local k="$1"
    cat <<EOF
1

$k $k -$k
$k -$k $k
-$k $k $k
EOF
}

run_mcsqs_trial() {
    local case_name="$1" k="$2" budget="$3" trial="$4" natom="$5"
    local dir="$ROOT/${case_name}_mcsqs/trial_${trial}"; mkdir -p "$dir"; cd "$dir"
    fcc_rndstr > rndstr.in
    fcc_sqscell "$k" > sqscell.out
    rm -f bestsqs.out bestcorr.out stopsqs mcsqs.log clusters.out

    mcsqs -2=3.6 > clus.log 2>&1
    local t0=$(date +%s)
    # mcsqs has no seed flag and uses /dev/urandom — different trials
    # naturally start from different RNG seeds.
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

run_case_trials() {
    local case_name="$1" k="$2" natom="$3" budget="$4" trials="$5"
    local src_test="$REPO/tests/${case_name}"

    echo "=========================================================="
    echo "Case: $case_name  (n=$natom, $trials trials)"
    echo "=========================================================="
    for trial in $(seq 1 $trials); do
        run_scraps_trial "$case_name" "$src_test" "$trial" "$natom"
        run_mcsqs_trial  "$case_name" "$k" "$budget" "$trial" "$natom"
    done
}

# Sweep: (case_dir, k, natoms, mcsqs_budget_s, num_trials)
run_case_trials stoich_fcc_25_50_25_n32   2   32   30  5
run_case_trials stoich_fcc_25_50_25_n108  3  108   60  5
run_case_trials stoich_fcc_25_50_25_n256  4  256  120  5
run_case_trials stoich_fcc_25_50_25_n500  5  500  180  3

echo ""
echo "=========================================================="
echo "Aggregated results"
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

print("| n | tool | trials | wall_s mean±sd | max|α| mean±sd | best max|α| | scraps hit-0 |")
print("|---|---|---|---|---|---|---|")
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
        line = (f"| {n} | {tool} | {len(trs)} | "
                f"{m_w:.0f}±{s_w:.0f} | "
                f"{m_a:.4f}±{s_a:.4f} | "
                f"{best_a:.4f} | {hit0} |")
        print(line)
EOF
