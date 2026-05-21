#!/bin/bash
# Benchmark SA vs cuckoo MC kernels on a handful of test cases.
# Both runs use the same RNG_SEED so the initial random pool is
# identical; only the inner acceptance rule differs.
set -e
SCRAPS=$HOME/codes/scraps-perpair/SCRAPs/scraps
MPIRUN=/opt/homebrew/bin/mpirun

run_case() {
    local label="$1" src_dir="$2" species="$3" extra="$4"
    local dir="/tmp/sa_vs_cuckoo/${label}"
    rm -rf "$dir"; mkdir -p "$dir"; cd "$dir"
    cp "$src_dir/POSCAR" "$src_dir/INPUT" .
    echo "RNG_SEED 42" >> INPUT
    echo "EARLY_STOP_PATIENCE 5" >> INPUT
    echo "$extra" >> INPUT
    cp "$SCRAPS" ./scraps
    local t0=$(date +%s%N)
    "$MPIRUN" -np 4 ./scraps > scraps.log 2>&1
    local t1=$(date +%s%N)
    local wall_ms=$(( (t1-t0)/1000000 ))
    local fit=$(grep "FINAL fitness" scraps.log | sed -E 's/.*= //' | head -1)
    echo "  $label: wall=${wall_ms}ms fitness=$fit"
}

# 4 test cases × {cuckoo, sa}
for case_info in \
    "stoich_fcc_25_50_25_n108 plateau A,B,C" \
    "b2_bcc_binary           b2 A,B" \
    "l12_fcc_a3b             l12 A,B" \
    "stoich_fcc_50_50        sqs A,B"; do
    src=$(echo $case_info | awk '{print $1}')
    tag=$(echo $case_info | awk '{print $2}')
    species=$(echo $case_info | awk '{print $3}')
    echo "==== $src ===="
    run_case "${tag}_cuckoo" $HOME/codes/scraps-perpair/tests/$src "$species" ""
    run_case "${tag}_sa"     $HOME/codes/scraps-perpair/tests/$src "$species" "MC_KERNEL sa
SA_T_INITIAL 2.0
SA_T_FINAL 1e-3"
done
