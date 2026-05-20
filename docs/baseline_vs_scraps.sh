#!/bin/bash
set -e
SCRAPS=$HOME/codes/scraps-perpair/SCRAPs/scraps
VERIFY=$HOME/codes/scraps-perpair/tools/verify_sro.py
SEEDS=(42 1234 7777)

run_one() {
    local label="$1" src_dir="$2" species="$3" extra="$4" seed="$5"
    local dir="/tmp/baseline_test/${label}_seed${seed}"
    rm -rf "$dir"; mkdir -p "$dir"; cd "$dir"
    cp "$src_dir/POSCAR" .
    cp "$src_dir/INPUT" INPUT
    echo "RNG_SEED $seed" >> INPUT
    [ -n "$extra" ] && echo "$extra" >> INPUT
    echo "CONVERGENCE_LOG conv.csv" >> INPUT
    cp "$SCRAPS" ./scraps
    t0=$(date +%s)
    /opt/homebrew/bin/mpirun -np 4 ./scraps > scraps.log 2>&1
    t1=$(date +%s)
    fit=$(grep "FINAL fitness" scraps.log | sed -E 's/.*= //' | head -1)
    init=$(grep "INITIAL FITNESS" scraps.log | head -1 | sed -E 's/.*min=([0-9.e+-]+).*/\1/')
    a=$(python3 $VERIFY SCRAPS.vasp --species "$species" --shells 2 --json 2>/dev/null \
        | python3 -c "import json,sys
d=json.load(sys.stdin); mx=0.0
for sh in d['shells']:
 for r in sh['alpha_matrix']:
  for v in r:
   if v is not None and v==v: mx=max(mx,abs(v))
print(f'{mx:.4f}')")
    echo "  $label seed=$seed: init_min=$init  final=$fit  max|a|=$a  wall=$((t1-t0))s"
}

for seed in "${SEEDS[@]}"; do
    echo "=================================================================="
    echo "Case A: 25:50:25 FCC n=108 (SQS target, infeasible -- floor ~0.71)"
    echo "  Seed=$seed"
    echo "=================================================================="
    run_one A_random  $HOME/codes/scraps-perpair/tests/stoich_fcc_25_50_25_n108 "A,B,C" "RANDOM_ONLY 1" $seed
    run_one A_scraps  $HOME/codes/scraps-perpair/tests/stoich_fcc_25_50_25_n108 "A,B,C" "" $seed

    echo ""
    echo "=================================================================="
    echo "Case B: B2 target on BCC n=128 (specific ordered, alpha_AB=-1 NN)"
    echo "  Seed=$seed"
    echo "=================================================================="
    run_one B_random  $HOME/codes/scraps-perpair/tests/b2_bcc_binary "A,B" "RANDOM_ONLY 1" $seed
    run_one B_scraps  $HOME/codes/scraps-perpair/tests/b2_bcc_binary "A,B" "" $seed
    echo ""
done
