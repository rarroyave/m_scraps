#!/bin/bash
# Fixed-budget comparison: give both tools BUDGET seconds, see what they achieve.
set -e

ROOT=/tmp/mcsqs_vs_scraps
SCRAPS_BIN=~/codes/scraps-perpair/SCRAPs/scraps
VERIFY=~/codes/scraps-perpair/tools/verify_sro.py

# ---- Helper: read max |alpha| from a verify_sro JSON over given species ----
max_alpha() {
    local poscar="$1"; local species="$2"; local extra="$3"
    python3 $VERIFY "$poscar" --species "$species" $extra --shells 1 --json 2>/dev/null | \
        python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); mx=0.0
    for sh in d['shells']:
        for row in sh['alpha_matrix']:
            for v in row:
                if v is not None and (v==v):  # not NaN
                    mx=max(mx,abs(v))
    print(f'{mx:.4f}')
except Exception as e:
    print('err')"
}

run_scraps() {
    local dir="$1" budget="$2"
    cd "$dir"
    t0=$(date +%s)
    /opt/homebrew/bin/mpirun -np 4 ./scraps > scraps.log 2>&1
    t1=$(date +%s)
    echo "$((t1-t0))"
}

run_mcsqs() {
    local dir="$1" budget="$2" natom="$3"
    cd "$dir"
    rm -f bestsqs.out bestcorr.out stopsqs mcsqs.log
    t0=$(date +%s)
    mcsqs -n=$natom > mcsqs.log 2>&1 &
    pid=$!
    elapsed=0
    while kill -0 $pid 2>/dev/null && [ $elapsed -lt $budget ]; do
        sleep 1; elapsed=$((elapsed+1))
    done
    touch stopsqs
    sleep 3  # mcsqs polls for stopsqs periodically
    if kill -0 $pid 2>/dev/null; then kill $pid 2>/dev/null; fi
    wait $pid 2>/dev/null
    t1=$(date +%s)
    str2cif < bestsqs.out > bestsqs.cif 2>/dev/null
    python3 -c "from pymatgen.core import Structure; Structure.from_file('bestsqs.cif').to(filename='bestsqs.poscar', fmt='poscar')" 2>/dev/null
    echo "$((t1-t0))"
}

# ============================================================
# Test 1: NiCoCr equiatomic, 108 atoms, FCC
# ============================================================
echo "=========================================================="
echo "Test 1: NiCoCr 108-atom equiatomic SQS"
echo "  SCRAPS: nicocr_random INPUT (default budget)"
echo "  mcsqs: 30 second budget"
echo "=========================================================="

# SCRAPS
SCDIR=$ROOT/test1_nicocr_scraps; mkdir -p "$SCDIR"; cd "$SCDIR"
cp $SCRAPS_BIN ./scraps
cp ~/codes/scraps-perpair/tests/nicocr_random/POSCAR .
cp ~/codes/scraps-perpair/tests/nicocr_random/INPUT .
sc_wall=$(run_scraps "$SCDIR" 30)
sc_alpha=$(max_alpha "$SCDIR/SCRAPS.vasp" "Ni,Co,Cr" "")

# mcsqs
MCDIR=$ROOT/test1_nicocr_mcsqs; mkdir -p "$MCDIR"; cd "$MCDIR"
cat > rndstr.in <<'EOF'
3.56 3.56 3.56 90 90 90
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5
0.0 0.0 0.0 Ni=0.3333333,Co=0.3333333,Cr=0.3333334
EOF
mcsqs -2=5.34 > clus.log 2>&1
mc_wall=$(run_mcsqs "$MCDIR" 30 108)
mc_alpha=$(max_alpha "$MCDIR/bestsqs.poscar" "Ni,Co,Cr" "")

echo ""
echo "  | tool   | atoms | parallelism | wall | max|α| shell 0 |"
echo "  | ------ | ----- | ----------- | ---- | -------------- |"
printf "  | scraps | 108   | 4 MPI       | %3ds | %s         |\n" "$sc_wall" "$sc_alpha"
printf "  | mcsqs  | 108   | 1 (serial)  | %3ds | %s         |\n" "$mc_wall" "$mc_alpha"
echo ""

# ============================================================
# Test 2: HEB (Hf,Zr,Ti,Ta,Nb)B2 150 atoms (AlB2 5x5x2 supercell)
# ============================================================
echo "=========================================================="
echo "Test 2: HEB (Hf,Zr,Ti,Ta,Nb)B2 150-atom equiatomic SQS"
echo "  SCRAPS: heb_5metal_random INPUT (with SPECTATOR_BASIS_SPECIES)"
echo "  mcsqs: native multi-sublattice via rndstr.in"
echo "=========================================================="

# SCRAPS
SCDIR=$ROOT/test2_heb_scraps; mkdir -p "$SCDIR"; cd "$SCDIR"
cp $SCRAPS_BIN ./scraps
cp ~/codes/scraps-perpair/tests/heb_5metal_random/POSCAR .
cp ~/codes/scraps-perpair/tests/heb_5metal_random/INPUT .
sc_wall=$(run_scraps "$SCDIR" 60)
sc_alpha=$(max_alpha "$SCDIR/SCRAPS.vasp" "Hf,Zr,Ti,Ta,Nb,B" "--spectator-species B")

# mcsqs - HEB AlB2 with mixed metals on basis 0, pure B on basis 1, 2
MCDIR=$ROOT/test2_heb_mcsqs; mkdir -p "$MCDIR"; cd "$MCDIR"
cat > rndstr.in <<'EOF'
1.0 1.0 1.1 90 90 120
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
0.0       0.0       0.0       Hf=0.2,Zr=0.2,Ti=0.2,Ta=0.2,Nb=0.2
0.333333  0.666667  0.5       B=1.0
0.666667  0.333333  0.5       B=1.0
EOF
mcsqs -2=1.2 > clus.log 2>&1
mc_wall=$(run_mcsqs "$MCDIR" 60 150)
# verify_sro with HEB spectator handling
mc_alpha=$(max_alpha "$MCDIR/bestsqs.poscar" "Hf,Zr,Ti,Ta,Nb,B" "--spectator-species B")

echo ""
echo "  | tool   | atoms | parallelism | wall | max|α_MM| shell 0 |"
echo "  | ------ | ----- | ----------- | ---- | ---------------- |"
printf "  | scraps | 150   | 4 MPI       | %3ds | %s           |\n" "$sc_wall" "$sc_alpha"
printf "  | mcsqs  | 150   | 1 (serial)  | %3ds | %s           |\n" "$mc_wall" "$mc_alpha"
echo ""