#!/bin/bash
set -euo pipefail
cd "$HOME/runs/scraps-perpair/random-selection-20261019"
python_bin="$HOME/src/external/MaterialsFramework/.venv/bin/python"
"$python_bin" -u check_random.py
"$python_bin" -u prepare_distribution.py
"$python_bin" -u "$HOME/runs/sqs-to-property-demo/max-ensemble-20261019/ensemble_energy.py" --scraps-repo "$HOME/runs/scraps-perpair/ensemble-20261019/source" evaluate --ensemble random32 --output random32-energies --model "$HOME/.cache/mace/20231210mace128L0_energy_epoch249model" --threads 4
