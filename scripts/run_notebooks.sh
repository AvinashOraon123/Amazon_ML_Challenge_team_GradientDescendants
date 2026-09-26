#!/usr/bin/env bash
# Execute the pipeline notebooks in order, in place (outputs are saved into the .ipynb).
#
#   scripts/run_notebooks.sh            # all notebooks 00..05
#   scripts/run_notebooks.sh 02 03 04   # only these
#
# Run inside tmux on EC2 so it survives SSH disconnects:
#   tmux new -d -s run "bash ~/business_entity_resolution/scripts/run_notebooks.sh > ~/run.log 2>&1"
#   tail -f ~/run.log
set -euo pipefail
cd "$(dirname "$0")/../notebooks"
[ -f ~/ber-venv/bin/activate ] && source ~/ber-venv/bin/activate
# BER_* settings (bucket, workers, sample mode) - non-interactive shells skip ~/.bashrc
# start from a clean slate: a long-lived tmux server can carry stale BER_* values
unset BER_SAMPLE_FRAC BER_N_JOBS BER_TRAIN_PCT BER_VALID_PCT
[ -f ~/ber.env ] && set -a && source ~/ber.env && set +a
python -c "import sys; sys.path.insert(0, \"../src\"); from ber import config as C; print(C.describe())"

sel=("$@")
[ ${#sel[@]} -eq 0 ] && sel=(00 01 02 03 04 05)

for p in "${sel[@]}"; do
  nbf=$(ls ${p}_*.ipynb | head -1)
  echo "=== $(date -u +%FT%TZ) START $nbf"
  start=$(date +%s)
  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$nbf"
  echo "=== $(date -u +%FT%TZ) DONE  $nbf in $(( $(date +%s) - start ))s"
done
echo "=== ALL DONE"
