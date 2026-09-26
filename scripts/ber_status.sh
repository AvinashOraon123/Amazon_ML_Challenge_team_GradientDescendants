#!/usr/bin/env bash
# One-shot status dump of the pipeline run on this instance (read by scripts/dashboard.ps1).
W=${BER_WORK_DIR:-$HOME/ber_work}
O=$HOME/business_entity_resolution/output
echo "now=$(date -u +%s)"
grep -E "=== .* (START|DONE|RESUME)" ~/run.log 2>/dev/null | sed 's/^=== /log=/'
grep -qE "ALL DONE" ~/run.log 2>/dev/null && echo "alldone=1"
grep -E "CellExecutionError|Killed|MemoryError" ~/run.log 2>/dev/null | tail -1 | sed 's/^/error=/'
# most recent kernel OOM kill (dmesg timestamps are seconds since boot)
oom=$(sudo -n dmesg 2>/dev/null | grep -i "killed process" | tail -1)
if [ -n "$oom" ]; then
  up=$(cut -d. -f1 /proc/uptime); t=$(echo "$oom" | sed -E 's/^\[ *([0-9]+)\..*/\1/')
  echo "oom_ago=$((up - t))"
fi
for f in "$W"/checkpoints/candidates/*/*.tmp "$W"/checkpoints/candidates/*/*.parquet "$W"/checkpoints/pruned/*/*.parquet "$W"/checkpoints/pruned/*/*.tmp; do
  [ -f "$f" ] && echo "cand=$(basename "$(dirname "$(dirname "$f")")")/$(basename "$(dirname "$f")")/$(basename "$f"):$(stat -c %s "$f")"
done
[ -d "$W/checkpoints/scores/test" ] && echo "scoreparts=$(ls "$W/checkpoints/scores/test" | wc -l)"
[ -f "$W/progress.json" ] && echo "progress=$(cat "$W/progress.json")"
for f in "$O"/matching_results.tsv "$O"/candidate_pairs.tsv; do
  [ -f "$f" ] && echo "out=$(basename "$f"):$(stat -c %s "$f"):$(stat -c %Y "$f")"
done
free -m | awk '/Mem/{print "mem="$3"/"$2}'
awk '{print "load="$1}' /proc/loadavg
df -BG / | awk 'NR==2{print "disk="$3"/"$2}'
tmux has-session -t run 2>/dev/null && echo "tmux=running" || echo "tmux=stopped"
pgrep -f "[n]bconvert" >/dev/null && echo "nbconvert=alive" || echo "nbconvert=none"
