#!/usr/bin/env bash
# One-shot status dump of the pipeline run on this instance (read by scripts/dashboard.ps1).
W=${BER_WORK_DIR:-$HOME/ber_work}
echo "now=$(date -u +%s)"
grep -E "=== .* (START|DONE)" ~/run.log 2>/dev/null | sed 's/^=== /log=/'
grep -qE "ALL DONE" ~/run.log 2>/dev/null && echo "alldone=1"
grep -E "CellExecutionError|Killed|MemoryError" ~/run.log 2>/dev/null | tail -1 | sed 's/^/error=/'
for f in "$W"/checkpoints/candidates/*/*.tmp "$W"/checkpoints/candidates/*/*.parquet; do
  [ -f "$f" ] && echo "cand=$(basename "$(dirname "$f")")/$(basename "$f"):$(stat -c %s "$f")"
done
[ -f "$W/progress.json" ] && echo "progress=$(cat "$W/progress.json")"
free -m | awk '/Mem/{print "mem="$3"/"$2}'
awk '{print "load="$1}' /proc/loadavg
df -BG / | awk 'NR==2{print "disk="$3"/"$2}'
tmux has-session -t run 2>/dev/null && echo "tmux=running" || echo "tmux=stopped"
