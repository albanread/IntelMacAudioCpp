#!/bin/zsh
# Per-op correctness sweep with a watchdog.
#
# Do NOT run the full test-backend-ops suite on a multi-GPU Intel Mac: a compute hang on the
# headless GPU stalls the display GPU through the shared IOAcceleratorFamily2 driver, and the
# userspace watchdog kills WindowServer after 40s. Each op here is killed at 30s instead.
#
# usage: GGML_METAL_DEVICE=Vega ./safe-sweep.sh [op ...]

cd "$(dirname "$0")"
BIN=/Volumes/S/audio.cpp/build/ggml-tests/bin/test-backend-ops
LOG=${LOG_DIR:-./sweep-logs}
mkdir -p "$LOG"

OPS=("$@")
if [ ${#OPS[@]} -eq 0 ]; then
  OPS=(GET_ROWS RMS_NORM NORM SOFT_MAX ROPE ADD MUL SCALE CPY CONT GLU DIV MUL_MAT)
fi

strip() { perl -pe 's/\e\[[0-9;]*m//g'; }

echo "=== per-op sweep $(date) ==="
for op in "${OPS[@]}"; do
  out=$LOG/op_$op.log
  ( $BIN -o $op -b MTL0 > "$out" 2>&1 ) &
  pid=$!
  ( sleep 30; kill -9 $pid 2>/dev/null; echo "[[WATCHDOG-KILLED]]" >> "$out" ) &
  w=$!
  wait $pid 2>/dev/null
  kill $w 2>/dev/null
  ok=$(strip < "$out" | grep -c ': OK')
  fail=$(strip < "$out" | grep -c ': FAIL')
  hung=$(grep -c 'WATCHDOG-KILLED' "$out")
  printf "%-12s OK=%-5s FAIL=%-5s %s\n" "$op" "$ok" "$fail" \
    "$([ "$hung" -gt 0 ] && echo '*** HUNG — GPU did not finish ***')"
done
echo "=== done — logs in $LOG ==="
