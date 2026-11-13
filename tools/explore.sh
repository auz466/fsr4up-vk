#!/bin/bash
# explore.sh <name> W H RW RH [VAR=value ...]: AMD's DLL for 2 frames, no dumps; prints frame 1's dispatches as carved#@groups,
# then "made:" and every pipeline made, in order
cd "$(dirname "$0")/.." || exit 1
n=$1; W=$2; H=$3; RW=$4; RH=$5; shift 5
env "$@" TRACE_DUMP_ALL= ./harness/run_amd.sh "explore/$n" "$W" "$H" "$RW" "$RH" 2 99 >/dev/null 2>&1
p=$(python3 -I tools/picks.py "re/explore/$n" 1)
echo "$n $(echo "$p" | head -1 | sed 's/^dispatch: //') made: $(echo "$p" | tail -1 | sed 's/^made: //')"
