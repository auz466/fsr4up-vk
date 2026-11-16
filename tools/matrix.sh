#!/bin/bash
# matrix.sh W H RW RH <out file>: every combination of the create flags that matter (2 4 8 32 256 512) and the colour
# dispatch flags (0 2 4 6), traced through AMD's DLL (MATRIX_JOBS runs at a time, default 2); one line each: "cf df picks
# of frame 1". The environment goes through to the harness (for the INT8 networks: TRACE_AMD_FAMILY, TRACE_AMD_REV,
# TRACE_HIDE_EXT); MATRIX_TAG names the run (its runs go to re/matrix-<tag>-W-H/).
cd "$(dirname "$0")/.." || exit 1
W=$1 H=$2 RW=$3 RH=$4 out=$5
tag=${MATRIX_TAG:+$MATRIX_TAG-}
dir=re/matrix-$tag$W-$H; mkdir -p $dir
python3 -c '
bits=[2,4,8,32,256,512]
for c in range(64):
    f=sum(b for i,b in enumerate(bits) if c>>i&1)
    for df in (0,2,4,6): print(f,df)' | xargs -P ${MATRIX_JOBS:-2} -L 1 bash -c '
n=m'"$tag$W"'_cf$0_df$1
for try in 1 2 3; do
  line=$(tools/explore.sh $n '"$W $H $RW $RH"' TRACE_CREATE_FLAGS=$0 TRACE_DISPATCH_FLAGS=$1)
  words=($line); [ ${#words[@]} -gt 1 ] && break
  grep -q "dispatch upscale -> [1-9]" re/explore/$n/trace.log 2>/dev/null && break
done
echo "$0 $1 ${line#* }" > '"$dir"'/$0_$1.txt; rm -rf re/explore/$n'
cat $dir/*.txt | sort -n -k1 -k2 > "$out"
