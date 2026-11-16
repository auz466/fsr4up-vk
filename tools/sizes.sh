#!/bin/bash
# sizes.sh <net> [VAR=value ...]: AMD's DLL at 20 display sizes, each at 1.5x and 3x (the standard and ultra performance
# presets), 2 frames, default flags; one line a run in re/int8/sizes-<net>.txt: "W H RW RH scratch size=N inits <upload
# offsets in the DLL> mem <memory answers> dispatch: carved@groups ..." (for tools/fit_groups.py and tools/make_model.py).
# The VAR=value (and the environment) go to the harness: for an INT8 network TRACE_AMD_FAMILY, TRACE_AMD_REV and
# TRACE_HIDE_EXT=VK_KHR_cooperative_matrix.
cd "$(dirname "$0")/.." || exit 1
net=$1; shift
mkdir -p re/int8; out=re/int8/sizes-$net.txt; : > $out
for s in 1920x1080 1001x777 1283x721 640x360 1366x768 1919x1079 1913x1033 1921x1080 1920x1081 2560x1440 3840x2160 3001x1999 2222x1777 1080x1920 3841x2160 3840x2161 5120x2880 4097x3001 7680x4320 1920x1088; do
  W=${s%x*} H=${s#*x}
  for d in 1.5 3; do
    RW=$(python3 -c "print(int($W/$d))") RH=$(python3 -c "print(int($H/$d))")
    n=sz_${net}_${W}x${H}_$d
    env "$@" TRACE_DUMP_ALL= harness/run_amd.sh explore/$n $W $H $RW $RH 2 99 >/dev/null 2>&1
    p=$(python3 -I tools/picks.py re/explore/$n 1 | head -1)
    sc=$(grep -m1 -oE "CreateBuffer buf[0-9]+ size=[0-9]+ usage=0x23" re/explore/$n/trace.log | grep -oE "size=[0-9]+")
    ini=$(python3 -I tools/inits.py re/explore/$n)
    mem=$(grep -oE "query memory v[12][^-]*" re/explore/$n/trace.log | tr '\n' ';')
    echo "$W $H $RW $RH scratch $sc inits $ini mem $mem $p" >> $out
    rm -rf re/explore/$n
  done
done
