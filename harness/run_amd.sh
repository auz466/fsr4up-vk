#!/bin/sh
# run_amd.sh <run name> [W H RW RH frames dumpFrom]: AMD's upscaler DLL under Wine, traced into re/<run name>
#
# Needs build/wine/trace.exe and AMD's DLL beside it (cmake with mingw-w64 and FSR4UP_AMD_DLL). Settings, from the
# environment or from <repo>/local.env (not committed):
#   FSR4UP_WINEPREFIX   the Wine prefix to run in (default: $WINEPREFIX, else Wine's own default)
#   FSR4UP_BUILD        the build directory (default: <repo>/build)
#   FSR4UP_RE           where runs go (default: <repo>/re)
here=$(cd "$(dirname "$0")" && pwd); root=$(dirname "$here")
[ -f "$root/local.env" ] && . "$root/local.env"
build=${FSR4UP_BUILD:-$root/build}; out=${FSR4UP_RE:-$root/re}
[ -n "$FSR4UP_WINEPREFIX" ] && export WINEPREFIX="$FSR4UP_WINEPREFIX"
for f in wine/trace.exe wine/amd_fidelityfx_upscaler_vk.dll; do
    [ -e "$build/$f" ] || { echo "run_amd.sh: $build/$f is missing (build with mingw-w64 and -DFSR4UP_AMD_DLL)" >&2; exit 1; }
done
name=$1; shift
rm -rf "${out:?}/$name"; mkdir -p "$out/$name"
cd "$build/wine" && WINEDEBUG=-all TRACE_DUMP_ALL=${TRACE_DUMP_ALL-1} \
  timeout 900 wine trace.exe amd_fidelityfx_upscaler_vk.dll "Z:$out/$name" "$@" 2>&1 | grep -v "radv is not"
