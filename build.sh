#!/bin/sh
# build.sh [path to AMD's amd_fidelityfx_upscaler_vk.dll]: configures and builds with CMake into build/.
# The DLL (FSR SDK 2.4.0) is needed the first time, or to switch to another copy; it is remembered after that.
#   build/amd_fidelityfx_upscaler_vk.so, build/trace, and build/wine/trace.exe (with mingw-w64)
set -e
cd "$(dirname "$0")"
if [ -n "$1" ]; then set -- "-DFSR4UP_AMD_DLL=$(realpath "$1")"; fi
if [ ! -f build/CMakeCache.txt ] && [ -z "$CXX" ] && command -v clang++ >/dev/null; then export CXX=clang++; fi
cmake -S . -B build -G Ninja "$@"
cmake --build build -j"$(nproc)"
ls -la build/amd_fidelityfx_upscaler_vk.so build/trace
