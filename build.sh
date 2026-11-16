#!/bin/sh
# build.sh [path to AMD's amd_fidelityfx_upscaler_vk.dll] [options]: see tools/build/linux.sh
exec "$(dirname "$0")/tools/build/linux.sh" "$@"
