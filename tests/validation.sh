#!/bin/sh
# validation.sh [W H RW RH frames] [VAR=value ...]: the port under VK_LAYER_KHRONOS_validation, with synchronisation
# validation, on the moving camera with a reset. Needs the Khronos validation layer (vulkan-validation-layers).
# Fails if the layer reports anything but the known below. Extra harness settings (TRACE_*) may follow the sizes.
#   FSR4UP_BUILD   the build directory (default: <repo>/build)
#   FSR4UP_RE      where the run goes (default: <repo>/re), as validation-<name>/ (VALIDATION_NAME, default: default)
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
build=${FSR4UP_BUILD:-$root/build}; out=${FSR4UP_RE:-$root/re}/validation-${VALIDATION_NAME:-default}
W=${1:-1920} H=${2:-1080} RW=${3:-1280} RH=${4:-720} FRAMES=${5:-4}
[ $# -ge 5 ] && shift 5 || shift $#
rm -rf "$out"; mkdir -p "$out"
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_KHRONOS_VALIDATION_VALIDATE_CORE=true
export VK_KHRONOS_VALIDATION_VALIDATE_SYNC=true
export VK_KHRONOS_VALIDATION_SYNCVAL_FULL_VALIDATION=true
export VK_KHRONOS_VALIDATION_SYNCVAL_SHADER_ACCESSES_HEURISTIC=true
export VK_KHRONOS_VALIDATION_VALIDATE_BEST_PRACTICES=false
export VK_KHRONOS_VALIDATION_REPORT_FLAGS=error,warn
export VK_KHRONOS_VALIDATION_DEBUG_ACTION=VK_DBG_LAYER_ACTION_LOG_MSG
export VK_KHRONOS_VALIDATION_LOG_FILENAME="$out/validation.txt"
export VK_KHRONOS_VALIDATION_DUPLICATE_MESSAGE_LIMIT=3
env TRACE_CAMERA=${TRACE_CAMERA-0.5} TRACE_RESET_AT=${TRACE_RESET_AT-2} TRACE_DUMP_ALL= FSR4UP_LOG="$out/fsr4up.log" "$@" \
    "$build/trace" "$build/amd_fidelityfx_upscaler_vk.so" "$out" "$W" "$H" "$RW" "$RH" "$FRAMES" 99 \
    > "$out/stdout.txt" 2>&1 || { echo "the harness failed: $out/stdout.txt"; exit 1; }
# (with VK_INSTANCE_LAYERS naming a layer that is missing, instance creation fails: a clean run did load it)
#
# Known, and AMD's (spirv-val reports the same on AMD's modules themselves; AMD's DLL binds the same images):
#   08737, 08740, 08742 at vkCreateShaderModule: the network declares SPV_NV_cooperative_matrix2 and
#     CooperativeMatrixConversionsNV, which RADV compiles but does not advertise;
#   06924 at vkCreateShaderModule: most prepass variants store an image object to a function variable;
#   08795 at vkCmdDispatch: the prepass and postpass write 3-component texels to RGBA16F storage images
#     (rw_reprojected_color, rw_mlsr_output_color, rw_history_color, rw_rcas_output): alpha is the driver's to choose;
#     RADV writes the same as under AMD's DLL (the suite compares every byte).
#   Undefined-Value-StorageImage-FormatMismatch-ImageView (a warning) at vkCmdDispatch: the postpass declares
#     rw_recurrent_0 as Rgba32f, while AMD's DLL makes that image R8G8B8A8_UNORM (traced), as the port does.
# Fixing them would mean rewriting AMD's shaders. Every other message (any SYNC-HAZARD included) fails the run.
known='VUID-VkShaderModuleCreateInfo-pCode-087(37|40|42)|VUID-StandaloneSpirv-OpTypeImage-06924|VUID-vkCmdDispatch-OpImageWrite-08795|Undefined-Value-StorageImage-FormatMismatch-ImageView'
other=$(grep -E '^Validation (Error|Warning)' "$out/validation.txt" 2>/dev/null | grep -vE "$known" || true)
if [ -n "$other" ]; then
    echo "validation reported problems: $out/validation.txt"
    echo "$other" | grep -oE 'VUID-[A-Za-z0-9_-]+|SYNC-[A-Z0-9-]+' | sort | uniq -c | sort -rn | head -20
    exit 1
fi
n=$(grep -c 'upscale -> 0' "$out/stdout.txt" || true)
[ "$n" = "$FRAMES" ] || { echo "only $n of $FRAMES frames upscaled: $out/stdout.txt"; exit 1; }
if grep -A1 'FormatMismatch' "$out/validation.txt" | grep -q 'variable "' && grep -A1 'FormatMismatch' "$out/validation.txt" | grep -o 'variable "[a-z_0-9]*"' | grep -qv rw_recurrent_0; then
    echo "a format mismatch on another image than the recurrent state: $out/validation.txt"; exit 1
fi
echo "validation clean ($n frames; AMD's own shader issues reported as known)"
