#!/usr/bin/env bash
# Builds build/amd_fidelityfx_upscaler_vk.so from AMD's own DLL (FidelityFX SDK 2.4.0), which you supply or let this
# script download. Nothing taken from AMD's DLL is committed to this repository: the .so is made on your machine.
#
#   tools/build/linux.sh                       use the remembered DLL, else download it from AMD's SDK repository
#   tools/build/linux.sh /path/to/amd_fidelityfx_upscaler_vk.dll
#   tools/build/linux.sh --download            fetch the DLL (36 MB) from github.com/GPUOpen-LibrariesAndSDKs
#   tools/build/linux.sh --harness             also build the test harness (build/trace, build/wine/trace.exe)
#   tools/build/linux.sh --install DIR         copy the finished .so into DIR
set -euo pipefail
cd "$(dirname "$0")/../.."

SHA256=4a7e895af6e0b615d40a84d59eb8c2b1054ad91cdd0c739567940270d51bdab7
URL=https://raw.githubusercontent.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/v2.4.0/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll
CACHED=build/dll/amd_fidelityfx_upscaler_vk.dll

dll= download=0 harness=OFF install=
while [ $# -gt 0 ]; do
    case $1 in
        --download) download=1 ;;
        --harness)  harness=ON ;;
        --install)  install=${2:?--install needs a directory}; shift ;;
        -h|--help)  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*)         echo "unknown option $1" >&2; exit 2 ;;
        *)          dll=$1 ;;
    esac
    shift
done

die() { echo "error: $*" >&2; exit 1; }

# ---- tools
missing=()
command -v cmake >/dev/null || missing+=(cmake)
command -v python3 >/dev/null || missing+=(python3)
command -v ninja >/dev/null || command -v make >/dev/null || missing+=("ninja or make")
command -v clang++ >/dev/null || command -v g++ >/dev/null || missing+=("a C++20 compiler (clang++ or g++)")
command -v sha256sum >/dev/null || missing+=(sha256sum)
[ "$download" = 1 ] && ! command -v curl >/dev/null && missing+=(curl)
[ -f /usr/include/vulkan/vulkan_core.h ] || [ -f /usr/local/include/vulkan/vulkan_core.h ] || [ -n "${VULKAN_SDK:-}" ] \
    || missing+=("the Vulkan headers (vulkan-headers / libvulkan-dev)")
if [ ${#missing[@]} -gt 0 ]; then
    printf 'missing: %s\n' "${missing[@]}" >&2
    die "install the above (Arch: cmake ninja python vulkan-headers; Debian/Ubuntu: cmake ninja-build python3 libvulkan-dev g++; Fedora: cmake ninja-build python3 vulkan-headers gcc-c++)"
fi

# ---- AMD's DLL
if [ -z "$dll" ]; then
    if [ "$download" = 0 ] && [ -f build/CMakeCache.txt ] && grep -q '^FSR4UP_AMD_DLL:FILEPATH=.\+' build/CMakeCache.txt \
        && [ -f "$(sed -n 's/^FSR4UP_AMD_DLL:FILEPATH=//p' build/CMakeCache.txt)" ]; then
        echo "using the DLL remembered from the last build"
    else
        [ -f "$CACHED" ] && [ "$(sha256sum "$CACHED" | cut -d' ' -f1)" = "$SHA256" ] && [ "$download" = 0 ] || {
            command -v curl >/dev/null || die "curl is needed to download AMD's DLL; or pass its path"
            echo "downloading amd_fidelityfx_upscaler_vk.dll (36 MB) from AMD's FidelityFX SDK v2.4.0 ..."
            mkdir -p build/dll
            curl -fL --retry 3 -o "$CACHED.part" "$URL"
            mv "$CACHED.part" "$CACHED"
        }
        dll=$CACHED
    fi
fi
if [ -n "$dll" ]; then
    [ -f "$dll" ] || die "$dll does not exist"
    got=$(sha256sum "$dll" | cut -d' ' -f1)
    [ "$got" = "$SHA256" ] || die "$dll is not the DLL this was made for.
  need: Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll of FidelityFX SDK v2.4.0 (SHA-256 $SHA256)
  got:  $got"
    dll=$(realpath "$dll")
fi

# ---- build
gen=(-G "$(command -v ninja >/dev/null && echo Ninja || echo 'Unix Makefiles')")
[ -f build/CMakeCache.txt ] && gen=()
if [ ! -f build/CMakeCache.txt ] && [ -z "${CXX:-}" ] && command -v clang++ >/dev/null; then export CXX=clang++; fi
args=("-DFSR4UP_BUILD_HARNESS=$harness" "-DFSR4UP_BUILD_HARNESS_EXE=$harness")
[ -n "$dll" ] && args+=("-DFSR4UP_AMD_DLL=$dll")
cmake -S . -B build "${gen[@]}" "${args[@]}"
cmake --build build -j"$(nproc)"

so=build/amd_fidelityfx_upscaler_vk.so
[ -f "$so" ] || die "the build finished but $so is missing"
if [ -n "$install" ]; then mkdir -p "$install"; cp "$so" "$install/"; so="$install/amd_fidelityfx_upscaler_vk.so"; fi
echo
echo "built: $(realpath "$so")"
echo "It embeds AMD's shaders and data from their DLL: keep THIRD_PARTY_NOTICES.md with it if you pass it on."
