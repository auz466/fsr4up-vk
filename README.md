# fsr4up-vk

A native Linux Vulkan implementation of **AMD FSR 4.1.2 upscaling** (the machine-learning one, FidelityFX SDK 2.4.0):
its FP8 network for RDNA 4 and its three INT8 networks for other GPUs. AMD ships 4.1.2 for Vulkan only as a Windows DLL (`amd_fidelityfx_upscaler_vk.dll`); this library answers the
same FSR API calls with AMD's own shaders and initializer data, which the build takes out of that DLL.

This is an independent implementation. It is not AMD's, and AMD does not support it. It reports itself as
`4.1.2 (fsr4up-vk)`.

It drops in where [fsr4-vk](https://github.com/Astyyyyy/fsr4-vk) (FSR 4.1.1, from SDK 2.3.0's DX12 DLL) is used: the
same file name, `amd_fidelityfx_upscaler_vk.so`, and the same entry points.

Heavily assisted by Claude.

## Status

- Output **bit-identical to AMD's DLL** (run under Wine on the same GPU) on the regression suite: `tests/against_amd.py`,
  every frame of every case, plus every API answer (see that file for the cases: sizes from 1001x777 to 5120x2880 and
  portrait, every quality mode, still and moving camera, resets, the depth modes, jitter cancellation, display-sized
  motion vectors, auto exposure, an exposure texture, pre-exposure, HDR, non-linear/sRGB/PQ colour, sharpening, the
  debug view, dynamic resolution, a smaller upscale size, input and output formats, the ultra performance preset).
- The INT8 networks too: AMD's VK DLL does carry them (an earlier version of this README said otherwise), and the suite
  runs AMD's DLL down each of them (see "Networks") and compares the same way, every byte of every frame and every API
  answer, on a representative set of the cases for each network (`int8_*` in `tests/against_amd.py`).
  `FSR4-i8` and `FSR4L-i8` are not deterministic at an odd display width or height: their prepass gives other results
  from run to run there, in AMD's DLL as in this library (with the GPU's memory zeroed too), so they are compared at
  even sizes only.
- Tested on an AMD Radeon RX 9070 XT (RDNA 4) with RADV (Mesa 26.2.6): FP8 and the three INT8 networks, all 197 cases
  of the suite. The INT8 cases also on the Ryzen 9800X3D's integrated Radeon (Raphael, RDNA 2; AMD's DLL made to run
  INT8 there the same way): bit-identical but for the three debug view cases, whose race (see the suite) spreads over
  more texels there than the suite allows (148 to 267 texels between two runs of AMD's DLL itself; the port's frames
  differ from AMD's by as much).
- Not done: AMD's FSR 3.1.6 and 2.3.4 providers (also in its DLL). `FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK`
  returns OK and records nothing, as AMD's 4.1.2 does.

## Networks

AMD's DLL holds four FSR 4.1.2 networks, all reported as `4.1.2` with the same version id; it lists one of them, the
first its device check accepts:

| network (AMD's name) | AMD's DLL offers it on | device check in AMD's DLL |
|---|---|---|
| FP8 (`FSR4`) | RDNA 4 | AMD vendor, not an integrated GPU, `VK_KHR_cooperative_matrix` and `VK_EXT_shader_float8` listed, `cooperativeMatrix` |
| INT8 `FSR4-i8` | RDNA 3 (Navi 3x), Strix Halo | family 0x91, revision 0x01..0xfe; family 0x96, revision 0xc0..0xcf; or, the family unknown, PCI device 0x7481 |
| INT8 `FSR4-i8-2` | RDNA 2 (Navi 21..24) | family 0x8f, revision 0x28..0x4f |
| INT8 `FSR4L-i8` (FSR 4 Lite) | Van Gogh (Steam Deck), Strix Point | family 0x90, revision 0x01..0x7f; family 0x96, revision 0x01..0x3f |

The family and revision are the ones AMD's Windows kernel driver reports (`D3DKMTQueryAdapterInfo`). Under Wine there is
no such driver, so AMD's DLL never offers an INT8 network on Linux. This library reads the same two numbers from the
amdgpu kernel driver (`AMDGPU_INFO_DEV_INFO`, through the device's render node, `VK_EXT_physical_device_drm`) and picks
as AMD's DLL does: FP8 where the device can run it, else the INT8 network of the table. **Deviation**: where AMD's DLL
would offer none (an AMD GPU outside the table, such as Raphael, Phoenix or Rembrandt, any other vendor's GPU, or the
family unreadable), this library offers `FSR4-i8` anyway if the device can run it. Each INT8 network has its own
passes for the ultra performance preset, its own postpasses there, and its own initializer data (`FSR4-i8` and
`FSR4-i8-2` share theirs).

## Requirements

- FP8: a GPU and driver with `VK_KHR_cooperative_matrix` and `VK_EXT_shader_float8` (FP8 cooperative matrices), Vulkan 1.3,
  32-wide compute subgroups (subgroup size control), 8/16-bit types, `storageBuffer16BitAccess`, the Vulkan memory
  model, variable pointers and format-less storage image writes. The caller must have **enabled** these on its
  `VkDevice`: `cooperativeMatrix`, `shaderFloat8`, `shaderFloat8CooperativeMatrix`, `shaderInt8`, `shaderInt16`,
  `shaderFloat16`, `storageBuffer16BitAccess`, `vulkanMemoryModel`, `variablePointers`, `variablePointersStorageBuffer`,
  `shaderStorageImageWriteWithoutFormat`, `shaderStorageImageExtendedFormats`, `subgroupSizeControl`
  (`computeFullSubgroups` is not needed). Auto exposure also uses compute shader derivatives
  (`VK_KHR_compute_shader_derivatives`, `computeDerivativeGroupLinear`). Unlike fsr4-vk, `VK_EXT_descriptor_buffer`
  is not needed.
- INT8: Vulkan 1.3 with `shaderIntegerDotProduct`, `shaderFloat16`, `shaderStorageImageWriteWithoutFormat`,
  `shaderStorageImageExtendedFormats`, `subgroupSizeControl` with 32-wide compute subgroups; `FSR4-i8` and `FSR4L-i8`
  also `shaderInt8` and compute shader derivatives (`VK_KHR_compute_shader_derivatives`, `computeDerivativeGroupLinear`:
  their prepass uses them), which `FSR4-i8-2` needs only for auto exposure. These come from the capabilities AMD's INT8
  shaders declare (AMD's DLL itself checks none of them). The caller must have enabled them on its `VkDevice`. Reading
  the GPU's family needs the render node to be openable (`/dev/dri/renderD*`); without it the table above cannot be
  used and `FSR4-i8` is offered (if the device can run it).
- A device that can run none of the networks gets no provider: GET_VERSIONS lists none and create returns
  `FFX_API_RETURN_NO_PROVIDER`, as AMD's DLL does for its 4.1.2 there.

## Build

This repository holds no AMD data. The build makes the `.so` on your machine, from your own copy of AMD's DLL.

Needs CMake, Ninja (or make), Python 3, a C++20 compiler (clang++ or g++), the Vulkan headers and `curl`
(Arch: `cmake ninja python vulkan-headers`; Debian/Ubuntu: `cmake ninja-build python3 libvulkan-dev g++`).

```sh
git clone https://github.com/auz466/fsr4up-vk && cd fsr4up-vk
tools/build/linux.sh
```

Output: `build/amd_fidelityfx_upscaler_vk.so`. The script downloads
`Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll` (36 MB) from AMD's
[FidelityFX SDK v2.4.0](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tree/v2.4.0/Kits/FidelityFX/signedbin)
into `build/dll/`, checks its SHA-256 (`4a7e895af6e0b615d40a84d59eb8c2b1054ad91cdd0c739567940270d51bdab7`), reads it as
data (never runs it), and embeds AMD's shaders and initializer data in the library. Later runs reuse it.

Options: pass a DLL path to use your own copy (`tools/build/linux.sh /path/to/amd_fidelityfx_upscaler_vk.dll`),
`--download` to refetch, `--install DIR` to copy the result somewhere, `--harness` to also build the test harness
(`build/trace`, and `build/wine/trace.exe` with mingw-w64). `./build.sh` is the same script.

## API surface

`ffxCreateContext` (upscale; backend `FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK` or the deprecated `0x3`; a version
override must name 4.1.2's id `0xf5a5ca1e01001002`), `ffxDestroyContext`, `ffxConfigure` (global debug callback,
upscale key-values), `ffxQuery` (GET_VERSIONS, GET_PROVIDER_VERSION, upscale ratio and render size by quality mode,
jitter phase count and offset, GPU memory usage V1 and V2, resource requirements), `ffxDispatch` (upscale; generate
reactive mask, which does nothing, as in AMD's 4.1.2). All the
create flags and dispatch flags of `ffx_upscale.h` are handled as AMD's DLL handles them.

One context is single-threaded, as AMD's is. Up to 15 frames of one context's work may be on the GPU unfinished.
`ffxDestroyContext` does not wait for the GPU: the caller must have finished the work that used it.

## Environment variables

- `FSR4UP_LOG=<file>`: append the library's messages to a file.
- `FSR4UP_VERBOSE=1`: print them to stderr (the network a context runs, and why, is among them).
- `FSR4UP_INT8`: the network. Unset: as above. `0`: no INT8 network (FP8 only). `1`: an INT8 network even where FP8
  runs (the one AMD's table gives for the GPU, else `FSR4-i8`). `i8`, `navi2`, `lite`: that INT8 network
  (`FSR4-i8`, `FSR4-i8-2`, `FSR4L-i8`), even where FP8 runs or the table gives another. For testing.

## Testing

- `tests/against_amd.py`: the regression suite against AMD's DLL under Wine (`harness/run_amd.sh`; set
  `FSR4UP_WINEPREFIX` in `local.env`). For its INT8 cases the harness stands in for AMD's kernel driver
  (`TRACE_AMD_FAMILY`, `TRACE_AMD_REV`) and hides `VK_KHR_cooperative_matrix` from AMD's DLL (`TRACE_HIDE_EXT`), by
  pointing the DLL's imports at its own functions in memory; see `HOW_IT_WAS_MADE.md`. `--gpu <name>` runs it on
  another GPU (e.g. `--gpu RAPHAEL`).
- `tests/validation.sh`: the port under the Khronos validation layer with synchronisation validation.

See `HOW_IT_WAS_MADE.md` for the method, and `THIRD_PARTY_NOTICES.md` for AMD's notice.

## Licence

MIT, see `LICENSE` (this repository's own code). AMD's headers in `third_party/` and the data a build takes from AMD's DLL stay under AMD's MIT licence; see `THIRD_PARTY_NOTICES.md`.
