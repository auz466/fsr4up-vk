# fsr4up-vk

A native Linux Vulkan implementation of **AMD FSR 4.1.2 upscaling** (the machine-learning one, FP8, FidelityFX SDK
2.4.0). AMD ships 4.1.2 for Vulkan only as a Windows DLL (`amd_fidelityfx_upscaler_vk.dll`); this library answers the
same FSR API calls with AMD's own shaders and initializer data, which the build takes out of that DLL.

This is an independent implementation. It is not AMD's, and AMD does not support it. It reports itself as
`4.1.2 (fsr4up-vk)`.

It drops in where [fsr4-vk](https://github.com/Astyyyyy/fsr4-vk) (FSR 4.1.1, from SDK 2.3.0's DX12 DLL) is used: the
same file name, `amd_fidelityfx_upscaler_vk.so`, and the same entry points.

## Status

- Output **bit-identical to AMD's DLL** (run under Wine on the same GPU) on the regression suite: `tests/against_amd.py`,
  every frame of every case, plus every API answer (see that file for the cases: sizes from 1001x777 to 5120x2880 and
  portrait, every quality mode, still and moving camera, resets, the depth modes, jitter cancellation, display-sized
  motion vectors, auto exposure, an exposure texture, pre-exposure, HDR, non-linear/sRGB/PQ colour, sharpening, the
  debug view, dynamic resolution, a smaller upscale size, input and output formats, the ultra performance preset).
- Tested only on an AMD Radeon RX 9070 XT (RDNA 4) with RADV (Mesa 26.2.6). FP8 only: RDNA 4.
- Not done: the INT8 path (RDNA 2/3; AMD's VK DLL carries no INT8 network), AMD's FSR 3.1.6 and 2.3.4 providers (also in
  its DLL). `FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK` returns OK and records nothing, as AMD's 4.1.2
  does.

## Requirements

- A GPU and driver with `VK_KHR_cooperative_matrix` and `VK_EXT_shader_float8` (FP8 cooperative matrices), Vulkan 1.3,
  32-wide compute subgroups (subgroup size control), 8/16-bit types, `storageBuffer16BitAccess`, the Vulkan memory
  model, variable pointers and format-less storage image writes. The caller must have **enabled** these on its
  `VkDevice`: `cooperativeMatrix`, `shaderFloat8`, `shaderFloat8CooperativeMatrix`, `shaderInt8`, `shaderInt16`,
  `shaderFloat16`, `storageBuffer16BitAccess`, `vulkanMemoryModel`, `variablePointers`, `variablePointersStorageBuffer`,
  `shaderStorageImageWriteWithoutFormat`, `shaderStorageImageExtendedFormats`, `subgroupSizeControl`
  (`computeFullSubgroups` is not needed). Auto exposure also uses compute shader derivatives
  (`VK_KHR_compute_shader_derivatives`, `computeDerivativeGroupLinear`). Unlike fsr4-vk, `VK_EXT_descriptor_buffer`
  is not needed.
- A device without them gets no provider: GET_VERSIONS lists none and create returns `FFX_API_RETURN_NO_PROVIDER`, as
  AMD's DLL does for its 4.1.2 there.

## Build

Needs CMake, a C++20 compiler, Python 3 and the Vulkan headers; for the test harness under Wine, mingw-w64 and Wine.

Get `Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_vk.dll` from AMD's FidelityFX SDK 2.4.0
(https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, tag `v2.4.0`; SHA-256
`4a7e895af6e0b615d40a84d59eb8c2b1054ad91cdd0c739567940270d51bdab7`). Then:

```sh
./build.sh /path/to/amd_fidelityfx_upscaler_vk.dll
```

The build checks the DLL's hash, reads it as data (never runs it), and writes `build/amd_fidelityfx_upscaler_vk.so`
with AMD's shaders and initializer data embedded. Later builds remember the DLL.

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
- `FSR4UP_VERBOSE=1`: print them to stderr.

## Testing

- `tests/against_amd.py`: the regression suite against AMD's DLL under Wine (`harness/run_amd.sh`; set
  `FSR4UP_WINEPREFIX` in `local.env`).
- `tests/validation.sh`: the port under the Khronos validation layer with synchronisation validation.

See `HOW_IT_WAS_MADE.md` for the method, and `THIRD_PARTY_NOTICES.md` for AMD's notice.

## Licence

Not chosen yet.
