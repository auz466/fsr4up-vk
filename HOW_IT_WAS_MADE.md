# How it was made

The method follows fsr4fg-vk (`~/Games/Hookshot/fsr4fg-vk`): AMD's DLL is the specification, and every behaviour was
traced from it before it was written.

1. **The DLL.** `amd_fidelityfx_upscaler_vk.dll` of FidelityFX SDK 2.4.0 (SHA-256 `4a7e895a...dab7`, the blob of AMD's
   GitHub repository at `v2.4.0`). Its 1,778 SPIR-V modules (1,145 distinct) are zlib streams in `.rdata`, each after
   an `FXZF` tag and its size (`tools/carve.py`). The network's initializer buffer (128 KiB, two presets) is stored
   uncompressed. Four networks are in it, each built for three display-size buckets: FP8, and three INT8 ones (see 7).
2. **The harness.** `harness/trace.cpp` runs any FSR API upscaling provider on a synthetic scene and logs every
   device-level Vulkan call it makes (pipelines, bindings, constant data, dispatch sizes, barriers, uploads) and dumps
   its images. It is built for Windows with mingw (`build/wine/trace.exe`, AMD's DLL under Wine:
   `harness/run_amd.sh`) and natively (`build/trace`, the port). The scene is drawn with integer and polynomial
   arithmetic only, so both builds make the same input bytes. Its Vulkan hook layer is fsr4fg-vk's.
3. **What AMD's DLL does.** From traces: the passes (an optional SPD auto-exposure pass, the prepass, 25 network passes,
   the postpass, optional RCAS, optional debug view), their dispatch sizes (`model_groups`; the border passes' counts
   come from constants in their own SPIR-V, read by `tools/reflect.py`), the constant blocks (layouts from the SPIR-V's
   debug names, `tools/spv_structs.py`; values fitted from traces: `tools/mlsr.py` decodes them), the resources and
   their formats, and the memory usage answers.
4. **Which variant.** AMD's DLL picks one of 30 prepasses and 16 postpasses per size bucket by 7 and 5 flag bits.
   `tools/matrix.sh` traces every combination per bucket (768 runs); `tools/make_model.py` turns that into
   `data/model.json` (module SHA-256s, so no offsets into AMD's file but the initializer's).
5. **When it starts over.** AMD's DLL starts its network again (pipelines made again, initializer uploaded again,
   `reset` forced) at the first upscale, when the sRGB/PQ dispatch flags change, and, without
   `FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION`, when the render or upscale size changes. Each time it picks the ultra
   performance initializer when the upscale width over the render width (the largest render width with dynamic
   resolution) is at least 2.99.
6. **The INT8 networks.** All of AMD's modules are named `fsr4_model_v07_fp8_no_scale_*`, but only some use cooperative
   matrices; the others use integer dot products (`SPV_KHR_integer_dot_product`). Their providers' RTTI names them
   (`ffxProvider_FSR4_Int8`, `_Int8_Navi2`, `ffxProvider_FSR4Lite_Int8`; models `FSR4-i8`, `FSR4-i8-2`, `FSR4L-i8`).
   - *The gate.* Disassembled (`llvm-objdump -d` of the DLL; the RTTI's complete-object locators lead to the vtables).
     A static initializer fills a table of six providers: FP8, i8, i8-2, Lite (all with 4.1.2's id `0xf5a5ca1e01001002`
     and name `4.1.2`), 3.1.6, 2.3.4. GET_VERSIONS and create walk it in that order and take every provider whose
     vtable slot 2 (its device check) accepts the device. FP8's check (`0x18000dba0`): vendor 0x1002, not an integrated
     GPU, `VK_KHR_cooperative_matrix` and `VK_EXT_shader_float8` in `vkEnumerateDeviceExtensionProperties`,
     `cooperativeMatrix`. The INT8 checks (`0x18000c7c0`, `0x18000d1c0`, `0x18000ad90`) call `0x18000eb40`, which takes
     the device's LUID (`VkPhysicalDeviceIDProperties`, it must be valid) and, for vendor 0x1002, `0x18000e8b0`: it
     loads `gdi32.dll`, `D3DKMTEnumAdapters`, and `D3DKMTQueryAdapterInfo(KMTQAITYPE_UMDRIVERPRIVATE)` on each adapter,
     reading the family at +0x30 and the external revision at +0x34 of AMD's driver data (or, in its longer form,
     matching vendor and device per GPU). The family/revision ranges are in README.md. With no answer the i8 check
     takes PCI device 0x7481 alone. Wine has no AMD kernel driver data, so the answer never comes: on the RX 9070 XT
     AMD's DLL takes FP8, on the Raphael iGPU (integrated, no cooperative matrices) nothing.
   - *Making AMD's DLL run them.* The DLL imports `vkEnumerateDeviceExtensionProperties` (and the other physical
     device functions) from `vulkan-1.dll` and finds the D3DKMT functions with `GetProcAddress`, outside the device
     functions the harness hooks. So the Windows harness (`spoof` in `harness/trace.cpp`) points two entries of the
     loaded DLL's import table at its own functions (in memory, after `LoadLibrary`; the file and the code are not
     touched): `GetProcAddress`, which hands out a `D3DKMTEnumAdapters` with one adapter and a
     `D3DKMTQueryAdapterInfo` that reports `TRACE_AMD_FAMILY` and `TRACE_AMD_REV`; and
     `vkEnumerateDeviceExtensionProperties`, which leaves out `TRACE_HIDE_EXT`. With
     `TRACE_HIDE_EXT=VK_KHR_cooperative_matrix` and family 0x91 revision 0x01 (a Navi 31), 0x8f 0x28 (Navi 21) or 0x90
     0x01 (Van Gogh) the RX 9070 XT runs AMD's i8, i8-2 or Lite network.
   - *Tracing them.* As for FP8: `tools/matrix.sh` with `MATRIX_TAG` and those variables for each network and bucket,
     at a standard ratio and again at 3x (`matrix-<net>-<bucket>.txt`, `matrix-<net>u-<bucket>.txt`): the prepass and
     postpass hang on the same 7 and 5 flag bits as FP8's. Unlike FP8, the ultra performance preset has network passes
     and postpasses of its own (its prepasses are the standard ones). `tools/sizes.sh` runs each network at 20
     display sizes at 1.5x and 3x: `tools/fit_groups.py` fits every pass's dispatch size (`[s, a, b]`: ceil((W8 >> s)
     / a) by ceil((H8 >> s) / b); the border passes' counts from their SPIR-V, as for FP8), and the runs give the
     scratch size per bucket and the two initializer buffers (128 KiB; 27,392 bytes for Lite; i8 and i8-2 share
     theirs). Lite has 8 passes between prepass and postpass and 9 tensors (levels 1,0,1,2,2,1,0,0,0, read from its
     `CsTensorSizes` data). The constants, resources, formats, pipelines (32-wide subgroups), memory usage formula and
     restarts are FP8's (the provider code is shared: `ffxProvider_FSR4_Common`); the suite's INT8 cases check that.
   - *Not deterministic.* At an odd display width or height, i8's and Lite's prepass (the ones that use compute
     shader derivatives) leave other bytes in the scratch buffer from run to run, also with `RADV_DEBUG=zerovram`, and
     AMD's DLL's frames differ from run to run there as the port's do (found with `TRACE_STOP_ENTRY` and
     `TRACE_DUMP_BUFFERS`); i8-2 is deterministic. The suite compares i8 and Lite at even sizes.
   - *What the port needs of a device* for each INT8 network comes from the capabilities its modules declare
     (`tools/setup.py` puts them in `GenNet::caps`).
7. **The suite.** `tests/against_amd.py` runs every case through both and compares every byte of every upscaled frame,
   the harness's inputs, and every API answer. AMD's results are cached per case.

## Redoing it for a new AMD release

1. Point `harness/run_amd.sh` at the new DLL (`-DFSR4UP_AMD_DLL`), and update `dll_sha256`, `version` and the
   initializer offsets in `data/model.json` (find them with `tools/inits.py` on a trace at a normal ratio and at 3x).
2. `python3 -I tools/carve.py <dll> re/carved`, then `tools/matrix.sh` for the three buckets
   (`1280 720 854 480`, `2560 1440 1280 720`, `5120 2880 1280 720`), then `tools/make_model.py`.
3. The INT8 networks (with `TRACE_HIDE_EXT=VK_KHR_cooperative_matrix`, and per network `TRACE_AMD_FAMILY`/`TRACE_AMD_REV`
   of a GPU AMD's check picks it for: i8 0x91/0x01, i8_navi2 0x8f/0x28, lite 0x90/0x01; check the gate in the new DLL
   first): `MATRIX_TAG=<net> tools/matrix.sh` for `1280 720 854 480`, `2560 1440 1280 720`, `5120 2880 2560 1440`
   into `re/matrix-<net>-<bucket>.txt`, and with `MATRIX_TAG=<net>u` for `1280 720 426 240`, `2560 1440 853 480`,
   `5120 2880 1706 960` into `re/matrix-<net>u-<bucket>.txt`; `tools/sizes.sh <net> TRACE_AMD_FAMILY=.. TRACE_AMD_REV=.. TRACE_HIDE_EXT=..` into
   `re/int8/sizes-<net>.txt`, `tools/fit_groups.py re/int8/sizes-<net>.txt re/int8/fit-<net>.json [--shift ...]`, then
   `INIT_SIZE=<bytes> tools/make_model.py --net <net> --shift <levels>` (the levels from a trace's `CsTensorSizes`).
4. Rebuild, run the suite, and trace again whatever fails: `tools/summary.py`, `tools/mlsr.py`, `tools/compare.py`.
