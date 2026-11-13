# How it was made

The method follows fsr4fg-vk (`~/Games/Hookshot/fsr4fg-vk`): AMD's DLL is the specification, and every behaviour was
traced from it before it was written.

1. **The DLL.** `amd_fidelityfx_upscaler_vk.dll` of FidelityFX SDK 2.4.0 (SHA-256 `4a7e895a...dab7`, the blob of AMD's
   GitHub repository at `v2.4.0`). Its 1,778 SPIR-V modules (1,145 distinct) are zlib streams in `.rdata`, each after
   an `FXZF` tag and its size (`tools/carve.py`). The network's initializer buffer (128 KiB, two presets) is stored
   uncompressed. Only the FP8 network is in the Vulkan DLL, built for three display-size buckets.
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
6. **The suite.** `tests/against_amd.py` runs every case through both and compares every byte of every upscaled frame,
   the harness's inputs, and every API answer. AMD's results are cached per case.

## Redoing it for a new AMD release

1. Point `harness/run_amd.sh` at the new DLL (`-DFSR4UP_AMD_DLL`), and update `dll_sha256`, `version` and the
   initializer offsets in `data/model.json` (find them with `tools/inits.py` on a trace at a normal ratio and at 3x).
2. `python3 -I tools/carve.py <dll> re/carved`, then `tools/matrix.sh` for the three buckets
   (`1280 720 854 480`, `2560 1440 1280 720`, `5120 2880 1280 720`), then `tools/make_model.py`.
3. Rebuild, run the suite, and trace again whatever fails: `tools/summary.py`, `tools/mlsr.py`, `tools/compare.py`.
