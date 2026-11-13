# Third-party notices

## AMD FidelityFX SDK 2.4.0 (MIT)

- `third_party/amd-fidelityfx-api/`: AMD's FidelityFX API headers (`Kits/FidelityFX/api/include`,
  `Kits/FidelityFX/upscalers/include`), unmodified. Their notice is beside them.
- `src/fsr4up.cpp`: the quality-mode ratios, the render size from a quality mode, the jitter phase count and the
  Halton jitter sequence follow AMD's open FSR 3 code (`Kits/FidelityFX/upscalers/fsr3/internal/ffx_fsr3upscaler.cpp`);
  the FFX surface-format and resource-state mappings follow AMD's open Vulkan backend
  (`Kits/FidelityFX/backend/vk/ffx_vk.cpp`). Each was checked against AMD's DLL's answers (tests/against_amd.py).
- The built library embeds AMD's SPIR-V shaders and the network's initializer data, taken at build time from AMD's
  `amd_fidelityfx_upscaler_vk.dll` (FSR SDK 2.4.0, `Kits/FidelityFX/signedbin/`). The SDK's licence
  (`Kits/FidelityFX/docs/license.md`) lists that DLL in its MIT section. Nothing extracted is committed here; a
  distributed binary carries AMD's data and needs this notice beside it. (Confirm that reading before any release.)

AMD's licence for these files (the MIT section of the SDK's `docs/license.md`):

Copyright (C)  Advanced Micro Devices, Inc. All rights reserved. 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: 

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
