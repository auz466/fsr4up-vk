// fsr4up-vk: a native Vulkan provider of AMD FSR 4.1.2 upscaling (the machine-learning one: FP8, and INT8), for Linux.
//
// AMD ships FSR 4.1.2 for Vulkan only as a Windows DLL (amd_fidelityfx_upscaler_vk.dll, FSR SDK 2.4.0). This library
// answers the same FSR API entry points (ffxCreateContext, ffxConfigure, ffxQuery, ffxDispatch, ffxDestroyContext) and
// records the same compute passes over the same shaders and initializer data, which tools/setup.py takes out of AMD's
// DLL at build time. What the passes are, in what order, over which images, with which constants, was read off AMD's DLL
// at work (harness/trace.cpp, run under Wine). The jitter and quality-mode helpers follow AMD's open FSR 3 code (MIT).
//
// An upscale is, as AMD's DLL records it:
//   [auto exposure: an SPD pass over the colour]  (FFX_UPSCALE_ENABLE_AUTO_EXPOSURE)
//   prepass: colour, depth, motion and the history reprojected into the network's input in the scratch buffer
//   the network: 25 passes (pass0_post, pass1, pass1_post, ..., pass12_post; 8 for FSR4-Lite) over the scratch buffer,
//            built for one of three size buckets (display up to 1920x1080, up to 3840x2160, or larger)
//   postpass: the network's output into the output (or, to be sharpened, an image of ours), the history and the
//            recurrent state
//   [RCAS sharpening into the output]  (enableSharpening)
//   [the debug view over the output]   (FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW, with FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION)
// Which of AMD's variants of the prepass and postpass runs follows the create and dispatch flags (GEN_NETS, from traces
// of every combination).
//
// The networks (AMD's DLL has four, all reported as 4.1.2): FP8 (AMD's "FSR4", RDNA 4), and three INT8 ones, which AMD's
// DLL offers on other AMD GPUs by their family and revision as its Windows kernel driver reports them: FSR4-i8 (RDNA 3),
// FSR4-i8-2 (RDNA 2) and FSR4L-i8 (FSR 4 Lite: the Steam Deck's Van Gogh, Strix Point). See choose_net().
//
// For FP8 the device must support, and the caller must have enabled on its VkDevice: Vulkan 1.3; VK_KHR_cooperative_matrix
// (cooperativeMatrix); VK_EXT_shader_float8 (shaderFloat8, shaderFloat8CooperativeMatrix); shaderInt8, shaderInt16,
// shaderFloat16, storageBuffer16BitAccess, vulkanMemoryModel, variablePointers(StorageBuffer),
// shaderStorageImageWriteWithoutFormat; subgroupSizeControl with 32-wide compute subgroups. For INT8: Vulkan 1.3,
// shaderIntegerDotProduct, shaderFloat16, shaderInt8 and VK_KHR_compute_shader_derivatives (computeDerivativeGroupLinear)
// but for FSR4-i8-2, shaderStorageImageWriteWithoutFormat, the 32-wide subgroups (see missing_int8()).
//
// Threads: one context is single-threaded, as AMD's is. Frames in flight: up to RING - 1 frames of one context's work may
// be on the GPU unfinished. ffxDestroyContext does not wait for the GPU: the caller must have finished the work.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

// (AMD's header marks the entry points __declspec(dllexport): here that is default visibility, with everything else hidden)
#define __declspec(x) __attribute__((visibility("default")))
#include "ffx/api/include/ffx_api.h"
#include "ffx/api/include/ffx_api_types.h"
#include "ffx/api/include/vk/ffx_api_vk.h"
#include "ffx/upscalers/include/ffx_upscale.h"

#include "assets_gen.h"

extern "C" const unsigned char fsr4up_assets[];   // (assets.S: build/gen/assets.bin)

namespace {

// AMD's id for its 4.1.2 provider, so a caller that picks it from GET_VERSIONS (or overrides with it) lands here
constexpr uint64_t VERSION_ID = 0xf5a5ca1e01001002ull;
constexpr const char* VERSION_NAME = "4.1.2 (fsr4up-vk)";
// AMD's DLL offers these too (traced: GET_VERSIONS lists 4.1.2, 3.1.6, 2.3.4); a version override naming them is refused
constexpr uint32_t RING = 16;
constexpr uint32_t SLOT_BYTES = 2048;   // a frame's constants: MLSR at 0 (and 256 for the debug pass), tensors at 512, SPD at 1024, RCAS at 1280
enum : uint32_t { OFF_MLSR = 0, OFF_TENSOR = 512, OFF_SPD = 1024, OFF_RCAS = 1280 };

// ---------------------------------------------------------------- logging
ffxApiMessage g_message = nullptr;
std::mutex g_message_mu;

void say(bool error, const char* fmt, ...) {
    static const char* path = getenv("FSR4UP_LOG");
    static const bool verbose = getenv("FSR4UP_VERBOSE") != nullptr;
    ffxApiMessage message;
    {
        std::lock_guard lock(g_message_mu);
        message = g_message;
    }
    if (!error && !verbose && !path) return;
    char text[1024];
    va_list a;
    va_start(a, fmt);
    vsnprintf(text, sizeof text, fmt, a);
    va_end(a);
    if (error || verbose) fprintf(stderr, "fsr4up: %s\n", text);
    if (path) {
        static std::mutex mu;
        std::lock_guard lock(mu);
        if (FILE* f = fopen(path, "a")) fprintf(f, "%s\n", text), fclose(f);
    }
    (void)message;
}
/// a message to the caller's callbacks (the context's fpMessage and the global debug one), as AMD's DLL sends them
void tell(ffxApiMessage own, uint32_t type, const char* text) {
    wchar_t wide[512];
    size_t i = 0;
    for (; text[i] && i < 511; i++) wide[i] = (unsigned char)text[i];
    wide[i] = 0;
    ffxApiMessage global;
    {
        std::lock_guard lock(g_message_mu);
        global = g_message;
    }
    if (global) global(type, wide);
    if (own && own != global) own(type, wide);
    say(type == FFX_API_MESSAGE_TYPE_ERROR, "%s", text);
}

// ---------------------------------------------------------------- the device's functions
#define DEVICE_FUNCS(X)                                                                                                \
    X(CreateBuffer) X(DestroyBuffer) X(GetBufferMemoryRequirements) X(AllocateMemory) X(FreeMemory) X(BindBufferMemory) \
    X(MapMemory) X(CreateImage) X(DestroyImage) X(GetImageMemoryRequirements) X(BindImageMemory) X(CreateImageView)     \
    X(DestroyImageView) X(CreateSampler) X(DestroySampler) X(CreateShaderModule) X(DestroyShaderModule)                \
    X(CreateDescriptorSetLayout) X(DestroyDescriptorSetLayout) X(CreatePipelineLayout) X(DestroyPipelineLayout)        \
    X(CreateComputePipelines) X(DestroyPipeline) X(CreateDescriptorPool) X(DestroyDescriptorPool)                      \
    X(ResetDescriptorPool) X(AllocateDescriptorSets) X(UpdateDescriptorSets) X(CmdBindPipeline)                        \
    X(CmdBindDescriptorSets) X(CmdDispatch) X(CmdPipelineBarrier) X(CmdCopyBuffer) X(CmdCopyBufferToImage)             \
    X(CmdClearColorImage)
struct Vk {
#define DECL(n) PFN_vk##n n = nullptr;
    DEVICE_FUNCS(DECL)
#undef DECL
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
    bool fresh = true;   // (not yet transitioned out of UNDEFINED)
};
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* map = nullptr;
};
struct Pipe {
    const GenModule* gen = nullptr;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

/// A caller's image for one dispatch: its handle, the format to view it in, and the layout it is in (by its FFX state)
struct Ext {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    uint32_t width = 0, height = 0;
    VkImageView view = VK_NULL_HANDLE;
};

struct Context {
    ffxAllocationCallbacks alloc{};
    Vk vk;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    ffxApiMessage message = nullptr;

    uint32_t flags = 0;
    uint32_t max_rw = 0, max_rh = 0;   // maxRenderSize
    uint32_t width = 0, height = 0;    // maxUpscaleSize
    const GenNet* net = nullptr;       // the network: FP8 or one of the INT8 ones
    const GenBucket* bucket = nullptr;

    Buffer scratch, init, init_ultra, init_staging, ubo, staging;   // (staging: the default exposure and the SPD counter)
    Image recurrent, history, reprojected, rcas_tmp, autoexp, exposure1, spd_atomic, spd_mip5, debug_vis;
    VkSampler sampler = VK_NULL_HANDLE;
    std::map<uint16_t, Pipe> pipes;   // by GEN_MODULES index, made on first use
    VkDescriptorPool pools[RING]{};
    std::vector<VkImageView> views[RING];

    bool initialised = false;   // (our images out of UNDEFINED, the uploads done)
    // AMD's DLL initialises its network again (traced: new pipelines, the initializer uploaded again, reset forced) at the
    // first upscale, when the sRGB/PQ flags change, and, without DYNAMIC_RESOLUTION, when the render or upscale size
    // changes; each time it picks the preset (ultra performance, or the standard one) by the upscale ratio then.
    bool have_last = false;
    uint32_t last_colour = 0, last_rw = 0, last_rh = 0, last_w = 0, last_h = 0;
    bool ultra = false;
    uint32_t slot = 0;
    uint64_t frames = 0;
    float jitter_prev[2] = {0.0f, 0.0f};
    float pre_exposure_prev = 0.0f;

    // FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE (stored; FSR 4 has no use for them, as AMD's DLL has none: traced)
    float kv[5] = {1.0f, 1.0f, 1.0f, 0.333f, -0.333f};
};

uint32_t memory_type(Context& c, uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c.memory.memoryTypeCount; i++)
        if ((bits >> i & 1) && (c.memory.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}
bool make_buffer(Context& c, Buffer& b, VkDeviceSize size, VkBufferUsageFlags usage, bool host) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size, ci.usage = usage;
    if (c.vk.CreateBuffer(c.device, &ci, nullptr, &b.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    c.vk.GetBufferMemoryRequirements(c.device, b.buffer, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memory_type(c, mr.memoryTypeBits, host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX || c.vk.AllocateMemory(c.device, &ai, nullptr, &b.memory) != VK_SUCCESS) return false;
    if (c.vk.BindBufferMemory(c.device, b.buffer, b.memory, 0) != VK_SUCCESS) return false;
    b.size = size;
    if (host && c.vk.MapMemory(c.device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map) != VK_SUCCESS) return false;
    return true;
}
void drop_buffer(Context& c, Buffer& b) {
    if (b.buffer) c.vk.DestroyBuffer(c.device, b.buffer, nullptr);
    if (b.memory) c.vk.FreeMemory(c.device, b.memory, nullptr);
    b = Buffer{};
}
bool make_image(Context& c, Image& im, VkFormat format, uint32_t w, uint32_t h) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = format, ci.extent = {w, h, 1}, ci.mipLevels = 1, ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT, ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (c.vk.CreateImage(c.device, &ci, nullptr, &im.image) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    c.vk.GetImageMemoryRequirements(c.device, im.image, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size, ai.memoryTypeIndex = memory_type(c, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX || c.vk.AllocateMemory(c.device, &ai, nullptr, &im.memory) != VK_SUCCESS) return false;
    if (c.vk.BindImageMemory(c.device, im.image, im.memory, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.image, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = format;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (c.vk.CreateImageView(c.device, &vi, nullptr, &im.view) != VK_SUCCESS) return false;
    im.format = format, im.width = w, im.height = h;
    return true;
}
void drop_image(Context& c, Image& im) {
    if (im.view) c.vk.DestroyImageView(c.device, im.view, nullptr);
    if (im.image) c.vk.DestroyImage(c.device, im.image, nullptr);
    if (im.memory) c.vk.FreeMemory(c.device, im.memory, nullptr);
    im = Image{};
}

// ---------------------------------------------------------------- formats (as AMD's open FFX Vulkan backend maps them, MIT)
VkFormat vk_format(uint32_t fmt, bool uav) {
    switch (fmt) {
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS: case FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT: return VK_FORMAT_R32G32B32A32_UINT;
    case FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R32G32_UINT: return VK_FORMAT_R32G32_UINT;
    case FFX_API_SURFACE_FORMAT_R8_UINT: return VK_FORMAT_R8_UINT;
    case FFX_API_SURFACE_FORMAT_R32_UINT: return VK_FORMAT_R32_UINT;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS: case FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB: return uav ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8G8B8A8_SRGB;
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS: case FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
    case FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB: return uav ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_B8G8R8A8_SRGB;
    case FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT: return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    case FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM: return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case FFX_API_SURFACE_FORMAT_R16G16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R16G16_UINT: return VK_FORMAT_R16G16_UINT;
    case FFX_API_SURFACE_FORMAT_R16G16_SINT: return VK_FORMAT_R16G16_SINT;
    case FFX_API_SURFACE_FORMAT_R16_FLOAT: return VK_FORMAT_R16_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R16_UINT: return VK_FORMAT_R16_UINT;
    case FFX_API_SURFACE_FORMAT_R16_UNORM: return VK_FORMAT_R16_UNORM;
    case FFX_API_SURFACE_FORMAT_R16_SNORM: return VK_FORMAT_R16_SNORM;
    case FFX_API_SURFACE_FORMAT_R8_UNORM: return VK_FORMAT_R8_UNORM;
    case FFX_API_SURFACE_FORMAT_R8G8_UNORM: return VK_FORMAT_R8G8_UNORM;
    case FFX_API_SURFACE_FORMAT_R8G8_UINT: return VK_FORMAT_R8G8_UINT;
    case FFX_API_SURFACE_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
    case FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP: return VK_FORMAT_E5B9G9R9_UFLOAT_PACK32;
    default: return VK_FORMAT_UNDEFINED;
    }
}
VkImageLayout layout_of_state(uint32_t state) {
    switch (state) {
    case FFX_API_RESOURCE_STATE_COMPUTE_READ: case FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ: case FFX_API_RESOURCE_STATE_PIXEL_READ:
        return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    case FFX_API_RESOURCE_STATE_COPY_SRC: return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    case FFX_API_RESOURCE_STATE_COPY_DEST: return VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    case FFX_API_RESOURCE_STATE_PRESENT: return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    case FFX_API_RESOURCE_STATE_RENDER_TARGET: return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    case FFX_API_RESOURCE_STATE_DEPTH_ATTACHMENT: return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    default: return VK_IMAGE_LAYOUT_GENERAL;   // (COMMON, GENERIC_READ, UNORDERED_ACCESS)
    }
}

// ---------------------------------------------------------------- the passes
const GenBucket* bucket_for(const GenNet& net, uint32_t w, uint32_t h) {
    for (const GenBucket& b : net.buckets)
        if (w <= b.max_w && h <= b.max_h) return &b;
    return &net.buckets[2];
}
/// AMD's choice of prepass and postpass (traced over every combination of the flags that change it; the bits, in order)
uint32_t prepass_key(uint32_t cf, uint32_t df) {
    return (cf & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS ? 1 : 0) | (cf & FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION ? 2 : 0) |
           (cf & FFX_UPSCALE_ENABLE_DEPTH_INVERTED ? 4 : 0) | (cf & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE ? 8 : 0) |
           (cf & FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE ? 16 : 0) | (df & FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB ? 32 : 0) |
           (df & FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ ? 64 : 0);
}
uint32_t postpass_key(uint32_t cf, uint32_t df) {
    return (cf & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE ? 1 : 0) | (cf & FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE ? 2 : 0) |
           (cf & FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION ? 4 : 0) | (df & FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB ? 8 : 0) |
           (df & FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ ? 16 : 0);
}
uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

/// The tensor sizes the network's passes read (CsTensorSizes): the display size padded to 8, by the level of each
/// (traced: the same at every size; 17 of them, 9 for FSR4-Lite)
void tensor_sizes(const GenNet& net, uint32_t w, uint32_t h, uint32_t out[17][4]) {
    const uint32_t pw = (w + 7) & ~7u, ph = (h + 7) & ~7u;
    memset(out, 0, sizeof(uint32_t) * 17 * 4);
    for (int i = 0; i < net.tensors; i++) out[i][0] = pw >> net.tensor_shift[i], out[i][1] = ph >> net.tensor_shift[i];
}
/// The groups of a pass (the prepass, a network pass, the postpass), as AMD's DLL dispatches them (traced: GenGroups)
void pass_groups(const GenGroups& r, const GenModule& m, uint32_t w, uint32_t h, const uint32_t t[17][4], uint32_t g[3]) {
    g[2] = 1;
    if (r.s == 255) {   // a border pass: one thread an element of the border it zeroes (the shader's own count)
        const uint32_t tx = t[m.post_tensor][0], ty = t[m.post_tensor][1];
        const uint32_t px = std::min(m.post_b[0] - tx, (uint32_t)m.post_cap[0]), py = std::min(m.post_b[1] - ty, (uint32_t)m.post_cap[1]);
        const uint32_t row = 1 + tx + px;
        const uint32_t count = row + ty + ty * px + row * py;
        g[0] = div_up(count, m.local), g[1] = 1;
        return;
    }
    const uint32_t pw = (w + 7) & ~7u, ph = (h + 7) & ~7u;
    g[0] = div_up(pw >> r.s, r.a), g[1] = div_up(ph >> r.s, r.b);
}

// The constants blocks, as AMD's shaders declare them (the member names are theirs, from the SPIR-V's debug names).
struct MlsrConstants {
    float inv_size[2], scale[2], inv_scale[2], jitter[2], mv_scale[2], tex_size[2], max_renderSize[2], fMotionVectorJitterCancellation[2];
    uint32_t width, height, reset, width_lr, height_lr;
    float preExposure, previous_preExposure;
    uint32_t rcas_enabled;
    float rcas_sharpness, pad;
};
struct SpdConstants {
    uint32_t mips, numWorkGroups, workGroupOffset[2];
    float invInputSize[2], preExposure, padding;
};
struct RcasConstants {
    uint32_t rcasConfig[4];
    float preExposure;
    uint32_t pad[3];
};
static_assert(sizeof(MlsrConstants) == 104 && sizeof(SpdConstants) == 32 && sizeof(RcasConstants) == 32);

Pipe* pipe(Context& c, uint16_t index) {
    auto it = c.pipes.find(index);
    if (it != c.pipes.end()) return &it->second;
    const GenModule& m = GEN_MODULES[index];
    Pipe p;
    p.gen = &m;
    VkDescriptorSetLayoutBinding b[16];
    for (uint32_t i = 0; i < m.bind_count; i++) {
        static const VkDescriptorType types[] = {VK_DESCRIPTOR_TYPE_SAMPLER, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
        b[i] = {m.binds[i].binding, types[m.binds[i].kind], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dl.bindingCount = m.bind_count, dl.pBindings = b;
    VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    si.codeSize = m.size, si.pCode = (const uint32_t*)(fsr4up_assets + m.offset);
    VkShaderModule module = VK_NULL_HANDLE;
    bool ok = c.vk.CreateDescriptorSetLayout(c.device, &dl, nullptr, &p.set_layout) == VK_SUCCESS;
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1, pl.pSetLayouts = &p.set_layout;
    ok = ok && c.vk.CreatePipelineLayout(c.device, &pl, nullptr, &p.layout) == VK_SUCCESS;
    ok = ok && c.vk.CreateShaderModule(c.device, &si, nullptr, &module) == VK_SUCCESS;
    if (ok) {
        // (AMD's DLL builds the FSR 4 passes for 32-wide subgroups, and the others as the driver likes: traced)
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sub{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        sub.requiredSubgroupSize = 32;
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        ci.stage.pNext = strncmp(m.entry, "fsr4_", 5) == 0 ? &sub : nullptr;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT, ci.stage.module = module, ci.stage.pName = m.entry;
        ci.layout = p.layout;
        ok = c.vk.CreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline) == VK_SUCCESS;
    }
    if (module) c.vk.DestroyShaderModule(c.device, module, nullptr);
    if (!ok) {
        say(true, "the pipeline of %s (module %u) was not made", m.entry, index);
        if (p.pipeline) c.vk.DestroyPipeline(c.device, p.pipeline, nullptr);
        if (p.layout) c.vk.DestroyPipelineLayout(c.device, p.layout, nullptr);
        if (p.set_layout) c.vk.DestroyDescriptorSetLayout(c.device, p.set_layout, nullptr);
        return nullptr;
    }
    return &c.pipes.emplace(index, p).first->second;
}

void destroy(Context* c) {
    for (auto& [i, p] : c->pipes) {
        c->vk.DestroyPipeline(c->device, p.pipeline, nullptr);
        c->vk.DestroyPipelineLayout(c->device, p.layout, nullptr);
        c->vk.DestroyDescriptorSetLayout(c->device, p.set_layout, nullptr);
    }
    for (uint32_t s = 0; s < RING; s++) {
        for (VkImageView v : c->views[s]) c->vk.DestroyImageView(c->device, v, nullptr);
        if (c->pools[s]) c->vk.DestroyDescriptorPool(c->device, c->pools[s], nullptr);
    }
    if (c->sampler) c->vk.DestroySampler(c->device, c->sampler, nullptr);
    for (Image* im : {&c->recurrent, &c->history, &c->reprojected, &c->rcas_tmp, &c->autoexp, &c->exposure1, &c->spd_atomic, &c->spd_mip5, &c->debug_vis})
        drop_image(*c, *im);
    for (Buffer* b : {&c->scratch, &c->init, &c->init_ultra, &c->init_staging, &c->ubo, &c->staging}) drop_buffer(*c, *b);
    const ffxAllocationCallbacks alloc = c->alloc;
    c->~Context();
    if (alloc.dealloc) alloc.dealloc(alloc.pUserData, c);
    else ::operator delete(c);
}

/// What AMD's DLL answers to the GPU memory usage queries (V1 with a context, V2 without): the sum of the memory its
/// context's resources take (traced: every allocation at create but its constant buffer ring and two 16-byte uploads),
/// asked of the driver for the same descriptions without allocating. Its resources are not quite the port's, but the
/// answer is AMD's: what a caller budgets for. Nothing is aliasable. (The INT8 networks: the same, with their own scratch
/// and initializer sizes: traced.)
bool amd_memory_usage(const Vk& vk, VkDevice device, const GenNet& net, uint32_t W, uint32_t H, uint32_t max_rw, uint32_t max_rh, uint32_t flags,
                      uint64_t* total) {
    auto image = [&](VkFormat f, uint32_t w, uint32_t h, VkImageUsageFlags usage) -> uint64_t {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D, ci.format = f, ci.extent = {w, h, 1}, ci.mipLevels = 1, ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT, ci.tiling = VK_IMAGE_TILING_OPTIMAL, ci.usage = usage;
        VkImage im;
        if (vk.CreateImage(device, &ci, nullptr, &im) != VK_SUCCESS) return 0;
        VkMemoryRequirements mr;
        vk.GetImageMemoryRequirements(device, im, &mr);
        vk.DestroyImage(device, im, nullptr);
        return mr.size;
    };
    auto buffer = [&](VkDeviceSize size, VkBufferUsageFlags usage) -> uint64_t {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size, bi.usage = usage;
        VkBuffer b;
        if (vk.CreateBuffer(device, &bi, nullptr, &b) != VK_SUCCESS) return 0;
        VkMemoryRequirements mr;
        vk.GetBufferMemoryRequirements(device, b, &mr);
        vk.DestroyBuffer(device, b, nullptr);
        return mr.size;
    };
    const VkImageUsageFlags all = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    uint64_t parts[] = {
        image(VK_FORMAT_R8G8B8A8_UNORM, W, H, all), 3 * image(VK_FORMAT_R16G16B16A16_SFLOAT, W, H, all), image(VK_FORMAT_R32_SFLOAT, 2, 1, all),
        image(VK_FORMAT_R32G32_SFLOAT, 1, 1, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT),
        image(VK_FORMAT_R32_UINT, 1, 1, all & ~VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
        image(VK_FORMAT_R32_SFLOAT, (max_rw + 15) / 16, (max_rh + 15) / 16, all),
        buffer(bucket_for(net, W, H)->scratch, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        2 * buffer(net.init_size, ssbo | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT),
        (flags & FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION) ? image(VK_FORMAT_R16_UNORM, W, H, all) : 1,
    };
    *total = 0;
    for (uint64_t x : parts) {
        if (!x) return false;
        *total += x;
    }
    if (!(flags & FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION)) *total -= 1;
    return true;
}

// ---------------------------------------------------------------- the device's support
/// The Vulkan loader's physical device functions (the process has the loader: the caller made a device)
struct Loader {
    PFN_vkGetPhysicalDeviceFeatures2 features2 = nullptr;
    PFN_vkGetPhysicalDeviceProperties2 properties2 = nullptr;
    PFN_vkEnumerateDeviceExtensionProperties extensions = nullptr;
};
const Loader* vulkan_loader() {
    static const Loader l = [] {
        Loader x;
        void* lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (!lib) lib = dlopen("libvulkan.so.1", RTLD_NOW);
        if (lib) {
            x.features2 = (PFN_vkGetPhysicalDeviceFeatures2)dlsym(lib, "vkGetPhysicalDeviceFeatures2");
            x.properties2 = (PFN_vkGetPhysicalDeviceProperties2)dlsym(lib, "vkGetPhysicalDeviceProperties2");
            x.extensions = (PFN_vkEnumerateDeviceExtensionProperties)dlsym(lib, "vkEnumerateDeviceExtensionProperties");
        }
        return x;
    }();
    return l.features2 && l.properties2 && l.extensions ? &l : nullptr;
}
std::vector<VkExtensionProperties> device_extensions(const Loader& l, VkPhysicalDevice phys) {
    uint32_t n = 0;
    l.extensions(phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> ext(n);
    l.extensions(phys, nullptr, &n, ext.data());
    ext.resize(n);
    return ext;
}
bool has_extension(const std::vector<VkExtensionProperties>& ext, const char* e) {
    return std::any_of(ext.begin(), ext.end(), [&](const VkExtensionProperties& x) { return !strcmp(x.extensionName, e); });
}

/// What the device lacks of what AMD's FP8 shaders need (from the capabilities and extensions they declare), as a list
/// for a message; empty when it has it all. Only support can be checked: the caller must also have enabled these features
/// on its VkDevice, which Vulkan gives no way to ask afterwards; a pipeline that then fails to build turns into
/// ERROR_RUNTIME_ERROR at create. Not checked: SPV_NV_cooperative_matrix2's conversions, which AMD's shaders also declare
/// and RADV compiles without advertising the extension; compute shader derivatives, which only the auto exposure pass
/// declares (a device without them fails at the first dispatch with auto exposure).
std::string missing_fp8(VkPhysicalDevice phys) {
    const Loader* l = vulkan_loader();
    if (!l) return "the Vulkan loader's physical device functions";
    if (!phys) return "a physical device";
    const auto ext = device_extensions(*l, phys);
    auto has = [&](const char* e) { return has_extension(ext, e); };
    std::string missing;
    auto need = [&](bool ok, const char* what) { if (!ok) missing += (missing.empty() ? "" : ", ") + std::string(what); };
    const bool coop = has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME), fp8 = has(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    need(coop, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME), need(fp8, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    f.pNext = &f11, f11.pNext = &f12, f12.pNext = &f13;
    void** tail = &f13.pNext;   // (only the extensions the device has may be chained)
    if (coop) *tail = &fc, tail = &fc.pNext;
    if (fp8) *tail = &f8, tail = &f8.pNext;
    l->features2(phys, &f);
    VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    p.pNext = &p13;
    l->properties2(phys, &p);
    need(p.properties.apiVersion >= VK_API_VERSION_1_3, "Vulkan 1.3");
    if (coop) need(fc.cooperativeMatrix, "cooperativeMatrix");
    if (fp8) need(f8.shaderFloat8, "shaderFloat8"), need(f8.shaderFloat8CooperativeMatrix, "shaderFloat8CooperativeMatrix");
    need(f.features.shaderInt16, "shaderInt16"), need(f.features.shaderStorageImageWriteWithoutFormat, "shaderStorageImageWriteWithoutFormat");
    need(f.features.shaderStorageImageExtendedFormats, "shaderStorageImageExtendedFormats");
    need(f11.storageBuffer16BitAccess, "storageBuffer16BitAccess");
    need(f12.shaderFloat16, "shaderFloat16"), need(f12.shaderInt8, "shaderInt8"), need(f12.vulkanMemoryModel, "vulkanMemoryModel");
    need(f11.variablePointers && f11.variablePointersStorageBuffer, "variablePointers");
    need(f13.subgroupSizeControl && p13.minSubgroupSize <= 32 && p13.maxSubgroupSize >= 32 &&
             (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT),
         "subgroupSizeControl with 32-wide compute subgroups");
    return missing;
}

/// The same for an INT8 network, from the capabilities its own passes declare (GenNet::caps, read off AMD's SPIR-V by
/// tools/setup.py). AMD's DLL checks none of this (its INT8 providers look only at the GPU's family: choose_net()).
std::string missing_int8(VkPhysicalDevice phys, const GenNet& net) {
    const Loader* l = vulkan_loader();
    if (!l) return "the Vulkan loader's physical device functions";
    if (!phys) return "a physical device";
    const auto ext = device_extensions(*l, phys);
    std::string missing;
    auto need = [&](bool ok, const char* what) { if (!ok) missing += (missing.empty() ? "" : ", ") + std::string(what); };
    const uint32_t caps = net.caps;
    const bool khr_derivatives = has_extension(ext, VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
    const bool nv_derivatives = has_extension(ext, VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
    VkPhysicalDeviceFeatures2 f{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR fd{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    f.pNext = &f11, f11.pNext = &f12, f12.pNext = &f13;
    if (khr_derivatives || nv_derivatives) f13.pNext = &fd;   // (the NV structure is the same, with the same sType)
    l->features2(phys, &f);
    VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceVulkan13Properties p13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    p.pNext = &p13;
    l->properties2(phys, &p);
    need(p.properties.apiVersion >= VK_API_VERSION_1_3, "Vulkan 1.3");
    need(!(caps & (CAP_COOPERATIVE_MATRIX | CAP_FLOAT8 | CAP_FLOAT8_COOPERATIVE_MATRIX)), "(an FP8 network: not checked here)");
    if (caps & (CAP_DOT_PRODUCT | CAP_DOT_PRODUCT_4X8_PACKED)) need(f13.shaderIntegerDotProduct, "shaderIntegerDotProduct");
    if (caps & CAP_FLOAT16) need(f12.shaderFloat16, "shaderFloat16");
    if (caps & CAP_INT8) need(f12.shaderInt8, "shaderInt8");
    if (caps & CAP_INT16) need(f.features.shaderInt16, "shaderInt16");
    if (caps & CAP_STORAGE_BUFFER_16BIT) need(f11.storageBuffer16BitAccess, "storageBuffer16BitAccess");
    if (caps & CAP_VULKAN_MEMORY_MODEL) need(f12.vulkanMemoryModel, "vulkanMemoryModel");
    if (caps & (CAP_VARIABLE_POINTERS | CAP_VARIABLE_POINTERS_STORAGE_BUFFER)) need(f11.variablePointers && f11.variablePointersStorageBuffer, "variablePointers");
    if (caps & CAP_IMAGE_WRITE_WITHOUT_FORMAT) need(f.features.shaderStorageImageWriteWithoutFormat, "shaderStorageImageWriteWithoutFormat");
    if (caps & CAP_IMAGE_EXTENDED_FORMATS) need(f.features.shaderStorageImageExtendedFormats, "shaderStorageImageExtendedFormats");
    if (caps & CAP_DERIVATIVE_GROUP_LINEAR)
        need((khr_derivatives || nv_derivatives) && fd.computeDerivativeGroupLinear, "VK_KHR_compute_shader_derivatives (computeDerivativeGroupLinear)");
    need(f13.subgroupSizeControl && p13.minSubgroupSize <= 32 && p13.maxSubgroupSize >= 32 &&
             (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT),
         "subgroupSizeControl with 32-wide compute subgroups");
    return missing;
}

/// The GPU as AMD's DLL sees it for its INT8 providers: the PCI vendor and device, and (AMD's GPUs) the family and the
/// external revision that its Windows kernel driver reports (D3DKMTQueryAdapterInfo); on Linux the amdgpu kernel driver
/// gives the same two numbers (AMDGPU_INFO_DEV_INFO), through the device's render node (VK_EXT_physical_device_drm).
struct GpuId {
    uint32_t vendor = 0, device = 0;
    bool known = false;   // (family and revision read)
    uint32_t family = 0, revision = 0;
};
// (the kernel's UAPI, drm/amdgpu_drm.h: DRM_IOCTL_AMDGPU_INFO with AMDGPU_INFO_DEV_INFO, whose answer starts with
// device_id, chip_rev, external_rev, pci_rev, family)
struct AmdgpuInfoRequest {
    uint64_t return_pointer;
    uint32_t return_size, query;
    uint32_t args[4];
};
static_assert(sizeof(AmdgpuInfoRequest) == 32);
constexpr unsigned long AMDGPU_INFO_IOCTL = _IOW('d', 0x40 + 0x05, AmdgpuInfoRequest);
constexpr uint32_t AMDGPU_INFO_DEV_INFO_QUERY = 0x16;

GpuId gpu_id(VkPhysicalDevice phys) {
    GpuId id;
    const Loader* l = vulkan_loader();
    if (!l || !phys) return id;
    const bool drm = has_extension(device_extensions(*l, phys), VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME);
    VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceDrmPropertiesEXT pd{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
    if (drm) p.pNext = &pd;
    l->properties2(phys, &p);
    id.vendor = p.properties.vendorID, id.device = p.properties.deviceID;
    if (id.vendor != 0x1002 || !drm || !pd.hasRender) return id;
    char path[64];
    snprintf(path, sizeof path, "/dev/dri/renderD%lld", (long long)pd.renderMinor);
    const int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) return id;
    uint32_t info[64] = {};   // (the kernel copies as much of its answer as asked for)
    AmdgpuInfoRequest q{};
    q.return_pointer = (uint64_t)(uintptr_t)info, q.return_size = sizeof info, q.query = AMDGPU_INFO_DEV_INFO_QUERY;
    if (ioctl(fd, AMDGPU_INFO_IOCTL, &q) == 0 && info[0] == id.device) id.known = true, id.revision = info[2], id.family = info[4];
    close(fd);
    return id;
}

/// AMD's choice among its INT8 networks, as its DLL makes it (each provider's IsSupported, in the order its providers are
/// listed, after FP8's: ffxProvider_FSR4_Int8, ffxProvider_FSR4_Int8_Navi2, ffxProvider_FSR4Lite_Int8; read in its code),
/// or -1: by the family and revision when they can be read, else FSR4-i8 for the PCI device 0x7481 alone.
int amd_int8_net(const GpuId& id) {
    if (!id.known) return id.device == 0x7481 ? NET_I8 : -1;
    const uint32_t f = id.family, r = id.revision;
    if ((f == 0x91 && r >= 0x01 && r <= 0xfe) || (f == 0x96 && r >= 0xc0 && r <= 0xcf)) return NET_I8;   // RDNA 3 (Navi 3x), Strix Halo
    if (f == 0x8f && r >= 0x28 && r <= 0x4f) return NET_I8_NAVI2;                                       // RDNA 2 (Navi 21 to 24)
    if ((f == 0x96 && r >= 0x01 && r <= 0x3f) || (f == 0x90 && r >= 0x01 && r <= 0x7f)) return NET_LITE; // Strix Point, Van Gogh
    return -1;
}

/// The network for a device (an index into GEN_NETS), or -1 with what it lacks in *why. As AMD's DLL: FP8 when the device
/// can run it; else the INT8 network AMD's DLL would pick for the GPU. Where AMD's DLL would pick none (it reads the GPU's
/// family from its Windows kernel driver, which Wine does not have: on Linux its INT8 providers never appear), this library
/// offers FSR4-i8 anyway, on any device that can run it: a deviation (README.md). FSR4UP_INT8 overrides it:
///   0                  no INT8 network (only FP8, as before this library had INT8)
///   1                  an INT8 network even where FP8 runs: the one AMD's DLL would pick, else FSR4-i8
///   i8, navi2, lite    that INT8 network (FSR4-i8, FSR4-i8-2, FSR4L-i8), even where FP8 runs or AMD would pick another
int choose_net(VkPhysicalDevice phys, std::string* why, std::string* how = nullptr) {
    const char* e = getenv("FSR4UP_INT8");
    const std::string env = e ? e : "";
    int forced = -2;   // -2: automatic, -1: no INT8, NET_COUNT: AMD's INT8 choice, else that network
    if (env == "0") forced = -1;
    else if (env == "1") forced = NET_COUNT;
    else if (env == "i8") forced = NET_I8;
    else if (env == "navi2" || env == "i8-2" || env == "i8_navi2") forced = NET_I8_NAVI2;
    else if (env == "lite") forced = NET_LITE;
    else if (!env.empty()) say(true, "FSR4UP_INT8=%s: not 0, 1, i8, navi2 or lite; ignored", env.c_str());
    const std::string fp8 = missing_fp8(phys);
    if (forced == -2 && fp8.empty()) {
        if (how) *how = "FP8";
        return NET_FP8;
    }
    if (forced == -1) {
        if (fp8.empty()) {
            if (how) *how = "FP8";
            return NET_FP8;
        }
        if (why) *why = fp8 + " (FP8; FSR4UP_INT8=0: no INT8)";
        return -1;
    }
    const GpuId id = gpu_id(phys);
    const int amd = amd_int8_net(id);
    const int net = forced >= 0 && forced < NET_COUNT ? forced : amd >= 0 ? amd : NET_I8;
    const std::string missing = missing_int8(phys, GEN_NETS[net]);
    char gpu[96];
    if (id.known) snprintf(gpu, sizeof gpu, "family 0x%x revision 0x%x", id.family, id.revision);
    else snprintf(gpu, sizeof gpu, "device 0x%04x:0x%04x, its family unknown", id.vendor, id.device);
    if (!missing.empty()) {
        if (why) *why = missing + " (INT8, " + GEN_NETS[net].amd_name + ")" + (fp8.empty() ? "" : "; for FP8: " + fp8);
        return -1;
    }
    if (how) {
        char t[256];
        snprintf(t, sizeof t, "INT8 %s (%s; %s)", GEN_NETS[net].amd_name, gpu,
                 forced >= 0 && forced < NET_COUNT ? "FSR4UP_INT8" : amd >= 0 ? "AMD's choice for it" : "AMD's DLL offers no FSR 4.1.2 here: offered anyway");
        *how = t;
    }
    return net;
}

const ffxApiHeader* find(const ffxApiHeader* h, uint64_t type) {
    for (; h; h = h->pNext)
        if (h->type == type) return h;
    return nullptr;
}

// ---------------------------------------------------------------- recording a frame
struct Recorder {
    Context& c;
    VkCommandBuffer cb;
    VkDescriptorPool pool;
    uint32_t slot;
    bool first = true;

    void barrier(bool transfer_before = false) {
        VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        c.vk.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | (transfer_before ? VK_PIPELINE_STAGE_HOST_BIT : 0),
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
    }
    void layout(VkImage im, VkImageLayout from, VkImageLayout to) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        b.oldLayout = from, b.newLayout = to;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im, b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        c.vk.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
};

/// What each resource of AMD's shaders is, this frame
struct Bindings {
    VkImageView image[RES_COUNT]{};
    VkBuffer buffer[RES_COUNT]{};
    VkDeviceSize offset[RES_COUNT]{}, range[RES_COUNT]{};
};

bool run(Recorder& r, uint16_t module, const Bindings& b, const uint32_t groups[3]) {
    Pipe* p = pipe(r.c, module);
    if (!p) return false;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = r.pool, ai.descriptorSetCount = 1, ai.pSetLayouts = &p->set_layout;
    VkDescriptorSet set;
    if (r.c.vk.AllocateDescriptorSets(r.c.device, &ai, &set) != VK_SUCCESS) return false;
    VkWriteDescriptorSet w[16];
    VkDescriptorImageInfo ii[16];
    VkDescriptorBufferInfo bi[16];
    const GenModule& m = *p->gen;
    for (uint32_t i = 0; i < m.bind_count; i++) {
        const GenBind& g = m.binds[i];
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = set, w[i].dstBinding = g.binding, w[i].descriptorCount = 1;
        switch (g.kind) {
        case KIND_SAMPLER:
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, ii[i] = {r.c.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED}, w[i].pImageInfo = &ii[i];
            break;
        case KIND_SAMPLED:
        case KIND_STORAGE:
            if (!b.image[g.res]) {
                say(true, "%s binding %u: no image for resource %u", m.entry, g.binding, g.res);
                return false;
            }
            w[i].descriptorType = g.kind == KIND_SAMPLED ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            ii[i] = {VK_NULL_HANDLE, b.image[g.res], VK_IMAGE_LAYOUT_GENERAL}, w[i].pImageInfo = &ii[i];
            break;
        case KIND_UBO:
        case KIND_SSBO:
            if (!b.buffer[g.res]) {
                say(true, "%s binding %u: no buffer for resource %u", m.entry, g.binding, g.res);
                return false;
            }
            w[i].descriptorType = g.kind == KIND_UBO ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bi[i] = {b.buffer[g.res], b.offset[g.res], b.range[g.res]}, w[i].pBufferInfo = &bi[i];
            break;
        }
    }
    r.c.vk.UpdateDescriptorSets(r.c.device, m.bind_count, w, 0, nullptr);
    if (!r.first) r.barrier();
    r.first = false;
    r.c.vk.CmdBindPipeline(r.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipeline);
    r.c.vk.CmdBindDescriptorSets(r.cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->layout, 0, 1, &set, 0, nullptr);
    r.c.vk.CmdDispatch(r.cb, groups[0], groups[1], groups[2]);
    return true;
}

ffxReturnCode_t dispatch_upscale(Context& c, const ffxDispatchDescUpscale& d) {
    VkCommandBuffer cb = (VkCommandBuffer)d.commandList;
    if (!cb || !d.color.resource || !d.depth.resource || !d.motionVectors.resource || !d.output.resource) {
        tell(c.message, FFX_API_MESSAGE_TYPE_ERROR, "an input or the output is missing");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const bool debug_view = (d.flags & FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW) != 0;
    if (debug_view && !(c.flags & FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION)) return FFX_API_RETURN_ERROR_PARAMETER;   // (as AMD's: traced)
    if ((c.flags & FFX_UPSCALE_ENABLE_DEBUG_CHECKING) && !d.exposure.resource && !(c.flags & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE)) {
        tell(c.message, FFX_API_MESSAGE_TYPE_ERROR, "exposure resource is null and auto exposure flag is unset");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const uint32_t rw = d.renderSize.width, rh = d.renderSize.height;
    if (!rw || !rh) {
        tell(c.message, FFX_API_MESSAGE_TYPE_ERROR, "the render size is zero");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const uint32_t W = d.upscaleSize.width ? d.upscaleSize.width : c.width, H = d.upscaleSize.height ? d.upscaleSize.height : c.height;
    const uint32_t cf = c.flags, df = d.flags;

    // the slot: its descriptor pool and the views of the caller's images it made RING frames ago are free again
    const uint32_t slot = c.slot;
    c.slot = (c.slot + 1) % RING;
    for (VkImageView v : c.views[slot]) c.vk.DestroyImageView(c.device, v, nullptr);
    c.views[slot].clear();
    c.vk.ResetDescriptorPool(c.device, c.pools[slot], 0);
    Recorder r{c, cb, c.pools[slot], slot};

    auto ext = [&](const FfxApiResource& res, bool uav) {
        Ext e;
        e.image = (VkImage)res.resource, e.format = vk_format(res.description.format, uav), e.layout = layout_of_state(res.state);
        e.width = res.description.width, e.height = res.description.height;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = e.image, vi.viewType = VK_IMAGE_VIEW_TYPE_2D, vi.format = e.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (c.vk.CreateImageView(c.device, &vi, nullptr, &e.view) == VK_SUCCESS) c.views[slot].push_back(e.view);
        else e.view = VK_NULL_HANDLE;
        return e;
    };
    Ext color = ext(d.color, false), depth = ext(d.depth, false), motion = ext(d.motionVectors, false), output = ext(d.output, true);
    Ext exposure = d.exposure.resource ? ext(d.exposure, false) : Ext{};
    if (!color.view || !depth.view || !motion.view || !output.view || (d.exposure.resource && !exposure.view)) return FFX_API_RETURN_ERROR_RUNTIME_ERROR;

    // what the caller wrote before made visible to the passes (AMD's DLL always starts with a barrier, whatever the
    // resources' states), and the caller's images in GENERAL for the frame, back to their own layouts after
    {
        VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        c.vk.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m,
                                0, nullptr, 0, nullptr);
    }
    std::vector<Ext*> callers = {&color, &depth, &motion, &output};
    if (exposure.image) callers.push_back(&exposure);
    for (Ext* e : callers)
        if (e->layout != VK_IMAGE_LAYOUT_GENERAL) r.layout(e->image, e->layout, VK_IMAGE_LAYOUT_GENERAL);
    // our own images: out of UNDEFINED once; the default exposure (1, 1), the SPD counter (0) and the initializer uploaded
    if (!c.initialised) {
        for (Image* im : {&c.recurrent, &c.history, &c.reprojected, &c.rcas_tmp, &c.autoexp, &c.exposure1, &c.spd_atomic, &c.spd_mip5, &c.debug_vis})
            if (im->image) r.layout(im->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkBufferImageCopy bc{};
        bc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, bc.imageExtent = {1, 1, 1};
        c.vk.CmdCopyBufferToImage(cb, c.staging.buffer, c.exposure1.image, VK_IMAGE_LAYOUT_GENERAL, 1, &bc);
        bc.bufferOffset = 256;
        c.vk.CmdCopyBufferToImage(cb, c.staging.buffer, c.spd_atomic.image, VK_IMAGE_LAYOUT_GENERAL, 1, &bc);
        VkBufferCopy copy{0, 0, c.net->init_size};
        c.vk.CmdCopyBuffer(cb, c.init_staging.buffer, c.init.buffer, 1, &copy);
        copy.srcOffset = c.net->init_size;
        c.vk.CmdCopyBuffer(cb, c.init_staging.buffer, c.init_ultra.buffer, 1, &copy);
        r.barrier(true);
        c.initialised = true;
    }

    // a new start of the network, as AMD's DLL makes one (see Context::have_last)
    const uint32_t colour = df & (FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB | FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ);
    const bool dynamic = (cf & FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION) != 0;
    const bool restart = !c.have_last || colour != c.last_colour ||
                         (!dynamic && (rw != c.last_rw || rh != c.last_rh || W != c.last_w || H != c.last_h));
    if (restart) {
        // (the ratio of the upscale width to the render width, or with DYNAMIC_RESOLUTION to the largest: traced, 2.99 the
        // edge, the widths as they are, not padded)
        c.ultra = (float)W / (float)(dynamic ? c.max_rw : rw) >= 2.99f;
        say(false, "network (re)started: the %s preset", c.ultra ? "ultra performance" : "standard");
    }
    c.have_last = true, c.last_colour = colour, c.last_rw = rw, c.last_rh = rh, c.last_w = W, c.last_h = H;

    // the constants
    uint8_t* slot_bytes = (uint8_t*)c.ubo.map + (size_t)slot * SLOT_BYTES;
    const VkDeviceSize slot_off = (VkDeviceSize)slot * SLOT_BYTES;
    MlsrConstants k{};
    k.inv_size[0] = 1.0f / (float)W, k.inv_size[1] = 1.0f / (float)H;
    k.scale[0] = (float)W / (float)rw, k.scale[1] = (float)H / (float)rh;
    k.inv_scale[0] = 1.0f / k.scale[0], k.inv_scale[1] = 1.0f / k.scale[1];   // (as AMD's: not render / display, traced)
    k.jitter[0] = d.jitterOffset.x, k.jitter[1] = d.jitterOffset.y;
    const bool display_mv = (cf & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0;
    k.mv_scale[0] = d.motionVectorScale.x / (float)(display_mv ? W : rw);
    k.mv_scale[1] = d.motionVectorScale.y / (float)(display_mv ? H : rh);
    k.tex_size[0] = (float)c.width, k.tex_size[1] = (float)c.height;   // (the largest upscale size: traced)
    k.max_renderSize[0] = (float)c.max_rw, k.max_renderSize[1] = (float)c.max_rh;
    if (cf & FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) {
        // (over the motion vectors' size: the render size, or the upscale size with display-sized ones: traced)
        k.fMotionVectorJitterCancellation[0] = (c.jitter_prev[0] - d.jitterOffset.x) / (float)(display_mv ? W : rw);
        k.fMotionVectorJitterCancellation[1] = (c.jitter_prev[1] - d.jitterOffset.y) / (float)(display_mv ? H : rh);
    }
    k.width = W, k.height = H, k.reset = d.reset || restart ? 1 : 0, k.width_lr = rw, k.height_lr = rh;
    k.preExposure = d.preExposure, k.previous_preExposure = c.pre_exposure_prev;
    k.rcas_enabled = d.enableSharpening ? 1 : 0, k.rcas_sharpness = d.enableSharpening ? d.sharpness : 0.0f;
    memcpy(slot_bytes + OFF_MLSR, &k, sizeof k);
    uint32_t tensors[17][4];
    tensor_sizes(*c.net, W, H, tensors);
    memcpy(slot_bytes + OFF_TENSOR, tensors, sizeof tensors);
    c.jitter_prev[0] = d.jitterOffset.x, c.jitter_prev[1] = d.jitterOffset.y;
    c.pre_exposure_prev = d.preExposure;

    Bindings b;
    b.buffer[RES_SCRATCH] = c.scratch.buffer, b.range[RES_SCRATCH] = VK_WHOLE_SIZE;
    b.buffer[RES_INIT] = c.ultra ? c.init_ultra.buffer : c.init.buffer, b.range[RES_INIT] = VK_WHOLE_SIZE;
    b.buffer[RES_UBO_MLSR] = c.ubo.buffer, b.offset[RES_UBO_MLSR] = slot_off + OFF_MLSR, b.range[RES_UBO_MLSR] = sizeof(MlsrConstants);
    b.buffer[RES_UBO_TENSOR] = c.ubo.buffer, b.offset[RES_UBO_TENSOR] = slot_off + OFF_TENSOR, b.range[RES_UBO_TENSOR] = 16 * c.net->tensors;
    b.buffer[RES_UBO_SPD] = c.ubo.buffer, b.offset[RES_UBO_SPD] = slot_off + OFF_SPD, b.range[RES_UBO_SPD] = sizeof(SpdConstants);
    b.buffer[RES_UBO_RCAS] = c.ubo.buffer, b.offset[RES_UBO_RCAS] = slot_off + OFF_RCAS, b.range[RES_UBO_RCAS] = sizeof(RcasConstants);
    b.image[RES_HISTORY] = c.history.view, b.image[RES_RECURRENT] = c.recurrent.view, b.image[RES_REPROJECTED] = c.reprojected.view;
    b.image[RES_AUTOEXP] = c.autoexp.view, b.image[RES_SPD_ATOMIC] = c.spd_atomic.view, b.image[RES_SPD_MIP5] = c.spd_mip5.view;
    b.image[RES_COLOR] = color.view, b.image[RES_DEPTH] = depth.view, b.image[RES_MOTION] = motion.view;
    b.image[RES_EXPOSURE] = exposure.view ? exposure.view : c.exposure1.view;
    b.image[RES_MLSR_OUT] = d.enableSharpening ? c.rcas_tmp.view : output.view;
    b.image[RES_DEBUG_VIS] = c.debug_vis.view;

    // (the INT8 networks have passes of their own for ultra performance, and postpasses: traced; FP8's are the same)
    const GenPreset& preset = c.bucket->preset[c.ultra ? 1 : 0];
    const GenNet& net = *c.net;
    uint32_t g[3];
    bool ok = true;
    if (debug_view && c.debug_vis.image) {
        const VkClearColorValue zero{};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        c.vk.CmdClearColorImage(cb, c.debug_vis.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        r.first = false;
    }
    if (cf & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE) {   // SPD over the colour (at the render size), as AMD's FFX SPD setup counts it
        SpdConstants s{};
        const uint32_t ex = (rw - 1) / 64, ey = (rh - 1) / 64;
        g[0] = ex + 1, g[1] = ey + 1, g[2] = 1;
        s.numWorkGroups = g[0] * g[1];
        s.mips = (uint32_t)std::log2((float)std::max(rw, rh));
        s.invInputSize[0] = 1.0f / (float)rw, s.invInputSize[1] = 1.0f / (float)rh;
        s.preExposure = d.preExposure;
        memcpy(slot_bytes + OFF_SPD, &s, sizeof s);
        ok = ok && run(r, GEN_SPD, b, g);
    }
    const uint16_t pre = preset.prepass[prepass_key(cf, df)];
    pass_groups(preset.groups[0], GEN_MODULES[pre], W, H, tensors, g);
    ok = ok && run(r, pre, b, g);
    for (int place = 0; place < net.passes && ok; place++) {
        pass_groups(preset.groups[1 + place], GEN_MODULES[preset.model[place]], W, H, tensors, g);
        ok = run(r, preset.model[place], b, g);
    }
    const uint16_t post = preset.postpass[postpass_key(cf, df)];
    pass_groups(preset.groups[1 + net.passes], GEN_MODULES[post], W, H, tensors, g);
    ok = ok && run(r, post, b, g);
    if (ok && d.enableSharpening) {
        RcasConstants rc{};
        // (FSR's RCAS: sharpness in stops, 0 the sharpest; the API's 0..1 the other way round)
        const float stops = -2.0f * d.sharpness + 2.0f;
        const float sharp = exp2f(-stops);
        memcpy(&rc.rcasConfig[0], &sharp, 4);
        uint16_t h;
        {
            // (half of sharp, as packHalf2x16)
            uint32_t x;
            memcpy(&x, &sharp, 4);
            const uint32_t sign = (x >> 16) & 0x8000;
            int e = (int)((x >> 23) & 0xff) - 127 + 15;
            uint32_t man = x & 0x7fffff;
            if (e <= 0) h = (uint16_t)(e < -10 ? sign : sign | (((man | 0x800000) >> (1 - e)) + 0x1000) >> 13);
            else if (e >= 31) h = (uint16_t)(sign | 0x7c00);
            else h = (uint16_t)(sign | (e << 10) | ((man + 0x1000) >> 13));
        }
        rc.rcasConfig[1] = (uint32_t)h | (uint32_t)h << 16;
        rc.preExposure = d.preExposure;
        memcpy(slot_bytes + OFF_RCAS, &rc, sizeof rc);
        b.image[RES_RCAS_IN] = c.rcas_tmp.view, b.image[RES_RCAS_OUT] = output.view;
        // (AMD's variants by the colour: PQ, else sRGB, else non-linear, else with auto exposure, else the plain one: traced)
        const uint16_t rcas = (df & FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ)     ? GEN_RCAS_219
                              : (df & FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB) ? GEN_RCAS_223
                              : (cf & FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE) ? GEN_RCAS_225
                              : (cf & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE)       ? GEN_RCAS_220
                                                                              : GEN_RCAS_222;
        g[0] = div_up(W, 16), g[1] = div_up(H, 16), g[2] = 1;
        ok = run(r, rcas, b, g);
    }
    if (ok && debug_view) {
        b.image[RES_RESULT] = output.view, b.image[RES_RCAS_OUT] = output.view, b.image[RES_MLSR_OUT] = output.view;
        g[0] = div_up(W, 8), g[1] = div_up(H, 8), g[2] = 1;
        ok = run(r, GEN_DEBUG, b, g);
    }
    {   // (the output, and anything else written, made visible to whatever the caller records next)
        VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        m.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        c.vk.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m,
                                0, nullptr, 0, nullptr);
    }
    for (Ext* e : callers)
        if (e->layout != VK_IMAGE_LAYOUT_GENERAL) r.layout(e->image, VK_IMAGE_LAYOUT_GENERAL, e->layout);
    c.frames++;
    return ok ? FFX_API_RETURN_OK : FFX_API_RETURN_ERROR_RUNTIME_ERROR;
}

// ---------------------------------------------------------------- the queries every provider answers the same way
// (AMD's open FSR 3 code, ffx_fsr3upscaler.cpp, MIT: the ratios, the render size, the jitter sequence)
float ratio_of(uint32_t mode) {
    switch (mode) {
    case FFX_UPSCALE_QUALITY_MODE_NATIVEAA: return 1.0f;
    case FFX_UPSCALE_QUALITY_MODE_QUALITY: return 1.5f;
    case FFX_UPSCALE_QUALITY_MODE_BALANCED: return 1.7f;
    case FFX_UPSCALE_QUALITY_MODE_PERFORMANCE: return 2.0f;
    case FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE: return 3.0f;
    default: return 0.0f;
    }
}
float halton(int32_t index, int32_t base) {
    float f = 1.0f, result = 0.0f;
    for (int32_t i = index; i > 0;) {
        f /= (float)base;
        result = result + f * (float)(i % base);
        i = (int32_t)(uint32_t)(floorf((float)i / (float)base));
    }
    return result;
}

} // namespace

// ---------------------------------------------------------------- the FSR API
extern "C" {

ffxReturnCode_t ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb) {
    if (!context || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    const auto* up = (const ffxCreateContextDescUpscale*)find(desc, FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE);
    if (!up) return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    const auto* be = (const ffxCreateBackendVKDesc*)find(desc, FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK);
    if (!be) be = (const ffxCreateBackendVKDesc*)find(desc, FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK_DEPRECATED);
    if (!be || !be->deviceContext.vkDevice || !be->deviceContext.vkPhysicalDevice || !be->deviceContext.vkDeviceProcAddr) return FFX_API_RETURN_ERROR_PARAMETER;
    if (const auto* ov = (const ffxOverrideVersion*)find(desc, FFX_API_DESC_TYPE_OVERRIDE_VERSION))
        if (ov->versionId != VERSION_ID) {
            say(true, "version 0x%llx asked for: only 4.1.2 (0x%llx) is here", (unsigned long long)ov->versionId, (unsigned long long)VERSION_ID);
            return FFX_API_RETURN_NO_PROVIDER;
        }
    // (AMD's DLL answers NO_PROVIDER for 4.1.2 on a device that cannot run it: traced on an iGPU without cooperative matrices)
    std::string missing, how;
    const int net = choose_net(be->deviceContext.vkPhysicalDevice, &missing, &how);
    if (net < 0) {
        say(true, "no FSR 4.1.2 on this device: it lacks %s", missing.c_str());
        return FFX_API_RETURN_NO_PROVIDER;
    }
    void* mem = memCb && memCb->alloc ? memCb->alloc(memCb->pUserData, sizeof(Context)) : ::operator new(sizeof(Context), std::nothrow);
    if (!mem) return FFX_API_RETURN_ERROR_MEMORY;
    Context* c = new (mem) Context();
    if (memCb) c->alloc = *memCb;
    c->device = be->deviceContext.vkDevice, c->physical = be->deviceContext.vkPhysicalDevice;
    PFN_vkGetDeviceProcAddr gdpa = be->deviceContext.vkDeviceProcAddr;
    bool loaded = true;
#define LOAD(n) loaded = loaded && (c->vk.n = (PFN_vk##n)gdpa(c->device, "vk" #n)) != nullptr;
    DEVICE_FUNCS(LOAD)
#undef LOAD
    // (the physical device's memory types: through the loader, which the process has, since the caller made a device)
    void* loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!loader) loader = dlopen("libvulkan.so.1", RTLD_NOW);
    const auto gpdmp = loader ? (PFN_vkGetPhysicalDeviceMemoryProperties)dlsym(loader, "vkGetPhysicalDeviceMemoryProperties") : nullptr;
    if (!loaded || !gpdmp) {
        say(true, "the device's functions are not all there");
        destroy(c);
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    gpdmp(c->physical, &c->memory);
    c->message = up->fpMessage;
    c->flags = up->flags;
    c->max_rw = up->maxRenderSize.width, c->max_rh = up->maxRenderSize.height;
    c->width = up->maxUpscaleSize.width, c->height = up->maxUpscaleSize.height;
    c->net = &GEN_NETS[net];
    c->bucket = bucket_for(*c->net, c->width, c->height);
    const uint32_t W = c->width, H = c->height;
    bool ok = true;
    // AMD's resources (traced): the recurrent state, the history, the reprojected colour, the sharpening's input, the
    // exposure from auto exposure, a default exposure of 1, SPD's counter and its fifth mip, the debug view's image
    ok = ok && make_image(*c, c->recurrent, VK_FORMAT_R8G8B8A8_UNORM, W, H);
    ok = ok && make_image(*c, c->history, VK_FORMAT_R16G16B16A16_SFLOAT, W, H);
    ok = ok && make_image(*c, c->reprojected, VK_FORMAT_R16G16B16A16_SFLOAT, W, H);
    ok = ok && make_image(*c, c->rcas_tmp, VK_FORMAT_R16G16B16A16_SFLOAT, W, H);
    ok = ok && make_image(*c, c->autoexp, VK_FORMAT_R32_SFLOAT, 2, 1);
    ok = ok && make_image(*c, c->exposure1, VK_FORMAT_R32G32_SFLOAT, 1, 1);
    ok = ok && make_image(*c, c->spd_atomic, VK_FORMAT_R32_UINT, 1, 1);
    ok = ok && make_image(*c, c->spd_mip5, VK_FORMAT_R32_SFLOAT, div_up(c->max_rw, 16), div_up(c->max_rh, 16));
    if (c->flags & FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION) ok = ok && make_image(*c, c->debug_vis, VK_FORMAT_R16_UNORM, W, H);
    ok = ok && make_buffer(*c, c->scratch, c->bucket->scratch, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    const uint32_t init_size = c->net->init_size;
    ok = ok && make_buffer(*c, c->init, init_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    ok = ok && make_buffer(*c, c->init_ultra, init_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
    ok = ok && make_buffer(*c, c->init_staging, 2 * init_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    ok = ok && make_buffer(*c, c->staging, 512, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    ok = ok && make_buffer(*c, c->ubo, (VkDeviceSize)SLOT_BYTES * RING, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
    if (ok) {
        memcpy(c->init_staging.map, fsr4up_assets + c->net->init_offset[0], init_size);
        memcpy((uint8_t*)c->init_staging.map + init_size, fsr4up_assets + c->net->init_offset[1], init_size);
        const float one[2] = {1.0f, 1.0f};
        memset(c->staging.map, 0, 512);
        memcpy(c->staging.map, one, sizeof one);
        // (AMD's sampler: linear, clamped to the edge, every pass the same: traced)
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR, si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = 1000.0f;
        ok = c->vk.CreateSampler(c->device, &si, nullptr, &c->sampler) == VK_SUCCESS;
        const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, 8}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64},
                                              {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64}};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 40, pi.poolSizeCount = 5, pi.pPoolSizes = sizes;
        for (uint32_t s = 0; s < RING && ok; s++) ok = c->vk.CreateDescriptorPool(c->device, &pi, nullptr, &c->pools[s]) == VK_SUCCESS;
    }
    // the pipelines of the passes the create flags call for (AMD's DLL makes them at the first dispatch)
    if (ok) {
        const GenPreset& b = c->bucket->preset[0];
        ok = pipe(*c, b.prepass[prepass_key(c->flags, 0)]) && pipe(*c, b.postpass[postpass_key(c->flags, 0)]);
        for (int i = 0; i < c->net->passes && ok; i++) ok = pipe(*c, b.model[i]) != nullptr;
    }
    if (!ok) {
        say(true, "create failed");
        destroy(c);
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    say(false, "context: %ux%u from up to %ux%u, flags 0x%x, the %s network for up to %ux%u", W, H, c->max_rw, c->max_rh, c->flags, how.c_str(),
        c->bucket->max_w, c->bucket->max_h);
    *context = (ffxContext)c;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb) {
    (void)memCb;
    if (!context || !*context) return FFX_API_RETURN_ERROR_PARAMETER;
    destroy((Context*)*context);
    *context = nullptr;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    if (!desc) return FFX_API_RETURN_ERROR_PARAMETER;
    if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG) {
        std::lock_guard lock(g_message_mu);
        g_message = ((const ffxConfigureDescGlobalDebug*)desc)->fpMessage;
        return FFX_API_RETURN_OK;
    }
    if (!context || !*context) return FFX_API_RETURN_ERROR_PARAMETER;
    Context& c = *(Context*)*context;
    if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE) {
        const auto* kv = (const ffxConfigureDescUpscaleKeyValue*)desc;
        if (kv->key > FFX_API_CONFIGURE_UPSCALE_KEY_FMINDISOCCLUSIONACCUMULATION) return FFX_API_RETURN_ERROR_PARAMETER;
        if (kv->ptr) c.kv[kv->key] = *(const float*)kv->ptr;
        return FFX_API_RETURN_OK;
    }
    return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
}

ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* desc) {
    if (!desc) return FFX_API_RETURN_ERROR_PARAMETER;
    switch (desc->type) {
    case FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: {
        auto* q = (ffxQueryDescGetVersions*)desc;
        if (q->createDescType != FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) {
            if (q->outputCount) *q->outputCount = 0;
            return FFX_API_RETURN_OK;
        }
        // (AMD's DLL lists 4.1.2 only on a device that can run it, and its FSR 3.1.6 and 2.3.4 after it: those are not here.
        // Its INT8 networks are listed as 4.1.2 too, with the same id: traced)
        std::string missing;
        const int net = choose_net(q->device ? ((VkDeviceContext*)q->device)->vkPhysicalDevice : VK_NULL_HANDLE, &missing);
        if (net < 0) say(false, "no FSR 4.1.2 on this device: it lacks %s", missing.c_str());
        const uint64_t have = net >= 0 ? 1 : 0;
        if (!q->outputCount) return FFX_API_RETURN_ERROR_PARAMETER;
        if (!q->versionIds && !q->versionNames) {
            *q->outputCount = have;
            return FFX_API_RETURN_OK;
        }
        const uint64_t n = std::min(*q->outputCount, have);
        if (n) {
            if (q->versionIds) q->versionIds[0] = VERSION_ID;
            if (q->versionNames) q->versionNames[0] = VERSION_NAME;
        }
        *q->outputCount = n;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: {
        auto* q = (ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*)desc;
        if (q->pOutUpscaleRatio) *q->pOutUpscaleRatio = ratio_of(q->qualityMode);
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: {
        auto* q = (ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*)desc;
        const float ratio = ratio_of(q->qualityMode);
        if (!q->pOutRenderWidth || !q->pOutRenderHeight || q->qualityMode > FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE) {
            if (q->pOutRenderWidth) *q->pOutRenderWidth = 0;
            if (q->pOutRenderHeight) *q->pOutRenderHeight = 0;
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }
        *q->pOutRenderWidth = (uint32_t)((float)q->displayWidth / ratio);
        *q->pOutRenderHeight = (uint32_t)((float)q->displayHeight / ratio);
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: {
        auto* q = (ffxQueryDescUpscaleGetJitterPhaseCount*)desc;
        if (q->pOutPhaseCount) *q->pOutPhaseCount = (int32_t)(8.0f * powf((float)q->displayWidth / (float)q->renderWidth, 2.0f));
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: {
        auto* q = (ffxQueryDescUpscaleGetJitterOffset*)desc;
        if (q->phaseCount <= 0) return FFX_API_RETURN_ERROR_PARAMETER;
        if (q->pOutX) *q->pOutX = halton((q->index % q->phaseCount) + 1, 2) - 0.5f;
        if (q->pOutY) *q->pOutY = halton((q->index % q->phaseCount) + 1, 3) - 0.5f;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: {
        auto* q = (ffxQueryGetProviderVersion*)desc;
        q->versionId = VERSION_ID, q->versionName = VERSION_NAME;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE: {
        auto* q = (ffxQueryDescUpscaleGetGPUMemoryUsage*)desc;
        if (!context || !*context || !q->gpuMemoryUsageUpscaler) return FFX_API_RETURN_ERROR_PARAMETER;
        Context& c = *(Context*)*context;
        uint64_t total = 0;
        if (!amd_memory_usage(c.vk, c.device, *c.net, c.width, c.height, c.max_rw, c.max_rh, c.flags, &total)) return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        q->gpuMemoryUsageUpscaler->totalUsageInBytes = total, q->gpuMemoryUsageUpscaler->aliasableUsageInBytes = 0;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE_V2: {
        auto* q = (ffxQueryDescUpscaleGetGPUMemoryUsageV2*)desc;
        const auto* dc = (const VkDeviceContext*)q->device;
        if (!dc || !dc->vkDevice || !dc->vkDeviceProcAddr || !q->gpuMemoryUsageUpscaler) return FFX_API_RETURN_ERROR_PARAMETER;
        Vk vk;
        bool loaded = true;
#define LOAD(n) loaded = loaded && (vk.n = (PFN_vk##n)dc->vkDeviceProcAddr(dc->vkDevice, "vk" #n)) != nullptr;
        DEVICE_FUNCS(LOAD)
#undef LOAD
        uint64_t total = 0;
        // (for the network a context made now would run; on a device with none, FP8's, as before)
        const int net = choose_net(dc->vkPhysicalDevice, nullptr);
        if (!loaded || !amd_memory_usage(vk, dc->vkDevice, GEN_NETS[net < 0 ? NET_FP8 : net], q->maxUpscaleSize.width, q->maxUpscaleSize.height,
                                         q->maxRenderSize.width, q->maxRenderSize.height, q->flags, &total))
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        q->gpuMemoryUsageUpscaler->totalUsageInBytes = total, q->gpuMemoryUsageUpscaler->aliasableUsageInBytes = 0;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS: {
        auto* q = (ffxQueryDescUpscaleGetResourceRequirements*)desc;
        // (traced: colour, depth, motion and exposure required, nothing optional; auto exposure: see the suite)
        const uint32_t cf = context && *context ? ((Context*)*context)->flags : 0;
        q->required_resources = FFX_API_QUERY_RESOURCE_INPUT_COLOR | FFX_API_QUERY_RESOURCE_INPUT_DEPTH | FFX_API_QUERY_RESOURCE_INPUT_MV |
                                ((cf & FFX_UPSCALE_ENABLE_AUTO_EXPOSURE) ? 0 : FFX_API_QUERY_RESOURCE_INPUT_EXPOSURE);
        q->optional_resources = 0;
        return FFX_API_RETURN_OK;
    }
    default:
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

ffxReturnCode_t ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
    if (!context || !*context || !desc) return FFX_API_RETURN_ERROR_PARAMETER;
    Context& c = *(Context*)*context;
    if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE) return dispatch_upscale(c, *(const ffxDispatchDescUpscale*)desc);
    // (AMD's 4.1.2 takes a reactive mask request, records nothing for it and returns OK: traced. FSR 4 makes no use of one.)
    if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK) return FFX_API_RETURN_OK;
    return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
}

} // extern "C"
