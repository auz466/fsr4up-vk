// Runs an FSR API upscaling provider (AMD's Windows DLL under Wine, or the native port) on a synthetic scene with every
// device-level Vulkan call it makes logged, and its images dumped. Built twice: for Windows (mingw) and for Linux.
// (The Vulkan hook layer comes from fsr4fg-vk's harness, ~/Games/Hookshot/fsr4fg-vk/harness/trace.cpp.)
//
//   trace <provider library> <out dir> [W H RW RH frames dumpFrom]
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#endif

// AMD's headers mark the entry points dllexport; here they are only called through pointers.
#ifndef _WIN32
#define __declspec(x)
#endif
#include "ffx/api/include/ffx_api.h"
#include "ffx/api/include/ffx_api_types.h"
#include "ffx/api/include/vk/ffx_api_vk.h"
#include "ffx/upscalers/include/ffx_upscale.h"

// ---------------------------------------------------------------- Vulkan, loaded by hand
#define INSTANCE_FUNCS(X)                                                                                              \
    X(vkCreateInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)                                 \
    X(vkGetPhysicalDeviceFeatures2) X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice)                          \
    X(vkGetDeviceProcAddr) X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceQueueFamilyProperties)
#define DEVICE_FUNCS(X)                                                                                                \
    X(vkGetDeviceQueue) X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer)                     \
    X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkDeviceWaitIdle) X(vkResetCommandBuffer)              \
    X(vkCreateImage) X(vkGetImageMemoryRequirements) X(vkAllocateMemory) X(vkBindImageMemory) X(vkCreateBuffer)        \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkMapMemory) X(vkCmdCopyBufferToImage)                    \
    X(vkCmdCopyImageToBuffer) X(vkCmdPipelineBarrier) X(vkCmdCopyBuffer) X(vkDestroyBuffer) X(vkFreeMemory)            \
    X(vkCreateImageView) X(vkCreateSampler) X(vkCreateShaderModule) X(vkCreateDescriptorSetLayout)                     \
    X(vkCreatePipelineLayout) X(vkCreateComputePipelines) X(vkCreateDescriptorPool) X(vkAllocateDescriptorSets)        \
    X(vkUpdateDescriptorSets) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdDispatch)                         \
    X(vkCmdDispatchIndirect) X(vkCmdCopyImage) X(vkCmdClearColorImage) X(vkCmdFillBuffer) X(vkCmdPushConstants)        \
    X(vkCreateBufferView) X(vkUnmapMemory) X(vkFlushMappedMemoryRanges) X(vkDestroyImage) X(vkDestroyImageView)        \
    X(vkDestroySampler) X(vkDestroyShaderModule) X(vkDestroyDescriptorSetLayout) X(vkDestroyPipelineLayout)            \
    X(vkDestroyPipeline) X(vkDestroyDescriptorPool) X(vkCreateQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp)     \
    X(vkGetQueryPoolResults)
#define DECL(n) static PFN_##n n;
INSTANCE_FUNCS(DECL)
DEVICE_FUNCS(DECL)
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;

static FILE* g_log;
static std::string g_out;
static bool g_quiet = false;   // (TRACE_TIMING: no log, so the CPU times are the provider's and not the harness's)
static void logf(const char* fmt, ...) {
    if (g_quiet) return;
    va_list a;
    va_start(a, fmt);
    vfprintf(g_log, fmt, a);
    va_end(a);
    fputc('\n', g_log);
    fflush(g_log);
}
[[noreturn]] static void die(const char* fmt, ...) {
    va_list a;
    va_start(a, fmt);
    vfprintf(stderr, fmt, a);
    va_end(a);
    fputc('\n', stderr);
    exit(1);
}

static void* load_lib(const char* path) {
#ifdef _WIN32
    return (void*)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}
static void* sym(void* lib, const char* name) {
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

// ---------------------------------------------------------------- what the provider made, by handle
struct ImageInfo {
    int id;
    VkFormat format;
    VkExtent3D extent;
    uint32_t mips;
    VkImageUsageFlags usage;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    std::string name;
    bool ours = false;
};
struct BufferInfo {
    int id;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    uint64_t memory = 0;
    VkDeviceSize offset = 0;
    std::string name;
};
struct ViewInfo {
    int id;
    uint64_t image;
    uint32_t mip;
    VkImageViewCreateInfo ci{};
    VkImageView as_rg32 = VK_NULL_HANDLE;
};
struct Write {
    VkDescriptorType type;
    uint64_t a = 0;   // view or buffer
    uint64_t b = 0;   // sampler
    VkDeviceSize offset = 0, range = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};
struct PipeInfo {
    int id;
    int module;
    std::string entry;
    uint64_t layout;
};
static std::map<uint64_t, ImageInfo> g_images;
static std::map<uint64_t, BufferInfo> g_buffers;
static std::map<uint64_t, ViewInfo> g_views;
static std::map<uint64_t, void*> g_mapped;
static std::map<uint64_t, int> g_modules, g_samplers, g_setlayouts, g_pipelayouts, g_sets, g_memories;
static std::map<uint64_t, PipeInfo> g_pipes;
static std::map<uint64_t, std::map<uint32_t, Write>> g_setwrites;   // set -> binding*1000+element -> write
static std::map<uint64_t, std::string> g_names;
static int g_next = 1, g_dispatch = 0, g_upload = 0;
static bool g_tracing = false;
static VkDevice g_device;

template <class T> static uint64_t H(T h) { return (uint64_t)h; }
static std::string hex(const void* p, size_t n) {
    std::string s;
    char b[4];
    for (size_t i = 0; i < n; i++) snprintf(b, sizeof b, "%02x", ((const uint8_t*)p)[i]), s += b;
    return s;
}
static void dump_file(const std::string& name, const void* p, size_t n) {
    if (g_quiet) return;
    FILE* f = fopen((g_out + "/" + name).c_str(), "wb");
    if (!f) die("cannot write %s", name.c_str());
    fwrite(p, 1, n, f);
    fclose(f);
}
static const void* buffer_bytes(uint64_t buffer, VkDeviceSize offset) {
    auto b = g_buffers.find(buffer);
    if (b == g_buffers.end()) return nullptr;
    auto m = g_mapped.find(b->second.memory);
    if (m == g_mapped.end() || !m->second) return nullptr;
    return (const uint8_t*)m->second + b->second.offset + offset;
}

// ---------------------------------------------------------------- the hooks
static bool probe_dispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z);   // (TRACE_PROBE: below)
namespace hook {
// (TRACE_FAIL_AT=n: the provider's n-th object creation fails with VK_ERROR_OUT_OF_DEVICE_MEMORY, to see that it
// gives up cleanly; g_live: the provider's objects not yet destroyed, logged after its context is destroyed)
static int g_creations = 0;
static std::map<uint64_t, int> g_live;   // handle -> 1
static bool fail_now() {
    static const int at = getenv("TRACE_FAIL_AT") ? atoi(getenv("TRACE_FAIL_AT")) : 0;
    if (++g_creations != at) return false;
    logf("FAILING creation %d on purpose", g_creations);
    return true;
}
template <typename T> static void live(const T& h) { g_live[(uint64_t)h] = 1; }
template <typename T> static void dead(const T& h) { g_live.erase((uint64_t)h); }
VkResult VKAPI_CALL CreateBuffer(VkDevice d, const VkBufferCreateInfo* ci, const VkAllocationCallbacks* a, VkBuffer* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VkBufferCreateInfo c = *ci;
    c.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkResult r = vkCreateBuffer(d, &c, a, out);
    if (r == VK_SUCCESS) live(*out);
    if (r == VK_SUCCESS) {
        g_buffers[H(*out)] = {g_next, ci->size, ci->usage};
        logf("CreateBuffer buf%d size=%llu usage=0x%x", g_next++, (unsigned long long)ci->size, ci->usage);
    } else logf("CreateBuffer FAILED %d size=%llu", r, (unsigned long long)ci->size);
    return r;
}
VkResult VKAPI_CALL CreateImage(VkDevice d, const VkImageCreateInfo* ci, const VkAllocationCallbacks* a, VkImage* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VkImageCreateInfo c = *ci;
    c.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (c.format == VK_FORMAT_R64_UINT) c.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;   // (sampled as R32G32_UINT: tools/spv_fix.py)
    VkResult r = vkCreateImage(d, &c, a, out);
    if (r == VK_SUCCESS) live(*out);
    if (r == VK_SUCCESS) {
        g_images[H(*out)] = {g_next, ci->format, ci->extent, ci->mipLevels, ci->usage, ci->initialLayout};
        logf("CreateImage img%d type=%d format=%d extent=%ux%ux%u mips=%u layers=%u usage=0x%x flags=0x%x", g_next++,
             ci->imageType, ci->format, ci->extent.width, ci->extent.height, ci->extent.depth, ci->mipLevels,
             ci->arrayLayers, ci->usage, ci->flags);
    } else logf("CreateImage FAILED %d format=%d", r, ci->format);
    return r;
}
VkResult VKAPI_CALL CreateImageView(VkDevice d, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks* a, VkImageView* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VkResult r = vkCreateImageView(d, ci, a, out);
    if (r == VK_SUCCESS) live(*out);
    if (r == VK_SUCCESS) {
        g_views[H(*out)] = {g_next, H(ci->image), ci->subresourceRange.baseMipLevel, *ci};
        auto im = g_images.find(H(ci->image));
        logf("CreateImageView view%d img%d format=%d viewType=%d mip=%u+%u aspect=0x%x", g_next++,
             im == g_images.end() ? -1 : im->second.id, ci->format, ci->viewType, ci->subresourceRange.baseMipLevel,
             ci->subresourceRange.levelCount, ci->subresourceRange.aspectMask);
    }
    return r;
}
VkResult VKAPI_CALL CreateSampler(VkDevice d, const VkSamplerCreateInfo* ci, const VkAllocationCallbacks* a, VkSampler* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VkResult r = vkCreateSampler(d, ci, a, out);
    if (r == VK_SUCCESS) live(*out);
    if (r == VK_SUCCESS && !g_samplers.count(H(*out))) {
        g_samplers[H(*out)] = g_next;
        logf("CreateSampler smp%d mag=%d min=%d mip=%d addr=%d,%d,%d maxLod=%g", g_next++, ci->magFilter, ci->minFilter,
             ci->mipmapMode, ci->addressModeU, ci->addressModeV, ci->addressModeW, ci->maxLod);
    }
    return r;
}
VkResult VKAPI_CALL CreateShaderModule(VkDevice d, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks* a, VkShaderModule* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    char name[64];
    snprintf(name, sizeof name, "mod_%04d.spv", g_next);
    dump_file(name, ci->pCode, ci->codeSize);
    // (a rewritten module in place of AMD's, by the FNV-1a of AMD's: tools/spv_fix.py, for what Mesa does not take)
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < ci->codeSize; i++) h = (h ^ ((const uint8_t*)ci->pCode)[i]) * 0x100000001b3ull;
    VkShaderModuleCreateInfo c = *ci;
    std::vector<uint32_t> patched;
    if (const char* dir = getenv("TRACE_PATCHES")) {
        char path[600];
        snprintf(path, sizeof path, "%s/%016llx.spv", dir, (unsigned long long)h);
        if (FILE* f = fopen(path, "rb")) {
            fseek(f, 0, SEEK_END);
            patched.resize(ftell(f) / 4);
            fseek(f, 0, SEEK_SET);
            fread(patched.data(), 4, patched.size(), f);
            fclose(f);
            c.pCode = patched.data(), c.codeSize = patched.size() * 4;
        }
    }
    VkResult r = vkCreateShaderModule(d, &c, a, out);
    if (r == VK_SUCCESS) live(*out);
    logf("CreateShaderModule mod%d size=%llu fnv=%016llx%s result=%d", g_next, (unsigned long long)ci->codeSize, (unsigned long long)h,
         patched.empty() ? "" : " PATCHED", r);
    if (r == VK_SUCCESS) g_modules[H(*out)] = g_next;
    g_next++;
    return r;
}
VkResult VKAPI_CALL CreateDescriptorSetLayout(VkDevice d, const VkDescriptorSetLayoutCreateInfo* ci, const VkAllocationCallbacks* a, VkDescriptorSetLayout* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VkResult r = vkCreateDescriptorSetLayout(d, ci, a, out);
    if (r == VK_SUCCESS) live(*out);
    std::string s;
    char b[64];
    for (uint32_t i = 0; i < ci->bindingCount; i++) {
        snprintf(b, sizeof b, " %u:t%d:n%u", ci->pBindings[i].binding, ci->pBindings[i].descriptorType, ci->pBindings[i].descriptorCount);
        s += b;
    }
    if (r == VK_SUCCESS) g_setlayouts[H(*out)] = g_next;
    logf("CreateDescriptorSetLayout dsl%d flags=0x%x result=%d bindings:%s", g_next++, ci->flags, r, s.c_str());
    return r;
}
VkResult VKAPI_CALL CreatePipelineLayout(VkDevice d, const VkPipelineLayoutCreateInfo* ci, const VkAllocationCallbacks* a, VkPipelineLayout* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VkResult r = vkCreatePipelineLayout(d, ci, a, out);
    if (r == VK_SUCCESS) live(*out);
    std::string s;
    char b[64];
    for (uint32_t i = 0; i < ci->setLayoutCount; i++) snprintf(b, sizeof b, " dsl%d", g_setlayouts[H(ci->pSetLayouts[i])]), s += b;
    for (uint32_t i = 0; i < ci->pushConstantRangeCount; i++)
        snprintf(b, sizeof b, " push(%u+%u)", ci->pPushConstantRanges[i].offset, ci->pPushConstantRanges[i].size), s += b;
    if (r == VK_SUCCESS) g_pipelayouts[H(*out)] = g_next;
    logf("CreatePipelineLayout pl%d result=%d:%s", g_next++, r, s.c_str());
    return r;
}
VkResult VKAPI_CALL CreateComputePipelines(VkDevice d, VkPipelineCache cache, uint32_t n, const VkComputePipelineCreateInfo* ci, const VkAllocationCallbacks* a, VkPipeline* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VkResult r = vkCreateComputePipelines(d, cache, n, ci, a, out);
    if (r == VK_SUCCESS) for (uint32_t i = 0; i < n; i++) live(out[i]);
    for (uint32_t i = 0; i < n; i++) {
        std::string chain;
        char b[64];
        for (auto* p = (const VkBaseInStructure*)ci[i].stage.pNext; p; p = p->pNext) {
            snprintf(b, sizeof b, " stageNext=%d", p->sType), chain += b;
            if (p->sType == VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO)
                snprintf(b, sizeof b, "(subgroup=%u)", ((const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo*)p)->requiredSubgroupSize), chain += b;
        }
        if (ci[i].stage.pSpecializationInfo) chain += " SPECIALIZED";
        int mod = g_modules[H(ci[i].stage.module)];
        if (r == VK_SUCCESS) g_pipes[H(out[i])] = {g_next, mod, ci[i].stage.pName, H(ci[i].layout)};
        logf("CreateComputePipeline pipe%d mod%d entry=%s pl%d flags=0x%x stageFlags=0x%x result=%d%s", g_next++, mod,
             ci[i].stage.pName, g_pipelayouts[H(ci[i].layout)], ci[i].flags, ci[i].stage.flags, r, chain.c_str());
    }
    return r;
}
VkResult VKAPI_CALL AllocateMemory(VkDevice d, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks* a, VkDeviceMemory* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VkResult r = vkAllocateMemory(d, ai, a, out);
    if (r == VK_SUCCESS) live(*out);
    if (r == VK_SUCCESS) g_memories[H(*out)] = g_next;
    logf("AllocateMemory mem%d size=%llu type=%u result=%d", g_next++, (unsigned long long)ai->allocationSize, ai->memoryTypeIndex, r);
    return r;
}
VkResult VKAPI_CALL MapMemory(VkDevice d, VkDeviceMemory m, VkDeviceSize off, VkDeviceSize size, VkMemoryMapFlags f, void** out) {
    VkResult r = vkMapMemory(d, m, off, size, f, out);
    if (r == VK_SUCCESS) g_mapped[H(m)] = (uint8_t*)*out - off;
    return r;
}
void VKAPI_CALL UnmapMemory(VkDevice d, VkDeviceMemory m) {
    // (kept mapped for the trace: an upload's bytes are read when its copy is recorded, which may be after the unmap)
    (void)d, (void)m;
}
VkResult VKAPI_CALL BindBufferMemory(VkDevice d, VkBuffer b, VkDeviceMemory m, VkDeviceSize off) {
    auto it = g_buffers.find(H(b));
    if (it != g_buffers.end()) it->second.memory = H(m), it->second.offset = off;
    return vkBindBufferMemory(d, b, m, off);
}
VkResult VKAPI_CALL AllocateDescriptorSets(VkDevice d, const VkDescriptorSetAllocateInfo* ai, VkDescriptorSet* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VkResult r = vkAllocateDescriptorSets(d, ai, out);
    if (r == VK_SUCCESS)
        for (uint32_t i = 0; i < ai->descriptorSetCount; i++) {
            g_sets[H(out[i])] = g_next;
            g_setwrites[H(out[i])].clear();
            logf("AllocateDescriptorSet set%d dsl%d", g_next++, g_setlayouts[H(ai->pSetLayouts[i])]);
        }
    else logf("AllocateDescriptorSets FAILED %d", r);
    return r;
}
void VKAPI_CALL UpdateDescriptorSets(VkDevice d, uint32_t n, const VkWriteDescriptorSet* w, uint32_t nc, const VkCopyDescriptorSet* c) {
    for (uint32_t i = 0; i < n; i++)
        for (uint32_t k = 0; k < w[i].descriptorCount; k++) {
            Write x{w[i].descriptorType};
            switch (w[i].descriptorType) {
            case VK_DESCRIPTOR_TYPE_SAMPLER: x.b = H(w[i].pImageInfo[k].sampler); break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                x.a = H(w[i].pImageInfo[k].imageView), x.b = H(w[i].pImageInfo[k].sampler), x.layout = w[i].pImageInfo[k].imageLayout;
                break;
            default:
                if (w[i].pBufferInfo) x.a = H(w[i].pBufferInfo[k].buffer), x.offset = w[i].pBufferInfo[k].offset, x.range = w[i].pBufferInfo[k].range;
            }
            g_setwrites[H(w[i].dstSet)][w[i].dstBinding * 1000 + w[i].dstArrayElement + k] = x;
        }
    if (nc) logf("UpdateDescriptorSets with %u COPIES (not traced)", nc);
    // (a sampled 64-bit image is bound as its two 32-bit halves: tools/spv_fix.py)
    std::vector<VkWriteDescriptorSet> ws(w, w + n);
    std::vector<std::vector<VkDescriptorImageInfo>> swapped;
    swapped.reserve(n);
    for (auto& x : ws) {
        if (x.descriptorType != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE) continue;
        bool any = false;
        std::vector<VkDescriptorImageInfo> infos(x.pImageInfo, x.pImageInfo + x.descriptorCount);
        for (auto& info : infos) {
            auto v = g_views.find(H(info.imageView));
            if (v == g_views.end()) continue;
            auto im = g_images.find(v->second.image);
            if (im == g_images.end() || im->second.format != VK_FORMAT_R64_UINT) continue;
            if (!v->second.as_rg32) {
                VkImageViewCreateInfo vc = v->second.ci;
                vc.pNext = nullptr, vc.format = VK_FORMAT_R32G32_UINT;
                if (vkCreateImageView(d, &vc, nullptr, &v->second.as_rg32)) logf("R32G32 view FAILED");
            }
            info.imageView = v->second.as_rg32, any = true;
        }
        if (any) swapped.push_back(std::move(infos)), x.pImageInfo = swapped.back().data();
    }
    vkUpdateDescriptorSets(d, n, ws.data(), nc, c);
}

struct Bound {
    uint64_t pipeline = 0;
    std::vector<uint64_t> sets;
    std::vector<uint32_t> dynamic;
};
static std::map<uint64_t, Bound> g_bound;   // by command buffer

void VKAPI_CALL CmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipeline p) {
    g_bound[H(cb)].pipeline = H(p);
    vkCmdBindPipeline(cb, bp, p);
}
void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n, const VkDescriptorSet* sets, uint32_t nd, const uint32_t* dyn) {
    Bound& b = g_bound[H(cb)];
    if (b.sets.size() < first + n) b.sets.resize(first + n);
    for (uint32_t i = 0; i < n; i++) b.sets[first + i] = H(sets[i]);
    b.dynamic.assign(dyn, dyn + nd);
    vkCmdBindDescriptorSets(cb, bp, l, first, n, sets, nd, dyn);
}
static void describe_bound(VkCommandBuffer cb, const char* what, uint32_t x, uint32_t y, uint32_t z) {
    Bound& b = g_bound[H(cb)];
    auto p = g_pipes.find(b.pipeline);
    logf("DISPATCH %d %s %ux%ux%u pipe%d mod%d entry=%s", g_dispatch, what, x, y, z, p == g_pipes.end() ? -1 : p->second.id,
         p == g_pipes.end() ? -1 : p->second.module, p == g_pipes.end() ? "?" : p->second.entry.c_str());
    size_t dyn = 0;
    for (size_t s = 0; s < b.sets.size(); s++) {
        for (auto& [key, w] : g_setwrites[b.sets[s]]) {
            const uint32_t binding = key / 1000, element = key % 1000;
            switch (w.type) {
            case VK_DESCRIPTOR_TYPE_SAMPLER: logf("  s%zu b%u[%u] sampler smp%d", s, binding, element, g_samplers[w.b]); break;
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
                auto v = g_views.find(w.a);
                if (v == g_views.end()) {
                    logf("  s%zu b%u[%u] %s view? layout=%d", s, binding, element, w.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? "storage" : "sampled", w.layout);
                    break;
                }
                auto im = g_images.find(v->second.image);
                logf("  s%zu b%u[%u] %s img%d mip%u layout=%d %s", s, binding, element,
                     w.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? "storage" : "sampled", im == g_images.end() ? -1 : im->second.id,
                     v->second.mip, w.layout, im == g_images.end() ? "" : im->second.name.c_str());
                break;
            }
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: {
                VkDeviceSize off = w.offset;
                if (w.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC && dyn < b.dynamic.size()) off += b.dynamic[dyn++];
                auto bi = g_buffers.find(w.a);
                const void* bytes = buffer_bytes(w.a, off);
                const VkDeviceSize range = w.range == VK_WHOLE_SIZE ? (bi == g_buffers.end() ? 0 : bi->second.size - off) : w.range;
                logf("  s%zu b%u[%u] ubo buf%d off=%llu range=%llu data=%s", s, binding, element, bi == g_buffers.end() ? -1 : bi->second.id,
                     (unsigned long long)off, (unsigned long long)range, bytes ? hex(bytes, (size_t)std::min<VkDeviceSize>(range, 8192)).c_str() : "(unmapped)");
                break;
            }
            default: {
                auto bi = g_buffers.find(w.a);
                logf("  s%zu b%u[%u] buffer(t%d) buf%d off=%llu range=%lld %s", s, binding, element, w.type, bi == g_buffers.end() ? -1 : bi->second.id,
                     (unsigned long long)w.offset, (long long)w.range, bi == g_buffers.end() ? "" : bi->second.name.c_str());
            }
            }
        }
    }
    g_dispatch++;
}
static bool g_stopped = false;
void VKAPI_CALL CmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    // TRACE_STOP_ENTRY=<entry>: every dispatch after that pass is left out (to compare the scratch buffer at that point)
    if (g_stopped) return;
    describe_bound(cb, "direct", x, y, z);
    probe_dispatch(cb, x, y, z);
    vkCmdDispatch(cb, x, y, z);
    static const char* stop = getenv("TRACE_STOP_ENTRY");
    auto p = g_pipes.find(g_bound[H(cb)].pipeline);
    if (stop && p != g_pipes.end() && p->second.entry == stop) g_stopped = true;
}
void VKAPI_CALL CmdDispatchIndirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off) {
    describe_bound(cb, "INDIRECT", 0, 0, 0);
    vkCmdDispatchIndirect(cb, b, off);
}
void VKAPI_CALL CmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off, uint32_t size, const void* v) {
    logf("PushConstants off=%u size=%u data=%s", off, size, hex(v, size).c_str());
    vkCmdPushConstants(cb, l, st, off, size, v);
}
void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t n, const VkBufferCopy* r) {
    for (uint32_t i = 0; i < n; i++) {
        const void* bytes = buffer_bytes(H(src), r[i].srcOffset);
        char name[64] = "-";
        if (bytes) snprintf(name, sizeof name, "upload_%04d.bin", g_upload++), dump_file(name, bytes, (size_t)r[i].size);
        logf("CopyBuffer buf%d+%llu -> buf%d+%llu size=%llu file=%s", g_buffers[H(src)].id, (unsigned long long)r[i].srcOffset,
             g_buffers[H(dst)].id, (unsigned long long)r[i].dstOffset, (unsigned long long)r[i].size, name);
    }
    vkCmdCopyBuffer(cb, src, dst, n, r);
}
void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer src, VkImage dst, VkImageLayout l, uint32_t n, const VkBufferImageCopy* r) {
    auto im = g_images.find(H(dst));
    for (uint32_t i = 0; i < n; i++) {
        const void* bytes = buffer_bytes(H(src), r[i].bufferOffset);
        char name[64] = "-";
        // (the size of the copy is not known without the format's texel size: the rest of the staging buffer is written)
        auto bi = g_buffers.find(H(src));
        if (bytes && bi != g_buffers.end())
            snprintf(name, sizeof name, "upload_%04d.bin", g_upload++), dump_file(name, bytes, (size_t)(bi->second.size - r[i].bufferOffset));
        logf("CopyBufferToImage buf%d+%llu -> img%d mip%u extent=%ux%ux%u file=%s", g_buffers[H(src)].id, (unsigned long long)r[i].bufferOffset,
             im == g_images.end() ? -1 : im->second.id, r[i].imageSubresource.mipLevel, r[i].imageExtent.width, r[i].imageExtent.height,
             r[i].imageExtent.depth, name);
    }
    vkCmdCopyBufferToImage(cb, src, dst, l, n, r);
}
void VKAPI_CALL CmdCopyImage(VkCommandBuffer cb, VkImage src, VkImageLayout sl, VkImage dst, VkImageLayout dl, uint32_t n, const VkImageCopy* r) {
    if (g_stopped) return;
    for (uint32_t i = 0; i < n; i++)
        logf("CopyImage img%d mip%u (%d,%d) -> img%d mip%u (%d,%d) extent=%ux%u", g_images[H(src)].id, r[i].srcSubresource.mipLevel, r[i].srcOffset.x,
             r[i].srcOffset.y, g_images[H(dst)].id, r[i].dstSubresource.mipLevel, r[i].dstOffset.x, r[i].dstOffset.y, r[i].extent.width, r[i].extent.height);
    vkCmdCopyImage(cb, src, sl, dst, dl, n, r);
}
void VKAPI_CALL CmdClearColorImage(VkCommandBuffer cb, VkImage im, VkImageLayout l, const VkClearColorValue* c, uint32_t n, const VkImageSubresourceRange* r) {
    logf("ClearColorImage img%d color=%g,%g,%g,%g (u32 %u,%u,%u,%u) mips=%u+%u", g_images[H(im)].id, c->float32[0], c->float32[1], c->float32[2],
         c->float32[3], c->uint32[0], c->uint32[1], c->uint32[2], c->uint32[3], n ? r[0].baseMipLevel : 0, n ? r[0].levelCount : 0);
    vkCmdClearColorImage(cb, im, l, c, n, r);
}
void VKAPI_CALL CmdFillBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off, VkDeviceSize size, uint32_t data) {
    logf("FillBuffer buf%d off=%llu size=%lld data=0x%x", g_buffers[H(b)].id, (unsigned long long)off, (long long)size, data);
    vkCmdFillBuffer(cb, b, off, size, data);
}
void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer cb, VkPipelineStageFlags ss, VkPipelineStageFlags ds, VkDependencyFlags df, uint32_t nm, const VkMemoryBarrier* m, uint32_t nb, const VkBufferMemoryBarrier* b, uint32_t ni, const VkImageMemoryBarrier* im) {
    // (every barrier, for 3.4: which dependencies the provider asks for between its passes)
    logf("BARRIER stages=0x%x->0x%x memory=%u buffers=%u images=%u", ss, ds, nm, nb, ni);
    for (uint32_t i = 0; i < nm; i++) logf("  memory access=0x%x->0x%x", m[i].srcAccessMask, m[i].dstAccessMask);
    for (uint32_t i = 0; i < nb; i++) {
        auto it = g_buffers.find(H(b[i].buffer));
        logf("  buffer buf%d+%llu size=%lld access=0x%x->0x%x", it == g_buffers.end() ? -1 : it->second.id, (unsigned long long)b[i].offset,
             (long long)b[i].size, b[i].srcAccessMask, b[i].dstAccessMask);
    }
    for (uint32_t i = 0; i < ni; i++) {
        auto it = g_images.find(H(im[i].image));
        logf("  image img%d layout=%d->%d access=0x%x->0x%x", it == g_images.end() ? -1 : it->second.id, im[i].oldLayout, im[i].newLayout,
             im[i].srcAccessMask, im[i].dstAccessMask);
    }
    for (uint32_t i = 0; i < ni; i++) {
        auto it = g_images.find(H(im[i].image));
        if (it != g_images.end()) it->second.layout = im[i].newLayout;
    }
    vkCmdPipelineBarrier(cb, ss, ds, df, nm, m, nb, b, ni, im);
}
VkResult VKAPI_CALL SetDebugUtilsObjectNameEXT(VkDevice, const VkDebugUtilsObjectNameInfoEXT* info) {
    const char* n = info->pObjectName ? info->pObjectName : "";
    g_names[info->objectHandle] = n;
    if (auto i = g_images.find(info->objectHandle); i != g_images.end()) i->second.name = n, logf("Name img%d = %s", i->second.id, n);
    else if (auto b = g_buffers.find(info->objectHandle); b != g_buffers.end()) b->second.name = n, logf("Name buf%d = %s", b->second.id, n);
    else if (auto p = g_pipes.find(info->objectHandle); p != g_pipes.end()) logf("Name pipe%d = %s", p->second.id, n);
    else logf("Name type%d %llx = %s", info->objectType, (unsigned long long)info->objectHandle, n);
    return VK_SUCCESS;
}
void VKAPI_CALL CmdBeginDebugUtilsLabelEXT(VkCommandBuffer, const VkDebugUtilsLabelEXT* l) { logf("LABEL begin %s", l->pLabelName); }
void VKAPI_CALL CmdEndDebugUtilsLabelEXT(VkCommandBuffer) { logf("LABEL end"); }

void VKAPI_CALL DestroyBuffer(VkDevice d, VkBuffer h, const VkAllocationCallbacks* a) { dead(h), g_buffers.erase(H(h)), vkDestroyBuffer(d, h, a); }   // (gone: not dumped)
void VKAPI_CALL DestroyImage(VkDevice d, VkImage h, const VkAllocationCallbacks* a) { dead(h), vkDestroyImage(d, h, a); }
void VKAPI_CALL DestroyImageView(VkDevice d, VkImageView h, const VkAllocationCallbacks* a) { dead(h), vkDestroyImageView(d, h, a); }
void VKAPI_CALL DestroySampler(VkDevice d, VkSampler h, const VkAllocationCallbacks* a) { dead(h), vkDestroySampler(d, h, a); }
void VKAPI_CALL DestroyShaderModule(VkDevice d, VkShaderModule h, const VkAllocationCallbacks* a) { dead(h), vkDestroyShaderModule(d, h, a); }
void VKAPI_CALL DestroyDescriptorSetLayout(VkDevice d, VkDescriptorSetLayout h, const VkAllocationCallbacks* a) { dead(h), vkDestroyDescriptorSetLayout(d, h, a); }
void VKAPI_CALL DestroyPipelineLayout(VkDevice d, VkPipelineLayout h, const VkAllocationCallbacks* a) { dead(h), vkDestroyPipelineLayout(d, h, a); }
void VKAPI_CALL DestroyPipeline(VkDevice d, VkPipeline h, const VkAllocationCallbacks* a) { dead(h), vkDestroyPipeline(d, h, a); }
void VKAPI_CALL FreeMemory(VkDevice d, VkDeviceMemory h, const VkAllocationCallbacks* a) { dead(h), vkFreeMemory(d, h, a); }
VkResult VKAPI_CALL CreateDescriptorPool(VkDevice d, const VkDescriptorPoolCreateInfo* ci, const VkAllocationCallbacks* a, VkDescriptorPool* out) {
    if (fail_now()) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    VkResult r = vkCreateDescriptorPool(d, ci, a, out);
    if (r == VK_SUCCESS) live(*out);
    return r;
}
void VKAPI_CALL DestroyDescriptorPool(VkDevice d, VkDescriptorPool h, const VkAllocationCallbacks* a) { dead(h), vkDestroyDescriptorPool(d, h, a); }

PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice d, const char* name) {
    // (timing: the provider gets the driver's own functions, but for the shader module hook that swaps in the rewritten
    // shaders AMD's DLL needs, so the CPU times are the provider's and the GPU runs the same work as when traced)
    if (g_quiet && strcmp(name, "vkCreateShaderModule") != 0) return vkGetDeviceProcAddr(d, name);
#define HOOK(n) \
    if (!strcmp(name, "vk" #n)) return (PFN_vkVoidFunction)n;
    HOOK(CreateBuffer) HOOK(CreateImage) HOOK(CreateImageView) HOOK(CreateSampler) HOOK(CreateShaderModule)
    HOOK(CreateDescriptorSetLayout) HOOK(CreatePipelineLayout) HOOK(CreateComputePipelines) HOOK(AllocateMemory)
    HOOK(MapMemory) HOOK(UnmapMemory) HOOK(BindBufferMemory) HOOK(AllocateDescriptorSets) HOOK(UpdateDescriptorSets)
    HOOK(CmdBindPipeline) HOOK(CmdBindDescriptorSets) HOOK(CmdDispatch) HOOK(CmdDispatchIndirect) HOOK(CmdPushConstants)
    HOOK(CmdCopyBuffer) HOOK(CmdCopyBufferToImage) HOOK(CmdCopyImage) HOOK(CmdClearColorImage) HOOK(CmdFillBuffer)
    HOOK(CmdPipelineBarrier) HOOK(SetDebugUtilsObjectNameEXT) HOOK(CmdBeginDebugUtilsLabelEXT) HOOK(CmdEndDebugUtilsLabelEXT)
    HOOK(DestroyBuffer) HOOK(DestroyImage) HOOK(DestroyImageView) HOOK(DestroySampler) HOOK(DestroyShaderModule)
    HOOK(DestroyDescriptorSetLayout) HOOK(DestroyPipelineLayout) HOOK(DestroyPipeline) HOOK(FreeMemory)
    HOOK(CreateDescriptorPool) HOOK(DestroyDescriptorPool)
#undef HOOK
    PFN_vkVoidFunction f = vkGetDeviceProcAddr(d, name);
    logf("GetDeviceProcAddr %s -> %s (not traced)", name, f ? "real" : "NULL");
    return f;
}
} // namespace hook

// ---------------------------------------------------------------- our own images and the scene
static VkPhysicalDevice g_phys;
static VkQueue g_queue;
static VkCommandBuffer g_cmd;
static VkPhysicalDeviceMemoryProperties g_memprops;

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_memprops.memoryTypeCount; i++)
        if ((bits >> i & 1) && (g_memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    die("no memory type");
}
struct Staging {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* map;
    VkDeviceSize size;
};
static Staging staging(VkDeviceSize size) {
    Staging s{};
    s.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size, ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(g_device, &ci, nullptr, &s.buffer)) die("staging buffer");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_device, s.buffer, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(g_device, &ai, nullptr, &s.memory)) die("staging memory");
    vkBindBufferMemory(g_device, s.buffer, s.memory, 0);
    vkMapMemory(g_device, s.memory, 0, VK_WHOLE_SIZE, 0, &s.map);
    return s;
}
static void free_staging(Staging& s) {
    vkDestroyBuffer(g_device, s.buffer, nullptr);
    vkFreeMemory(g_device, s.memory, nullptr);
}
static VkImage our_image(VkFormat format, uint32_t w, uint32_t h, const char* name) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = format, ci.extent = {w, h, 1}, ci.mipLevels = 1, ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImage im;
    if (vkCreateImage(g_device, &ci, nullptr, &im)) die("image %s", name);
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(g_device, im, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size, ai.memoryTypeIndex = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory mem;
    if (vkAllocateMemory(g_device, &ai, nullptr, &mem)) die("image memory");
    vkBindImageMemory(g_device, im, mem, 0);
    ImageInfo info{g_next, format, ci.extent, 1, ci.usage, VK_IMAGE_LAYOUT_UNDEFINED, name, true};
    g_images[H(im)] = info;
    logf("OurImage img%d %s format=%d %ux%u", g_next++, name, format, w, h);
    return im;
}
static void begin() {
    vkResetCommandBuffer(g_cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(g_cmd, &bi);
}
static void submit() {
    vkEndCommandBuffer(g_cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &g_cmd;
    if (VkResult r = vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE)) die("submit %d", r);
    if (VkResult r = vkQueueWaitIdle(g_queue)) die("wait idle %d (device lost?)", r);
}

// ---------------------------------------------------------------- TRACE_PROBE: which network passes depend on which
// TRACE_PROBE=<frame> TRACE_PROBE_ZERO=<entries>: in that frame, at each pass named in TRACE_PROBE_ZERO (the passes that
// only write zeros to the scratch buffer, the tensors' borders: tools/probe_borders.py lists them) the work recorded so
// far is run, and that pass and the last pass before it that is not one of them (c) are run alone on chosen scratch
// contents, then the real contents are put back and recording goes on. Logged, one line a pair (PROBE ...):
//   wz     words the zero pass writes (scratch all ones before: what becomes zero)
//   wc     words c writes (scratch in two different patterns of moderate numbers: what differs from either)
//   waw    words both write; waw_nonzero: of those, how many c writes as anything but zero (in any of the four runs of c
//          above and below): where it writes zeros there too, the order of the two makes no difference
//   infl   words of c's output, outside wz, that differ when the zero pass runs before c (real contents; and random
//          contents of moderate numbers): c reads what the zero pass writes
// The pair may run with no barrier between them only if waw_nonzero and both infl are 0. c must write no image (else skipped).
static std::string g_probe_zero;
static int g_probe_frame = -1, g_frame_now = -1;
struct Recorded {
    uint64_t pipeline = 0;
    std::vector<uint64_t> sets;
    std::vector<uint32_t> dynamic;
    uint32_t groups[3]{};
    std::string entry;
    bool writes_image = false;
    VkBuffer scratch = VK_NULL_HANDLE;
    VkDeviceSize scratch_size = 0;
};
static Recorded g_last_work;
static Recorded recorded(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    Recorded r;
    hook::Bound& b = hook::g_bound[H(cb)];
    r.pipeline = b.pipeline, r.sets = b.sets, r.dynamic = b.dynamic, r.groups[0] = x, r.groups[1] = y, r.groups[2] = z;
    auto p = g_pipes.find(b.pipeline);
    r.entry = p == g_pipes.end() ? "?" : p->second.entry;
    for (uint64_t set : b.sets)
        for (auto& [key, w] : g_setwrites[set]) {
            if (w.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) r.writes_image = true;
            if (w.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && key / 1000 == 141) {   // (the upscaler's ScratchBuffer)
                r.scratch = (VkBuffer)w.a;
                r.scratch_size = g_buffers[w.a].size;
            }
        }
    return r;
}
static void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m.srcAccessMask = m.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
}
static void replay(VkCommandBuffer cb, const Recorded& r) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)r.pipeline);
    std::vector<VkDescriptorSet> sets;
    for (uint64_t s : r.sets) sets.push_back((VkDescriptorSet)s);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)g_pipes[r.pipeline].layout, 0, (uint32_t)sets.size(), sets.data(),
                            (uint32_t)r.dynamic.size(), r.dynamic.data());
    vkCmdDispatch(cb, r.groups[0], r.groups[1], r.groups[2]);
    full_barrier(cb);
}
static bool probe_dispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    if (g_frame_now != g_probe_frame || cb != g_cmd) return false;
    Recorded now = recorded(cb, x, y, z);
    if (now.entry.compare(0, 5, "mlfi_") != 0 || !now.scratch) return false;
    const bool zero = ("," + g_probe_zero + ",").find("," + now.entry + ",") != std::string::npos;
    if (!zero) { g_last_work = now; return false; }
    const Recorded& c = g_last_work;
    if (c.entry.empty() || c.scratch != now.scratch) return false;
    if (c.writes_image) { logf("PROBE %s after %s: skipped (it writes an image)", now.entry.c_str(), c.entry.c_str()); fprintf(stderr, "PROBE %s skipped\n", now.entry.c_str()); return false; }
    const VkDeviceSize size = now.scratch_size;
    const size_t words = (size_t)(size / 4);
    // run what was recorded so far
    vkEndCommandBuffer(g_cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1, si.pCommandBuffers = &g_cmd;
    if (vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE) || vkQueueWaitIdle(g_queue)) die("probe: submit");
    // the buffers of the probe, made once
    static VkBuffer saved = VK_NULL_HANDLE, host = VK_NULL_HANDLE;
    static void* host_map = nullptr;
    static std::vector<uint32_t> random_words;
    if (!saved) {
        auto make = [&](VkBuffer& b, VkMemoryPropertyFlags want, void** map) {
            VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            ci.size = size, ci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            if (vkCreateBuffer(g_device, &ci, nullptr, &b)) die("probe buffer");
            VkMemoryRequirements mr;
            vkGetBufferMemoryRequirements(g_device, b, &mr);
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = mr.size, ai.memoryTypeIndex = memory_type(mr.memoryTypeBits, want);
            VkDeviceMemory m;
            if (vkAllocateMemory(g_device, &ai, nullptr, &m)) die("probe memory");
            vkBindBufferMemory(g_device, b, m, 0);
            if (map) vkMapMemory(g_device, m, 0, VK_WHOLE_SIZE, 0, map);
        };
        make(saved, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, nullptr);
        make(host, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &host_map);
        // (moderate numbers in every format the network uses: each byte 0x30..0x3f)
        random_words.resize(words);
        uint32_t st = 12345;
        for (auto& w : random_words) {
            w = 0;
            for (int k = 0; k < 4; k++) st = st * 1664525u + 1013904223u, w |= (0x30u | (st >> 28)) << (8 * k);
        }
    }
    VkCommandBuffer& pc = g_cmd;   // (the harness's own command buffer, reused between submissions)
    auto begin_ = [&]() {
        vkResetCommandBuffer(pc, 0);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(pc, &bi);
    };
    auto end_ = [&]() {
        vkEndCommandBuffer(pc);
        if (vkQueueSubmit(g_queue, 1, &si, VK_NULL_HANDLE) || vkQueueWaitIdle(g_queue)) die("probe: submit");
    };
    auto copy_ = [&](VkBuffer from, VkBuffer to) {
        VkBufferCopy r{0, 0, size};
        vkCmdCopyBuffer(pc, from, to, 1, &r);
        full_barrier(pc);
    };
    auto fill_ = [&](uint32_t v) { vkCmdFillBuffer(pc, now.scratch, 0, size, v); full_barrier(pc); };
    auto upload_random = [&]() { memcpy(host_map, random_words.data(), (size_t)size); copy_(host, now.scratch); };
    // (each step: set the scratch buffer, run passes, read it back)
    auto step = [&](auto&& set, std::initializer_list<const Recorded*> passes) {
        begin_();
        set();
        for (const Recorded* r : passes) replay(pc, *r);
        copy_(now.scratch, host);
        end_();
        return std::vector<uint32_t>((const uint32_t*)host_map, (const uint32_t*)host_map + words);
    };
    begin_(), copy_(now.scratch, saved), end_();
    auto from_saved = [&]() { copy_(saved, now.scratch); };
    const std::vector<uint32_t> z1 = step([&] { fill_(0xffffffffu); }, {&now});
    std::vector<uint8_t> wz(words), wc(words);
    size_t n_wz = 0, n_wc = 0, n_waw = 0, notzero = 0, waw_nonzero = 0;
    for (size_t i = 0; i < words; i++) {
        if (z1[i] == 0) wz[i] = 1, n_wz++;
        else if (z1[i] != 0xffffffffu) notzero++;
    }
    {
        // (two different patterns of moderate numbers, not uniform fills: a pass that copies within the scratch buffer
        // would write a uniform fill's own value back)
        const std::vector<uint32_t> a = step(upload_random, {&c});
        for (auto& w : random_words) w ^= 0x0f0f0f0fu;
        const std::vector<uint32_t> b = step(upload_random, {&c});
        for (auto& w : random_words) w ^= 0x0f0f0f0fu;
        for (size_t i = 0; i < words; i++)
            if (a[i] != random_words[i] || b[i] != (random_words[i] ^ 0x0f0f0f0fu)) wc[i] = 1, n_wc++, n_waw += wz[i], waw_nonzero += wz[i] && (a[i] || b[i]);
    }
    auto influence = [&](auto&& set) {
        const std::vector<uint32_t> o1 = step(set, {&c});
        const std::vector<uint32_t> o2 = step(set, {&now, &c});
        size_t n = 0;
        for (size_t i = 0; i < words; i++) n += wc[i] && !wz[i] && o1[i] != o2[i], waw_nonzero += wc[i] && wz[i] && o1[i];
        return n;
    };
    const size_t infl_real = influence(from_saved), infl_random = influence(upload_random);
    begin_(), from_saved(), end_();   // (the real contents back)
    logf("PROBE %s after %s: wz=%zu wc=%zu waw=%zu waw_nonzero=%zu infl_real=%zu infl_random=%zu notzero=%zu", now.entry.c_str(), c.entry.c_str(), n_wz, n_wc,
         n_waw, waw_nonzero, infl_real, infl_random, notzero);
    fprintf(stderr, "PROBE %s after %s: wz=%zu wc=%zu waw=%zu waw_nonzero=%zu infl_real=%zu infl_random=%zu notzero=%zu\n", now.entry.c_str(), c.entry.c_str(),
            n_wz, n_wc, n_waw, waw_nonzero, infl_real, infl_random, notzero);
    // go on recording where the frame was: the pass's own state bound again
    begin_();
    full_barrier(pc);
    vkCmdBindPipeline(pc, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipeline)now.pipeline);
    std::vector<VkDescriptorSet> sets;
    for (uint64_t s2 : now.sets) sets.push_back((VkDescriptorSet)s2);
    vkCmdBindDescriptorSets(pc, VK_PIPELINE_BIND_POINT_COMPUTE, (VkPipelineLayout)g_pipes[now.pipeline].layout, 0, (uint32_t)sets.size(), sets.data(),
                            (uint32_t)now.dynamic.size(), now.dynamic.data());
    return true;
}
static void to_layout(VkImage im, VkImageLayout to, uint32_t mips = 1) {
    ImageInfo& info = g_images[H(im)];
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.oldLayout = info.layout, b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = im, b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
    vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    info.layout = to;
}
static uint32_t texel_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_SINT: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16_SNORM: return 2;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT:
    case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R64_UINT: case VK_FORMAT_R64_SINT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
    default: return 0;
    }
}
static void upload(VkImage im, const void* data, size_t bytes) {
    ImageInfo& info = g_images[H(im)];
    Staging s = staging(bytes);
    memcpy(s.map, data, bytes);
    begin();
    to_layout(im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy r{};
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, r.imageExtent = info.extent;
    vkCmdCopyBufferToImage(g_cmd, s.buffer, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    to_layout(im, VK_IMAGE_LAYOUT_GENERAL);
    submit();
    free_staging(s);
}
/// Every image the provider made (and ours), each mip, as raw texels: <prefix>_img<id>_m<mip>.raw
static void dump_images(const char* prefix, bool ours_only) {
    for (auto& [handle, info] : g_images) {
        const uint32_t tb = texel_bytes(info.format);
        if (ours_only != info.ours && ours_only) continue;
        if (!tb || info.layout == VK_IMAGE_LAYOUT_UNDEFINED || info.extent.depth != 1) {
            if (!tb) logf("dump: img%d format %d not dumped", info.id, info.format);
            continue;
        }
        for (uint32_t mip = 0; mip < info.mips; mip++) {
            const uint32_t w = std::max(1u, info.extent.width >> mip), h = std::max(1u, info.extent.height >> mip);
            Staging s = staging((VkDeviceSize)w * h * tb);
            const VkImageLayout was = info.layout;
            begin();
            to_layout((VkImage)handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, info.mips);
            VkBufferImageCopy r{};
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1}, r.imageExtent = {w, h, 1};
            vkCmdCopyImageToBuffer(g_cmd, (VkImage)handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.buffer, 1, &r);
            to_layout((VkImage)handle, was, info.mips);
            submit();
            char name[128];
            snprintf(name, sizeof name, "%s_img%d_m%u.raw", prefix, info.id, mip);
            dump_file(name, s.map, (size_t)s.size);
            logf("dump %s img%d mip%u %ux%u format=%d name=%s", name, info.id, mip, w, h, info.format, info.name.c_str());
            free_staging(s);
        }
    }
}
static void dump_buffers(const char* prefix, VkDeviceSize limit) {
    for (auto& [handle, info] : g_buffers) {
        if (!(info.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) || info.size > limit) continue;
        Staging s = staging(info.size);
        begin();
        VkBufferCopy r{0, 0, info.size};
        vkCmdCopyBuffer(g_cmd, (VkBuffer)handle, s.buffer, 1, &r);
        submit();
        char name[128];
        snprintf(name, sizeof name, "%s_buf%d.raw", prefix, info.id);
        dump_file(name, s.map, (size_t)s.size);
        logf("dump %s buf%d size=%llu name=%s", name, info.id, (unsigned long long)info.size, info.name.c_str());
        free_staging(s);
    }
}

static uint16_t half(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man = (man | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | ((man + 0x1000) >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | ((man + 0x1000) >> 13));
}
/// sin and cos from + - * / and floor only, so the mingw and glibc builds draw the same pictures and send the same camera
/// (their libm sin() differs in the last bit for some inputs). Accuracy is beside the point; determinism is the point.
static double det_sin(double x) {
    const double two_pi = 6.283185307179586, pi = 3.141592653589793;
    x -= two_pi * std::floor(x / two_pi + 0.5);   // [-pi, pi]
    if (x > pi * 0.5) x = pi - x;
    if (x < -pi * 0.5) x = -pi - x;               // [-pi/2, pi/2]
    const double x2 = x * x;   // (Taylor to x^15: under 1e-10 off on this range)
    double p = -1.0 / 1307674368000.0;
    p = p * x2 + 1.0 / 6227020800.0;
    p = p * x2 - 1.0 / 39916800.0;
    p = p * x2 + 1.0 / 362880.0;
    p = p * x2 - 1.0 / 5040.0;
    p = p * x2 + 1.0 / 120.0;
    p = p * x2 - 1.0 / 6.0;
    p = p * x2 + 1.0;
    return x * p;
}
static float dsin(float x) { return (float)det_sin(x); }
static float dcos(float x) { return (float)det_sin((double)x + 1.5707963267948966); }
static uint32_t hash2(int x, int y) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
/// The camera of a frame: still, or (TRACE_CAMERA=k) moving and turning a little more each frame, k times as fast
struct Cam { float pos[3], right[3], up[3], fwd[3]; };
static Cam camera_at(int frame) {
    const float t = getenv("TRACE_CAMERA") ? (float)frame * (float)atof(getenv("TRACE_CAMERA")) : 0.0f;
    const float yaw = 0.07f * t + 0.013f * t * t, pitch = 0.03f * t;
    Cam c;
    const float f[3] = {dsin(yaw) * dcos(pitch), dsin(pitch), dcos(yaw) * dcos(pitch)};
    const float rgt[3] = {dcos(yaw), 0.0f, -dsin(yaw)};
    const float up[3] = {f[1] * rgt[2] - f[2] * rgt[1], f[2] * rgt[0] - f[0] * rgt[2], f[0] * rgt[1] - f[1] * rgt[0]};
    const float pos[3] = {0.31f * t, 0.11f * t * t, -0.23f * t};
    for (int i = 0; i < 3; i++) c.pos[i] = pos[i], c.right[i] = rgt[i], c.up[i] = up[i], c.fwd[i] = f[i];
    return c;
}
static double dot3(const double* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
/// The scene's 8-bit RGBA in the back buffer's format: 8-bit formats as they are (BGRA swapped), 10-bit as v * 1023 / 255
/// (integer, exact), 16-bit float as v / 255 * TRACE_COLOR_GAIN (default 1; above 1 makes values past 1.0, for scRGB).
static std::vector<uint8_t> in_format(const std::vector<uint8_t>& rgba, VkFormat f) {
    const size_t n = rgba.size() / 4;
    std::vector<uint8_t> out;
    switch (f) {
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
        out = rgba;
        for (size_t i = 0; i < n; i++) std::swap(out[i * 4], out[i * 4 + 2]);
        return out;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: {
        out.resize(n * 4);
        for (size_t i = 0; i < n; i++) {
            const uint32_t r = rgba[i * 4] * 1023u / 255u, g = rgba[i * 4 + 1] * 1023u / 255u, b = rgba[i * 4 + 2] * 1023u / 255u;
            const uint32_t v = r | g << 10 | b << 20 | 3u << 30;
            memcpy(&out[i * 4], &v, 4);
        }
        return out;
    }
    case VK_FORMAT_R16G16B16A16_SFLOAT: {
        const float gain = getenv("TRACE_COLOR_GAIN") ? (float)atof(getenv("TRACE_COLOR_GAIN")) : 1.0f;
        out.resize(n * 8);
        for (size_t i = 0; i < n; i++)
            for (int c = 0; c < 4; c++) {
                const uint16_t h = half(c == 3 ? 1.0f : rgba[i * 4 + c] / 255.0f * gain);
                memcpy(&out[i * 8 + c * 2], &h, 2);
            }
        return out;
    }
    default: return rgba;   // (8-bit RGBA, UNORM or SRGB)
    }
}
// (TRACE_ALLOC: allocation callbacks for create and destroy that count what they hand out and take back)
static int g_allocs = 0, g_frees = 0;
static void* count_alloc(void*, size_t size) { g_allocs++; return malloc(size); }
static void count_free(void*, void* mem) { if (mem) g_frees++; free(mem); }

static void message(uint32_t type, const wchar_t* m) {
    std::string s;
    for (; m && *m; m++) s += (char)*m;
    logf("FFX MESSAGE (%u): %s", type, s.c_str());
    fprintf(stderr, "ffx: %s\n", s.c_str());
}

/// A wall seen through the camera, and a nearer square that moves across the picture on its own, rendered at the render
/// size with the frame's jitter, with the depth and motion that go with them. Motion is in pixels at the motion vectors'
/// size, pointing from this frame's pixel to where it was (FSR's convention). The camera: x right, y up, looking along +z,
/// vertical field of view fov, aspect of the render size.
struct Scene {
    uint32_t rw, rh;           // the render size: colour and depth
    uint32_t mw, mh;           // the motion vectors' size: the render size, or the display size (DISPLAY_RESOLUTION_MOTION_VECTORS)
    bool inverted, infinite;   // (the depth convention of the create flags)
    float near_, far_, fov;
    static constexpr double WALL_Z = 20.0;
    float depth_at(float d) const {
        const double n = near_, f = far_, z = infinite ? 1.0 - n / d : f * (d - n) / (d * (f - n));
        return (float)(inverted ? 1.0 - z : z);
    }
    std::vector<uint8_t> color;
    std::vector<float> depth;
    std::vector<uint16_t> motion;
    static void background(float x, float y, uint8_t* out) {
        const int cx = (int)std::floor(x / 48.0f), cy = (int)std::floor(y / 48.0f);
        const uint32_t hsh = hash2(cx, cy);
        const float fx = x / 48.0f - cx, fy = y / 48.0f - cy;
        const float edge = (fx < 0.06f || fy < 0.06f) ? 0.35f : 1.0f;
        const float stripe = 0.75f + 0.25f * dsin((x + y) * 0.21f);
        out[0] = (uint8_t)(edge * stripe * (60 + (hsh & 0x7f)));
        out[1] = (uint8_t)(edge * stripe * (60 + ((hsh >> 8) & 0x7f)));
        out[2] = (uint8_t)(edge * stripe * (60 + ((hsh >> 16) & 0x7f)));
        out[3] = 255;
    }
    void wall(const Cam& c, double u, double v, double aspect, double* p, double* dist) const {
        const double th = det_sin(fov * 0.5) / det_sin(fov * 0.5 + 1.5707963267948966);
        const double nx = (2.0 * u - 1.0) * th * aspect, ny = (1.0 - 2.0 * v) * th;
        double d[3];
        for (int i = 0; i < 3; i++) d[i] = c.fwd[i] + c.right[i] * nx + c.up[i] * ny;
        const double t = (WALL_Z - c.pos[2]) / d[2];
        for (int i = 0; i < 3; i++) p[i] = c.pos[i] + t * d[i];
        *dist = t;
    }
    void project(const Cam& c, const double* p, double aspect, double* u, double* v) const {
        const double th = det_sin(fov * 0.5) / det_sin(fov * 0.5 + 1.5707963267948966);
        const double q[3] = {p[0] - c.pos[0], p[1] - c.pos[1], p[2] - c.pos[2]};
        const double z = dot3(q, c.fwd);
        *u = (dot3(q, c.right) / (z * th * aspect) + 1.0) * 0.5;
        *v = (1.0 - dot3(q, c.up) / (z * th)) * 0.5;
    }
    /// frame's picture; jx, jy: the jitter in render pixels (the sample of pixel x is at x + 0.5 + jx, FSR's convention)
    void make(int frame, float jx, float jy) {
        color.resize((size_t)rw * rh * 4), depth.resize((size_t)rw * rh), motion.resize((size_t)mw * mh * 2);
        const Cam cam = camera_at(frame), prev = camera_at(frame > 0 ? frame - 1 : 0);
        const float sqx = 0.70f - 0.017f * frame, sqy = 0.35f + 0.006f * frame, sqvx = -0.017f, sqvy = 0.006f, sqr = 0.11f;
        const double aspect = (double)rw / rh;
        auto in_square = [&](float u, float v) { return std::fabs(u - sqx) < sqr / (float)aspect && std::fabs(v - sqy) < sqr; };
        for (uint32_t y = 0; y < rh; y++)
            for (uint32_t x = 0; x < rw; x++) {
                const float u = (x + 0.5f + jx) / rw, v = (y + 0.5f + jy) / rh;
                uint8_t* px = &color[((size_t)y * rw + x) * 4];
                double p[3], dist;
                if (in_square(u, v)) {
                    const float lu = (u - sqx) * (float)aspect, lv = v - sqy;
                    const bool check = (((int)std::floor(lu * 40) + (int)std::floor(lv * 40)) & 1) != 0;
                    px[0] = check ? 230 : 40, px[1] = check ? 200 : 90, px[2] = check ? 40 : 200, px[3] = 255;
                    dist = 2.0;
                } else {
                    wall(cam, u, v, aspect, p, &dist);
                    background((float)(p[0] * 50.0), (float)(-p[1] * 50.0), px);
                }
                depth[(size_t)y * rw + x] = depth_at((float)dist);
            }
        for (uint32_t y = 0; y < mh; y++)
            for (uint32_t x = 0; x < mw; x++) {
                const float u = (x + 0.5f) / mw, v = (y + 0.5f) / mh;
                float mx, my;
                if (in_square(u, v)) mx = -sqvx * mw, my = -sqvy * mh;
                else {
                    double p[3], dist, pu, pv;
                    wall(cam, u, v, aspect, p, &dist);
                    project(prev, p, aspect, &pu, &pv);
                    mx = (float)((pu - u) * mw), my = (float)((pv - v) * mh);
                }
                motion[((size_t)y * mw + x) * 2 + 0] = half(mx);
                motion[((size_t)y * mw + x) * 2 + 1] = half(my);
            }
    }
};

static float envf(const char* n, float d) { const char* v = getenv(n); return v ? (float)atof(v) : d; }
static int envi(const char* n, int d) { const char* v = getenv(n); return v ? (int)strtol(v, nullptr, 0) : d; }
/// the k-th comma-separated number of an environment variable, or d
static long env_list(const char* n, int k, long d) {
    const char* q = getenv(n);
    for (int i = 0; i < k && q; i++) q = strchr(q, ','), q = q ? q + 1 : nullptr;
    return q && *q ? strtol(q, nullptr, 0) : d;
}

/// the k-th comma-separated float of an environment variable, or d
static float envf_list(const char* n, int k, float d) {
    const char* q = getenv(n);
    for (int i = 0; i < k && q; i++) q = strchr(q, ','), q = q ? q + 1 : nullptr;
    return q && *q ? (float)strtod(q, nullptr) : d;
}

#ifdef _WIN32
// ---------------------------------------------------------------- what AMD's DLL learns of the GPU outside Vulkan
// AMD's DLL picks among its FSR 4.1.2 providers by asking the device itself (vkGetPhysicalDeviceProperties2,
// vkEnumerateDeviceExtensionProperties, vkGetPhysicalDeviceFeatures2, imported from vulkan-1.dll, not through the device's
// functions the harness hands it) and, for its INT8 providers, the GPU's family and revision from AMD's Windows kernel
// driver (D3DKMTQueryAdapterInfo, KMTQAITYPE_UMDRIVERPRIVATE, found with GetProcAddress in gdi32.dll), which Wine does
// not have. To trace those providers, the harness can answer for the driver and hide extensions, by pointing the DLL's
// import table (in memory, after loading it: the file is not touched, no code is changed) at the functions below:
//   TRACE_AMD_FAMILY=n TRACE_AMD_REV=n   the family and revision the "kernel driver" reports (e.g. 0x91 0x01 for Navi 31)
//   TRACE_HIDE_EXT=name,...              device extensions the DLL's own enumeration does not see (VK_KHR_cooperative_matrix:
//                                        no FP8 provider, so the INT8 one is taken on an RDNA 4 GPU)
namespace spoof {
static std::string g_hide;
static long g_family = -1, g_rev = 0;
static PFN_vkEnumerateDeviceExtensionProperties real_enum;
static FARPROC(WINAPI* real_gpa)(HMODULE, LPCSTR);

static VkResult VKAPI_CALL enum_ext(VkPhysicalDevice p, const char* layer, uint32_t* n, VkExtensionProperties* out) {
    uint32_t all = 0;
    if (VkResult r = real_enum(p, layer, &all, nullptr)) return r;
    std::vector<VkExtensionProperties> e(all);
    real_enum(p, layer, &all, e.data());
    e.resize(all);
    e.erase(std::remove_if(e.begin(), e.end(), [](const VkExtensionProperties& x) {
                return g_hide.find(std::string(",") + x.extensionName + ",") != std::string::npos;
            }), e.end());
    if (!out) return *n = (uint32_t)e.size(), VK_SUCCESS;
    const uint32_t k = std::min<uint32_t>(*n, (uint32_t)e.size());
    std::copy(e.begin(), e.begin() + k, out);
    *n = k;
    return k < e.size() ? VK_INCOMPLETE : VK_SUCCESS;
}
// (the layouts of d3dkmthk.h: D3DKMT_ENUMADAPTERS, D3DKMT_QUERYADAPTERINFO)
struct KmtAdapter { UINT handle; LUID luid; ULONG sources; BOOL precise; };
struct KmtEnum { ULONG count; KmtAdapter adapters[16]; };
struct KmtQuery { UINT adapter; UINT type; void* data; UINT size; };
static LONG APIENTRY kmt_enum(KmtEnum* e) {
    memset(e, 0, sizeof *e);
    e->count = 1, e->adapters[0].handle = 0x40000040u, e->adapters[0].sources = 1;
    logf("SPOOF D3DKMTEnumAdapters: one adapter");
    return 0;
}
static LONG APIENTRY kmt_query(KmtQuery* q) {
    // (AMD's private data in its first, short form: the family at +0x30, the revision at +0x34; traced in the DLL's reading)
    if (q->type != 0 || q->size < 0x38) return (LONG)0xc000000du;   // STATUS_INVALID_PARAMETER
    memset(q->data, 0, q->size);
    ((uint32_t*)q->data)[0x30 / 4] = (uint32_t)g_family, ((uint32_t*)q->data)[0x34 / 4] = (uint32_t)g_rev;
    logf("SPOOF D3DKMTQueryAdapterInfo type=%u size=%u: family=0x%lx revision=0x%lx", q->type, q->size, g_family, g_rev);
    return 0;
}
static FARPROC WINAPI gpa(HMODULE m, LPCSTR name) {
    if ((uintptr_t)name > 0xffff && g_family >= 0) {
        if (!strcmp(name, "D3DKMTEnumAdapters")) return (FARPROC)kmt_enum;
        if (!strcmp(name, "D3DKMTQueryAdapterInfo")) return (FARPROC)kmt_query;
    }
    return real_gpa(m, name);
}
/// points the import of <dll>!<name> of the loaded module at f; returns what it pointed at
static void* patch_import(void* module, const char* dll, const char* name, void* f) {
    uint8_t* base = (uint8_t*)module;
    auto* nt = (IMAGE_NT_HEADERS64*)(base + ((IMAGE_DOS_HEADER*)base)->e_lfanew);
    const IMAGE_DATA_DIRECTORY dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    for (auto* d = (IMAGE_IMPORT_DESCRIPTOR*)(base + dir.VirtualAddress); d->Name; d++) {
        if (_stricmp((const char*)(base + d->Name), dll)) continue;
        auto* names = (IMAGE_THUNK_DATA64*)(base + d->OriginalFirstThunk);
        auto* iat = (IMAGE_THUNK_DATA64*)(base + d->FirstThunk);
        for (; names->u1.AddressOfData; names++, iat++) {
            if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal)) continue;
            if (strcmp((const char*)((IMAGE_IMPORT_BY_NAME*)(base + names->u1.AddressOfData))->Name, name)) continue;
            DWORD old;
            VirtualProtect(&iat->u1.Function, 8, PAGE_READWRITE, &old);
            void* was = (void*)iat->u1.Function;
            iat->u1.Function = (ULONGLONG)f;
            VirtualProtect(&iat->u1.Function, 8, old, &old);
            return was;
        }
    }
    die("spoof: %s!%s is not imported", dll, name);
}
static void install(void* module) {
    if (const char* f = getenv("TRACE_AMD_FAMILY")) {
        g_family = strtol(f, nullptr, 0), g_rev = getenv("TRACE_AMD_REV") ? strtol(getenv("TRACE_AMD_REV"), nullptr, 0) : 0;
        real_gpa = (FARPROC(WINAPI*)(HMODULE, LPCSTR))patch_import(module, "KERNEL32.dll", "GetProcAddress", (void*)gpa);
        logf("spoof: the kernel driver reports family 0x%lx revision 0x%lx", g_family, g_rev);
    }
    if (const char* h = getenv("TRACE_HIDE_EXT")) {
        g_hide = std::string(",") + h + ",";
        real_enum = (PFN_vkEnumerateDeviceExtensionProperties)patch_import(module, "vulkan-1.dll", "vkEnumerateDeviceExtensionProperties", (void*)enum_ext);
        logf("spoof: device extensions hidden from the provider: %s", h);
    }
}
} // namespace spoof
#endif

int main(int argc, char** argv) {
    if (argc < 3) die("usage: trace <provider library> <out dir> [W H RW RH frames dumpFrom]");
    g_out = argv[2];
    const uint32_t W = argc > 3 ? atoi(argv[3]) : 1920, Hh = argc > 4 ? atoi(argv[4]) : 1080;
    const uint32_t RW = argc > 5 ? atoi(argv[5]) : W, RH = argc > 6 ? atoi(argv[6]) : Hh;
    const int frames = argc > 7 ? atoi(argv[7]) : 4, dump_from = argc > 8 ? atoi(argv[8]) : 2;
#ifdef _WIN32
    _mkdir(g_out.c_str());
#else
    mkdir(g_out.c_str(), 0755);
#endif
    g_log = fopen((g_out + "/trace.log").c_str(), "w");
    if (!g_log) die("cannot write to %s", g_out.c_str());
    // TRACE_TIMING=1: GPU timestamps around the upscale, CPU time of create and dispatch, to <out>/timing.txt; no log or dumps
    const bool timing = envi("TRACE_TIMING", 0) > 0;
    g_quiet = timing;
    if (const char* pf = getenv("TRACE_PROBE")) g_probe_frame = atoi(pf), g_probe_zero = getenv("TRACE_PROBE_ZERO") ? getenv("TRACE_PROBE_ZERO") : "";
    using Clock = std::chrono::steady_clock;
    auto ms_since = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };

#ifdef _WIN32
    void* vk = load_lib("vulkan-1.dll");
#else
    void* vk = load_lib("libvulkan.so.1");
#endif
    if (!vk) die("no Vulkan loader");
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)sym(vk, "vkGetInstanceProcAddr");
    vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(nullptr, "vkCreateInstance");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    if (VkResult r = vkCreateInstance(&ici, nullptr, &inst)) die("instance %d", r);
#define LOADI(n) n = (PFN_##n)vkGetInstanceProcAddr(inst, #n);
    INSTANCE_FUNCS(LOADI)
    uint32_t n = 8;
    VkPhysicalDevice devs[8];
    vkEnumeratePhysicalDevices(inst, &n, devs);
    if (!n) die("no GPU");
    g_phys = devs[0];
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { g_phys = devs[i]; break; }
    }
    if (const char* want = getenv("TRACE_GPU"))
        for (uint32_t i = 0; i < n; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(devs[i], &p);
            if (strstr(p.deviceName, want)) { g_phys = devs[i]; break; }
        }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_phys, &props);
    logf("GPU %s vendor=0x%x device=0x%x driver=0x%x api=0x%x", props.deviceName, props.vendorID, props.deviceID, props.driverVersion, props.apiVersion);
    fprintf(stderr, "GPU %s\n", props.deviceName);
    vkGetPhysicalDeviceMemoryProperties(g_phys, &g_memprops);

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> exts(ne);
    vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &ne, exts.data());
    auto has = [&](const char* e) { return std::any_of(exts.begin(), exts.end(), [&](auto& x) { return !strcmp(x.extensionName, e); }); };
    std::vector<const char*> enable;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat8FeaturesEXT ff8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT fi64{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT};
    VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR fcsd{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
    VkPhysicalDeviceDescriptorBufferFeaturesEXT fdb{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT};
    void** tail = &f2.pNext;
    auto chain = [&](void* s) { *tail = s, tail = (void**)&((VkBaseOutStructure*)s)->pNext; };
    chain(&f11), chain(&f12), chain(&f13);
    struct Opt { const char* name; void* feat; };
    // (TRACE_NO_EXT=name,...: leave these device extensions out, to see what the provider does without them)
    const std::string no_ext = getenv("TRACE_NO_EXT") ? std::string(",") + getenv("TRACE_NO_EXT") + "," : "";
    for (Opt o : {Opt{"VK_KHR_cooperative_matrix", &fcm}, Opt{"VK_EXT_shader_float8", &ff8}, Opt{"VK_EXT_shader_image_atomic_int64", &fi64},
                  Opt{"VK_KHR_compute_shader_derivatives", &fcsd}, Opt{"VK_AMD_buffer_marker", nullptr},
                  Opt{"VK_AMD_device_coherent_memory", nullptr}, Opt{"VK_KHR_push_descriptor", nullptr},
                  // (TRACE_DESCRIPTOR_BUFFER=1: also VK_EXT_descriptor_buffer, which fsr4-vk needs and AMD's DLL does not)
                  Opt{getenv("TRACE_DESCRIPTOR_BUFFER") ? "VK_EXT_descriptor_buffer" : "", &fdb}}) {
        if (!*o.name) continue;
        const bool h = has(o.name) && no_ext.find(std::string(",") + o.name + ",") == std::string::npos;
        logf("extension %s: %s", o.name, h ? "yes" : "NO");
        if (!h) continue;
        enable.push_back(o.name);
        if (o.feat) chain(o.feat);
    }
    vkGetPhysicalDeviceFeatures2(g_phys, &f2);
    logf("features: coopmat=%d float8=%d float8coop=%d imageInt64=%d shaderInt64=%d float16=%d int8=%d", fcm.cooperativeMatrix, ff8.shaderFloat8,
         ff8.shaderFloat8CooperativeMatrix, fi64.shaderImageInt64Atomics, f2.features.shaderInt64, f12.shaderFloat16, f12.shaderInt8);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &nq, qf.data());
    uint32_t family = 0;
    for (uint32_t i = 0; i < nq; i++)
        if ((qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { family = i; break; }
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family, qci.queueCount = 1, qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2, dci.queueCreateInfoCount = 1, dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enable.size(), dci.ppEnabledExtensionNames = enable.data();
    if (VkResult r = vkCreateDevice(g_phys, &dci, nullptr, &g_device)) die("device %d", r);
#define LOADD(n) n = (PFN_##n)vkGetDeviceProcAddr(g_device, #n);
    DEVICE_FUNCS(LOADD)
    vkGetDeviceQueue(g_device, family, 0, &g_queue);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, pci.queueFamilyIndex = family;
    VkCommandPool pool;
    vkCreateCommandPool(g_device, &pci, nullptr, &pool);
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool, cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(g_device, &cai, &g_cmd);

    // ---- the provider
    void* lib = load_lib(argv[1]);
    if (!lib) die("%s did not load", argv[1]);
#ifdef _WIN32
    spoof::install(lib);
#endif
    auto create = (PfnFfxCreateContext)sym(lib, "ffxCreateContext");
    auto destroy = (PfnFfxDestroyContext)sym(lib, "ffxDestroyContext");
    auto configure = (PfnFfxConfigure)sym(lib, "ffxConfigure");
    auto query = (PfnFfxQuery)sym(lib, "ffxQuery");
    auto dispatch = (PfnFfxDispatch)sym(lib, "ffxDispatch");
    if (!create || !destroy || !configure || !query || !dispatch) die("the provider lacks an entry point");

    ffxConfigureDescGlobalDebug dbg{};
    dbg.header.type = FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG;
    dbg.fpMessage = message, dbg.debugLevel = FFX_API_CONFIGURE_GLOBALDEBUG_LEVEL_VERBOSE;
    logf("configure(global debug) -> %u", configure(nullptr, &dbg.header));

    VkDeviceContext dc{g_device, g_phys, hook::GetDeviceProcAddr};
    uint64_t count = 0, ids[16]{};
    const char* names[16]{};
    ffxQueryDescGetVersions qv{};
    qv.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    qv.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, qv.device = &dc, qv.outputCount = &count;
    ffxReturnCode_t r = query(nullptr, &qv.header);
    logf("query versions -> %u count=%llu", r, (unsigned long long)count);
    count = std::min<uint64_t>(count, 16);
    qv.versionIds = ids, qv.versionNames = names;
    r = query(nullptr, &qv.header);
    for (uint64_t i = 0; i < count; i++) {
        logf("  version %llu: id=0x%llx name=%s", (unsigned long long)i, (unsigned long long)ids[i], names[i] ? names[i] : "?");
        fprintf(stderr, "provider version: %s (0x%llx)\n", names[i] ? names[i] : "?", (unsigned long long)ids[i]);
    }

    // ---- the context-free queries (for the suite: compared with AMD's answers)
    for (uint32_t mode = 0; mode <= 5; mode++) {
        float ratio = -1.0f;
        ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode q{};
        q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE, q.qualityMode = mode, q.pOutUpscaleRatio = &ratio;
        const ffxReturnCode_t a = query(nullptr, &q.header);
        uint32_t ow = 0, oh = 0;
        ffxQueryDescUpscaleGetRenderResolutionFromQualityMode q2{};
        q2.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE;
        q2.displayWidth = W, q2.displayHeight = Hh, q2.qualityMode = mode, q2.pOutRenderWidth = &ow, q2.pOutRenderHeight = &oh;
        const ffxReturnCode_t b = query(nullptr, &q2.header);
        logf("=== query (no context) quality %u: ratio=%.9g -> %u; render for %ux%u = %ux%u -> %u", mode, ratio, a, W, Hh, ow, oh, b);
    }

    // ---- the context
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.deviceContext = dc;
    // TRACE_CALLER=eglantine: create as Eglantine's (Lichiistein's) Linux FSR 4 path does: the deprecated backend type
    // (fsr4-vk knows only that one) and a version override with the first GET_VERSIONS entry whose name starts with 4..9
    const bool eglantine = getenv("TRACE_CALLER") && !strcmp(getenv("TRACE_CALLER"), "eglantine");
    if (eglantine) backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK_DEPRECATED;
    ffxOverrideVersion over{};
    over.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
    over.header.pNext = &backend.header;
    if (eglantine)
        for (uint64_t i = 0; i < count; i++)
            if (names[i] && names[i][0] >= '4' && names[i][0] <= '9') { over.versionId = ids[i]; break; }
    if (const char* want = getenv("TRACE_VERSION")) {
        over.versionId = strtoull(want, nullptr, 0);
        logf("override version 0x%llx", (unsigned long long)over.versionId);
    }
    ffxCreateContextDescUpscaleVersion version{};
    version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE_VERSION;
    version.header.pNext = over.versionId ? &over.header : &backend.header;
    version.version = FFX_UPSCALER_VERSION;
    ffxCreateContextDescUpscale cc{};
    cc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    cc.header.pNext = &version.header;
    // (default: what Eglantine's Ocarina path asks for, inverted infinite depth, linear HDR-ish colour)
    cc.flags = (uint32_t)envi("TRACE_CREATE_FLAGS", FFX_UPSCALE_ENABLE_DEPTH_INVERTED | FFX_UPSCALE_ENABLE_DEPTH_INFINITE);
    // (TRACE_MAX=w,h: the maximum render size, if larger than the render size; dynamic resolution)
    cc.maxRenderSize = {(uint32_t)env_list("TRACE_MAX", 0, RW), (uint32_t)env_list("TRACE_MAX", 1, RH)};
    cc.maxUpscaleSize = {W, Hh};
    cc.fpMessage = message;
    auto memory_v2 = [&]() {
        FfxApiEffectMemoryUsage mu{};
        ffxQueryDescUpscaleGetGPUMemoryUsageV2 q2{};
        q2.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE_V2;
        q2.device = &dc, q2.maxRenderSize = cc.maxRenderSize, q2.maxUpscaleSize = cc.maxUpscaleSize, q2.flags = cc.flags;
        q2.gpuMemoryUsageUpscaler = &mu;
        const ffxReturnCode_t qr = query(nullptr, &q2.header);
        logf("=== query memory v2 flags=0x%x total=%llu aliasable=%llu -> %u", cc.flags, (unsigned long long)mu.totalUsageInBytes,
             (unsigned long long)mu.aliasableUsageInBytes, qr);
    };
    memory_v2();
    ffxContext ctx = nullptr;
    const VkFormat color_format = (VkFormat)envi("TRACE_COLOR_FORMAT", VK_FORMAT_R16G16B16A16_SFLOAT);
    const VkFormat out_format = (VkFormat)envi("TRACE_OUT_FORMAT", color_format);
    logf("=== create context display=%ux%u render=%ux%u max=%ux%u flags=0x%x format=%d out=%d", W, Hh, RW, RH, cc.maxRenderSize.width,
         cc.maxRenderSize.height, cc.flags, color_format, out_format);
    ffxAllocationCallbacks counting{nullptr, count_alloc, count_free};
    const ffxAllocationCallbacks* callbacks = getenv("TRACE_ALLOC") ? &counting : nullptr;
    const Clock::time_point create_start = Clock::now();
    r = create(&ctx, &cc.header, callbacks);
    const double create_ms = ms_since(create_start);
    logf("=== create context -> %u ctx=%p", r, ctx);
    fprintf(stderr, "create context -> %u\n", r);
    if (r != FFX_API_RETURN_OK || !ctx) {
        logf("LIVE %zu of the provider's objects left after the failed create", hook::g_live.size());
        return 2;
    }
    auto memory_v1 = [&](const char* when) {
        FfxApiEffectMemoryUsage mu{};
        ffxQueryDescUpscaleGetGPUMemoryUsage q1{};
        q1.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE;
        q1.gpuMemoryUsageUpscaler = &mu;
        const ffxReturnCode_t qr = query(&ctx, &q1.header);
        logf("=== query memory v1 %s total=%llu aliasable=%llu -> %u", when, (unsigned long long)mu.totalUsageInBytes,
             (unsigned long long)mu.aliasableUsageInBytes, qr);
    };
    memory_v1("after create");
    ffxQueryGetProviderVersion pv{};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    if (query(&ctx, &pv.header) == FFX_API_RETURN_OK) {
        logf("provider in use: %s (0x%llx)", pv.versionName ? pv.versionName : "?", (unsigned long long)pv.versionId);
        fprintf(stderr, "in use: %s\n", pv.versionName ? pv.versionName : "?");
    }
    {
        ffxQueryDescUpscaleGetResourceRequirements rq{};
        rq.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS;
        const ffxReturnCode_t qr = query(&ctx, &rq.header);
        logf("=== query resource requirements required=0x%llx optional=0x%llx -> %u", (unsigned long long)rq.required_resources,
             (unsigned long long)rq.optional_resources, qr);
    }
    int32_t phases = 0;
    {
        ffxQueryDescUpscaleGetJitterPhaseCount q{};
        q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT, q.renderWidth = RW, q.displayWidth = W, q.pOutPhaseCount = &phases;
        const ffxReturnCode_t qr = query(&ctx, &q.header);
        logf("=== query jitter phase count %u -> %u: %d", RW, qr, phases);
    }
    // (key-value configuration: TRACE_KV=key:float,key:float...)
    if (const char* kv = getenv("TRACE_KV"))
        for (const char* q = kv; q && *q;) {
            ffxConfigureDescUpscaleKeyValue c{};
            c.header.type = FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE;
            float value = 0.0f;
            c.key = strtoull(q, (char**)&q, 0);
            if (*q == ':') value = (float)strtod(q + 1, (char**)&q);
            c.ptr = &value;
            logf("=== configure key %llu = %.9g -> %u", (unsigned long long)c.key, value, configure(&ctx, &c.header));
            q = strchr(q, ','), q = q ? q + 1 : nullptr;
        }

    FILE* timing_out = nullptr;
    VkQueryPool queries = VK_NULL_HANDLE;
    if (timing) {
        if (!props.limits.timestampComputeAndGraphics) die("TRACE_TIMING: the device has no timestamps on this queue");
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP, qi.queryCount = 2;
        if (VkResult qr = vkCreateQueryPool(g_device, &qi, nullptr, &queries)) die("query pool %d", qr);
        timing_out = fopen((g_out + "/timing.txt").c_str(), "w");
        if (!timing_out) die("cannot write timing.txt");
        fprintf(timing_out, "# GPU %s, display %ux%u, render %ux%u\n# create_ms %.3f\n# frame upscale_gpu_ms dispatch_cpu_ms\n", props.deviceName, W, Hh, RW,
                RH, create_ms);
    }
    const bool display_mv = (cc.flags & FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS) != 0;
    const uint32_t MRW = cc.maxRenderSize.width, MRH = cc.maxRenderSize.height;
    // (the inputs are made at the largest render size; each frame's render size is the part used)
    VkImage color = our_image(color_format, MRW, MRH, "in_color");
    VkImage depth = our_image(VK_FORMAT_R32_SFLOAT, MRW, MRH, "in_depth");
    VkImage motion = our_image(VK_FORMAT_R16G16_SFLOAT, display_mv ? W : MRW, display_mv ? Hh : MRH, "in_motion");
    VkImage output = our_image(out_format, W, Hh, "out_upscaled");
    VkImage exposure = VK_NULL_HANDLE, reactive = VK_NULL_HANDLE, tc = VK_NULL_HANDLE;
    begin();
    to_layout(output, VK_IMAGE_LAYOUT_GENERAL);
    submit();
    // TRACE_EXPOSURE=v: a 1x1 R32_SFLOAT exposure of v
    if (const char* e = getenv("TRACE_EXPOSURE")) {
        exposure = our_image(VK_FORMAT_R32_SFLOAT, 1, 1, "in_exposure");
        const float v = (float)atof(e);
        upload(exposure, &v, 4);
    }
    // TRACE_REACTIVE / TRACE_TC: R8 masks at the render size, a square of 0.5 / 0.75 over the middle of the picture
    auto mask = [&](const char* name, uint8_t value) {
        VkImage im = our_image(VK_FORMAT_R8_UNORM, MRW, MRH, name);
        std::vector<uint8_t> m((size_t)MRW * MRH);
        for (uint32_t y = 0; y < MRH; y++)
            for (uint32_t x = 0; x < MRW; x++) m[(size_t)y * MRW + x] = (x > MRW / 3 && x < MRW * 2 / 3 && y > MRH / 3 && y < MRH * 2 / 3) ? value : 0;
        upload(im, m.data(), m.size());
        return im;
    };
    if (getenv("TRACE_REACTIVE")) reactive = mask("in_reactive", 128);
    // TRACE_GEN_REACTIVE=flags: ffxDispatchDescUpscaleGenerateReactiveMask before each upscale, from the colour (as opaque
    // and as final), into an R8 image of ours
    VkImage gen_reactive = getenv("TRACE_GEN_REACTIVE") ? our_image(VK_FORMAT_R8_UNORM, MRW, MRH, "out_reactive") : VK_NULL_HANDLE;
    if (gen_reactive) begin(), to_layout(gen_reactive, VK_IMAGE_LAYOUT_GENERAL), submit();
    if (getenv("TRACE_TC")) tc = mask("in_tc", 192);
    // (Eglantine's caller: inputs in COMPUTE_READ, usage 0, in SHADER_READ_ONLY_OPTIMAL; otherwise GENERIC_READ, in GENERAL)
    auto res = [&](VkImage im, VkFormat f, uint32_t w, uint32_t h, bool write) {
        FfxApiResourceDescription d{};
        d.type = FFX_API_RESOURCE_TYPE_TEXTURE2D, d.format = ffxApiGetSurfaceFormatVK(f);
        d.width = w, d.height = h, d.depth = 1, d.mipCount = 1, d.flags = FFX_API_RESOURCE_FLAGS_NONE;
        d.usage = write ? FFX_API_RESOURCE_USAGE_UAV : (eglantine ? 0 : FFX_API_RESOURCE_USAGE_READ_ONLY);
        return ffxApiGetResourceVK((void*)im, d, write ? FFX_API_RESOURCE_STATE_UNORDERED_ACCESS
                                                        : eglantine ? FFX_API_RESOURCE_STATE_COMPUTE_READ : FFX_API_RESOURCE_STATE_GENERIC_READ);
    };
    const float cam_near = envf("TRACE_NEAR", 0.1f), cam_far = envf("TRACE_FAR", 3.402823466e+38f), fov = envf("TRACE_FOV", 1.0f);

    for (int frame = 0; frame < frames; frame++) {
        // (TRACE_RENDER_W / TRACE_RENDER_H=a,b,c...: the render size frame by frame, for dynamic resolution)
        const uint32_t rw = (uint32_t)env_list("TRACE_RENDER_W", frame, RW), rh = (uint32_t)env_list("TRACE_RENDER_H", frame, RH);
        float jx = 0.0f, jy = 0.0f;
        if (!getenv("TRACE_NO_JITTER")) {
            int32_t ph = 0;
            ffxQueryDescUpscaleGetJitterPhaseCount q{};
            q.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT, q.renderWidth = rw, q.displayWidth = W, q.pOutPhaseCount = &ph;
            query(&ctx, &q.header);
            ffxQueryDescUpscaleGetJitterOffset qo{};
            qo.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET, qo.index = frame, qo.phaseCount = ph, qo.pOutX = &jx, qo.pOutY = &jy;
            const ffxReturnCode_t qr = query(&ctx, &qo.header);
            logf("=== query jitter offset %d of %d -> %u: %.9g,%.9g", frame, ph, qr, jx, jy);
        }
        Scene scene{rw, rh, display_mv ? W : rw, display_mv ? Hh : rh, (cc.flags & FFX_UPSCALE_ENABLE_DEPTH_INVERTED) != 0,
                    (cc.flags & FFX_UPSCALE_ENABLE_DEPTH_INFINITE) != 0, cam_near, cam_far, fov};
        scene.make(frame, jx, jy);
        // (the inputs fill the top left of the largest-size images: rows padded to the image's width)
        auto padded = [&](const void* src, uint32_t w, uint32_t h, uint32_t bpp, uint32_t iw, uint32_t ih) {
            std::vector<uint8_t> out((size_t)iw * ih * bpp);
            for (uint32_t y = 0; y < h; y++) memcpy(&out[(size_t)y * iw * bpp], (const uint8_t*)src + (size_t)y * w * bpp, (size_t)w * bpp);
            return out;
        };
        const std::vector<uint8_t> texels = in_format(scene.color, color_format);
        const uint32_t cb = texel_bytes(color_format);
        const auto c = padded(texels.data(), rw, rh, cb, MRW, MRH);
        upload(color, c.data(), c.size());
        const auto d = padded(scene.depth.data(), rw, rh, 4, MRW, MRH);
        upload(depth, d.data(), d.size());
        const uint32_t miw = display_mv ? W : MRW, mih = display_mv ? Hh : MRH;
        const auto m = padded(scene.motion.data(), scene.mw, scene.mh, 4, miw, mih);
        upload(motion, m.data(), m.size());
        logf("=== FRAME %d render=%ux%u", frame, rw, rh);
        g_frame_now = frame;

        begin();
        if (timing) {
            vkCmdResetQueryPool(g_cmd, queries, 0, 2);
            vkCmdWriteTimestamp(g_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
        }
        if (gen_reactive) {
            ffxDispatchDescUpscaleGenerateReactiveMask gm{};
            gm.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK;
            gm.commandList = (void*)g_cmd;
            gm.colorOpaqueOnly = res(color, color_format, MRW, MRH, false), gm.colorPreUpscale = res(color, color_format, MRW, MRH, false);
            gm.outReactive = res(gen_reactive, VK_FORMAT_R8_UNORM, MRW, MRH, true);
            gm.renderSize = {rw, rh};
            gm.scale = 1.0f, gm.cutoffThreshold = 0.2f, gm.binaryValue = 0.9f;
            gm.flags = (uint32_t)envi("TRACE_GEN_REACTIVE", 0);
            logf("=== dispatch generate reactive mask -> %u", dispatch(&ctx, &gm.header));
        }
        ffxDispatchDescUpscale up{};
        up.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        up.commandList = (void*)g_cmd;
        up.color = res(color, color_format, MRW, MRH, false);
        up.depth = res(depth, VK_FORMAT_R32_SFLOAT, MRW, MRH, false);
        up.motionVectors = res(motion, VK_FORMAT_R16G16_SFLOAT, miw, mih, false);
        if (exposure) up.exposure = res(exposure, VK_FORMAT_R32_SFLOAT, 1, 1, false);
        if (reactive) up.reactive = res(reactive, VK_FORMAT_R8_UNORM, MRW, MRH, false);
        if (tc) up.transparencyAndComposition = res(tc, VK_FORMAT_R8_UNORM, MRW, MRH, false);
        up.output = res(output, out_format, W, Hh, true);
        // (the jitter as the camera applied it: FSR's sign convention, the offset itself)
        up.jitterOffset = {jx, jy};
        up.motionVectorScale = {envf("TRACE_MVSCALE_X", 1.0f), envf("TRACE_MVSCALE_Y", 1.0f)};
        up.renderSize = {rw, rh};
        // (TRACE_UPSCALE_W / TRACE_UPSCALE_H=a,b,...: the upscale size frame by frame, below the create's maximum)
        up.upscaleSize = {(uint32_t)env_list("TRACE_UPSCALE_W", frame, W), (uint32_t)env_list("TRACE_UPSCALE_H", frame, Hh)};
        up.enableSharpening = getenv("TRACE_SHARPNESS") != nullptr && env_list("TRACE_SHARPEN_ON", frame, 1) != 0;   // (_ON=1,0,...: per frame)
        up.sharpness = envf_list("TRACE_SHARPNESS", frame, 0.0f);
        up.frameTimeDelta = envf("TRACE_FRAMETIME", 16.6667f);
        up.preExposure = envf_list("TRACE_PREEXPOSURE", frame, 1.0f);   // (a,b,c...: frame by frame)
        up.reset = frame == 0 || frame == envi("TRACE_RESET_AT", -1);
        up.cameraNear = cam_near, up.cameraFar = cam_far, up.cameraFovAngleVertical = fov;
        up.viewSpaceToMetersFactor = envf("TRACE_V2M", 1.0f);
        up.flags = (uint32_t)env_list("TRACE_DISPATCH_FLAGS_LIST", frame, envi("TRACE_DISPATCH_FLAGS", 0));   // (_LIST: frame by frame)
        std::vector<VkImage> inputs = {color, depth, motion};
        if (exposure) inputs.push_back(exposure);
        if (eglantine)
            for (VkImage im : inputs) to_layout(im, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        logf("=== dispatch upscale reset=%d jitter=%.9g,%.9g sharpen=%d %.9g flags=0x%x upscale=%ux%u preexposure=%.9g", up.reset, jx, jy,
             up.enableSharpening, up.sharpness, up.flags, up.upscaleSize.width, up.upscaleSize.height, up.preExposure);
        const Clock::time_point cpu_start = Clock::now();
        r = dispatch(&ctx, &up.header);
        const double cpu_ms = ms_since(cpu_start);
        if (timing) vkCmdWriteTimestamp(g_cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
        if (eglantine)
            for (VkImage im : inputs) to_layout(im, VK_IMAGE_LAYOUT_GENERAL);
        logf("=== dispatch upscale -> %u", r);
        if (!timing) fprintf(stderr, "frame %d: upscale -> %u, %d dispatches so far\n", frame, r, g_dispatch);
        submit();
        if (timing) {
            uint64_t t[2];
            if (VkResult qr = vkGetQueryPoolResults(g_device, queries, 0, 2, sizeof t, t, sizeof t[0], VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT))
                die("query results %d", qr);
            const uint64_t mask_ = qf[family].timestampValidBits < 64 ? (1ull << qf[family].timestampValidBits) - 1 : ~0ull;
            fprintf(timing_out, "%d %.4f %.4f\n", frame, (double)((t[1] - t[0]) & mask_) * props.limits.timestampPeriod * 1e-6, cpu_ms);
        }
        logf("=== frame %d submitted and finished", frame);
        if (!timing && frame >= dump_from) {
            char prefix[32];
            snprintf(prefix, sizeof prefix, "f%d", frame);
            dump_images(prefix, !getenv("TRACE_DUMP_ALL") || !*getenv("TRACE_DUMP_ALL"));
            if (getenv("TRACE_DUMP_BUFFERS")) dump_buffers(prefix, (VkDeviceSize)atoll(getenv("TRACE_DUMP_BUFFERS")));
        }
    }
    if (timing_out) fclose(timing_out);
    memory_v1("after the frames");
    vkDeviceWaitIdle(g_device);
    r = destroy(&ctx, callbacks);
    logf("=== destroy -> %u", r);
    logf("LIVE %zu of the provider's objects left after destroy", hook::g_live.size());
    if (!timing) fprintf(stderr, "provider objects left after destroy: %zu\n", hook::g_live.size());
    if (callbacks) logf("ALLOC allocations=%d frees=%d", g_allocs, g_frees);
    fprintf(stderr, "done: %d dispatches traced\n", g_dispatch);
    fclose(g_log);
    return 0;
}
