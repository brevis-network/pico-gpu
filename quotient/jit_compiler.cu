#include "jit_compiler.cuh"
#include <nvrtc.h>
#include <iostream>
#include <sstream>
#include <map>
#include <mutex>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

namespace quotient_jit {
 
#ifndef QUOTIENT_CACHE_PATH
#define QUOTIENT_CACHE_PATH "private-pico-gpu/quotient/cache"
#endif
 
#ifndef PICO_GPU_INCLUDE_PATH
#define PICO_GPU_INCLUDE_PATH "private-pico-gpu"
#endif

// static cache/mutex removed in favor of instance members for thread-local usage

#define NVRTC_SAFE_CALL(x)                                        \
    do {                                                          \
        nvrtcResult result = x;                                   \
        if (result != NVRTC_SUCCESS) {                            \
            std::cerr << "NVRTC error: " << nvrtcGetErrorString(result) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                              \
        }                                                         \
    } while (0)

#define CUDA_SAFE_CALL(x)                                         \
    do {                                                          \
        CUresult result = x;                                      \
        if (result != CUDA_SUCCESS) {                             \
            const char* msg;                                      \
            cuGetErrorName(result, &msg);                         \
            std::cerr << "CUDA Driver error: " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1);                                              \
        }                                                         \
    } while (0)

#include <cuda_runtime.h>

JITCompiler::JITCompiler() {
    // metadata only
    cudaFree(0);

    CUresult res = cuInit(0);
    if (res != CUDA_SUCCESS) {
        std::cerr << "CUDA Driver error: " << res << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        // exit(1);
    }
}

JITCompiler::~JITCompiler() {
    std::lock_guard<std::mutex> lock(cacheMutex);
    for (auto& entry : kernelCache) {
        cuModuleUnload(entry.second.module);
    }
}

CUfunction JITCompiler::getOrCompile(const std::string& source, const std::string& cacheKey) {
    std::lock_guard<std::mutex> lock(cacheMutex);
    
    // 1. Memory Cache Lookup: Instant return if already loaded in this process
    if (kernelCache.count(cacheKey)) {
        return kernelCache[cacheKey].function;
    }

    std::string cacheDir = QUOTIENT_CACHE_PATH;
    std::string cubinPath = cacheDir + "/" + cacheKey + ".cubin";

    // 2. Disk Cache Lookup: Load pre-compiled .cubin (machine code) to skip NVRTC cost
    std::ifstream cacheFile(cubinPath, std::ios::binary);
    if (cacheFile.is_open()) {
        std::cout << "[JIT] Loading cached CUBIN for key: " << cacheKey << std::endl;
        std::stringstream buffer;
        buffer << cacheFile.rdbuf();
        std::string cubin = buffer.str();

        CUmodule module;
        CUfunction function;
        // Direct binary load: bypassed Driver's secondary JIT optimization
        CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, cubin.c_str(), 0, nullptr, nullptr));
        CUDA_SAFE_CALL(cuModuleGetFunction(&function, module, "jit_computeValues"));

        kernelCache[cacheKey] = {module, function};
        return function;
    }

    // 3. JIT Compilation: Only occurs if both memory and disk caches miss
    std::cout << "[JIT] Compiling new kernel for key: " << cacheKey << " (this may take a few seconds...)" << std::endl;

    const char* header_sources[] = {
        "#pragma once\n#undef assert\n#define assert(x)\n",
        "#pragma once\ntypedef unsigned char uint8_t;\ntypedef unsigned short uint16_t;\ntypedef unsigned int uint32_t;\ntypedef unsigned long long uint64_t;\ntypedef unsigned long long uintptr_t;\n",
        "#pragma once\n#define printf(fmt, ...)\n"
    };
    const char* header_names[] = {
        "cassert",
        "cstdint",
        "cstdio"
    };

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, source.c_str(), "jit_kernel.cu", 3, header_sources, header_names));

    // Include path for resolving dependencies like <ff/kb31_t.hpp>
    std::string includePathStr = std::string("-I") + PICO_GPU_INCLUDE_PATH;
    const char* includePath = includePathStr.c_str();
    
    std::vector<const char*> opts = {
        "-std=c++17",
        includePath,
        "-arch=sm_120", // This is for rtx5090 - generates direct machine code (CUBIN)
        // "-arch=sm_89", // This is for rtx4090 
        "-DFEATURE_KOALA_BEAR",
        "-DKOALA_BEAR_CANONICAL",
        "-default-device"
    };

    nvrtcResult compile_res = nvrtcCompileProgram(prog, opts.size(), opts.data());
    
    if (compile_res != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        std::vector<char> log(log_size);
        nvrtcGetProgramLog(prog, log.data());
        std::cerr << "[JIT] Compilation failed:\n" << log.data() << std::endl;
        exit(1);
    }

    size_t cubin_size;
    NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &cubin_size));
    std::vector<char> cubin(cubin_size);
    // Retrieve native machine code (SASS) instead of virtual assembly (PTX)
    NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, cubin.data()));

    // 4. Persistence: Save generated CUBIN to avoid future recompilation
    mkdir(cacheDir.c_str(), 0777);
    std::ofstream outCache(cubinPath, std::ios::binary);
    if (outCache.is_open()) {
        outCache.write(cubin.data(), cubin.size());
    } else {
        std::cerr << "[JIT] Warning: Could not save CUBIN to " << cubinPath << std::endl;
    }

    CUmodule module;
    CUfunction function;
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr));
    CUDA_SAFE_CALL(cuModuleGetFunction(&function, module, "jit_computeValues"));

    kernelCache[cacheKey] = {module, function};
    
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    return function;
}

} // namespace quotient_jit
