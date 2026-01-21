#pragma once

#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <cuda.h>

namespace quotient_jit {

struct JITKernel {
    CUmodule module;
    CUfunction function;
};

// Simple JIT compiler using NVRTC
class JITCompiler {
public:
    static JITCompiler& getInstance() {
        static thread_local JITCompiler instance;
        return instance;
    }

    // Compiles the source and returns a kernel handle. 
    // Uses internal cache based on source hash.
    CUfunction getOrCompile(const std::string& source, const std::string& cacheKey);

private:
    JITCompiler();
    ~JITCompiler();
    
    std::map<std::string, JITKernel> kernelCache;
    std::mutex cacheMutex;
    
    // Disallow copy
    JITCompiler(const JITCompiler&) = delete;
    JITCompiler& operator=(const JITCompiler&) = delete;
};

} // namespace quotient_jit
