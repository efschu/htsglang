#pragma once
#include <dlfcn.h>
#include "macro.h"

namespace APIForwarder {
    cudaError_t call_real_cuda_malloc(void **ptr, size_t size);
    cudaError_t call_real_cuda_free(void *ptr);
}
