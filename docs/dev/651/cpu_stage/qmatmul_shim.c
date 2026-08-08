// qmatmul_shim.c - minimal C shim exposing ggml-cpu quantized MUL_MAT to ctypes.
//
// Existence proof for #651's CPU PP-prefill stage: compute Q4-class matmuls
// directly on PACKED GGUF blocks via llama.cpp's ggml-cpu kernels, without
// dequantizing weights to dense bf16 first.
//
// Build (see build_shim.sh):
//   gcc -O2 -shared -fPIC qmatmul_shim.c -I llama.cpp/ggml/include \
//       -L llama.cpp/build/bin -lggml -lggml-cpu -lggml-base \
//       -Wl,-rpath,'$ORIGIN/llama.cpp/build/bin' -o libqmatmul_shim.so
//
// API notes for fork integration:
//   - The context only holds tensor/graph METADATA (no_alloc = true); tensor
//     data pointers are patched to caller buffers, including dst -> out, so
//     there is zero copying of weights or results.
//   - The work buffer (src1 quantized to the vec_dot_type, e.g. Q8_K) is
//     malloc'd per call. A real integration should cache it, and reuse a
//     ggml_threadpool instead of the per-call thread spawn ggml_graph_compute
//     does when cplan.threadpool == NULL.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-cpu.h"

// Error codes
#define QMM_OK               0
#define QMM_ERR_TYPE        -1  // unknown/unsupported ggml type
#define QMM_ERR_ROW_BYTES   -2  // row_bytes != ggml_row_size(type, cols)
#define QMM_ERR_CTX         -3  // ggml_init failed
#define QMM_ERR_COMPUTE     -4  // graph compute returned non-success
#define QMM_ERR_ARGS        -5  // bad dimensions / null pointers

static int g_cpu_initialized = 0;

// 1 if ggml-cpu has a direct vec_dot kernel for this type (i.e. MUL_MAT runs
// natively on packed blocks), 0 otherwise.
int qmm_supported(int ggml_type) {
    if (ggml_type < 0 || ggml_type >= GGML_TYPE_COUNT) {
        return 0;
    }
    const struct ggml_type_traits_cpu * traits =
        ggml_get_type_traits_cpu((enum ggml_type) ggml_type);
    return traits != NULL && traits->vec_dot != NULL;
}

// Returns ggml_row_size(type, cols) so callers can pre-validate, -1 on bad type.
long long qmm_row_size(int ggml_type, long long cols) {
    if (ggml_type < 0 || ggml_type >= GGML_TYPE_COUNT) {
        return -1;
    }
    return (long long) ggml_row_size((enum ggml_type) ggml_type, (int64_t) cols);
}

// out[t][rows] = x[t][cols] @ dequant(w[rows][row_bytes])^T
// w:   packed quantized weight, rows x row_bytes, row-major (GGUF layout)
// x:   f32 activations, t x cols, row-major
// out: f32 result, t x rows, row-major
int qmm(int ggml_type, const void * w, long long rows, long long row_bytes,
        const float * x, long long t, long long cols, float * out,
        int n_threads) {
    if (w == NULL || x == NULL || out == NULL ||
        rows <= 0 || cols <= 0 || t <= 0 || n_threads <= 0) {
        return QMM_ERR_ARGS;
    }
    if (ggml_type < 0 || ggml_type >= GGML_TYPE_COUNT) {
        return QMM_ERR_TYPE;
    }
    enum ggml_type type = (enum ggml_type) ggml_type;
    if (!qmm_supported(ggml_type)) {
        return QMM_ERR_TYPE;
    }
    if ((long long) ggml_row_size(type, (int64_t) cols) != row_bytes) {
        return QMM_ERR_ROW_BYTES;
    }

    if (!g_cpu_initialized) {
        ggml_cpu_init();  // fp16 tables etc.; idempotent, but avoid re-entry
        g_cpu_initialized = 1;
    }

    // Metadata-only context: tensors point at caller memory (no_alloc = true).
    struct ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8 +
                          ggml_graph_overhead() + 4096,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ip);
    if (ctx == NULL) {
        return QMM_ERR_CTX;
    }

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, type, (int64_t) cols, (int64_t) rows);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) cols, (int64_t) t);
    a->data = (void *) w;   // packed GGUF bytes, used in place
    b->data = (void *) x;

    struct ggml_tensor * dst = ggml_mul_mat(ctx, a, b);  // f32, ne = [rows, t]
    dst->data = out;        // result written straight into caller buffer

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, NULL);
    void * work = NULL;
    if (plan.work_size > 0) {
        work = malloc(plan.work_size);
        if (work == NULL) {
            ggml_free(ctx);
            return QMM_ERR_CTX;
        }
        plan.work_data = (uint8_t *) work;
    }

    enum ggml_status st = ggml_graph_compute(gf, &plan);

    free(work);
    ggml_free(ctx);

    return st == GGML_STATUS_SUCCESS ? QMM_OK : QMM_ERR_COMPUTE;
}
