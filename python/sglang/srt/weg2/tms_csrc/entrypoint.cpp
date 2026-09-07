#include "utils.h"
#include "core.h"
#include "api_forwarder.h"
#include <optional>
#include <cstring>
#include "macro.h"

// ----------------------------------------------- threadlocal configs --------------------------------------------------

class ThreadLocalConfig {
public:
    std::string current_tag_ = "default";

    bool is_interesting_region() {
        if (!is_interesting_region_.has_value()) {
            is_interesting_region_ = get_bool_env_var("TMS_INIT_ENABLE");
        }
        return is_interesting_region_.value();
    }

    void set_interesting_region(bool value) {
        is_interesting_region_ = value;
    }

    bool enable_cpu_backup() {
        if (!enable_cpu_backup_.has_value()) {
            enable_cpu_backup_ = get_bool_env_var("TMS_INIT_ENABLE_CPU_BACKUP");
        }
        return enable_cpu_backup_.value();
    }

    void set_enable_cpu_backup(bool value) {
        enable_cpu_backup_ = value;
    }

private:
    std::optional<bool> is_interesting_region_;
    std::optional<bool> enable_cpu_backup_;
};
static thread_local ThreadLocalConfig thread_local_config;

// ------------------------------------------------- entrypoints :: hook ------------------------------------------------

#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(
            ptr, CUDAUtils::cu_ctx_get_device(), size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup());
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}

cudaError_t cudaFree(void *ptr) {
    if (thread_local_config.is_interesting_region()) {
        return TorchMemorySaver::instance().free(ptr);
    } else {
        return APIForwarder::call_real_cuda_free(ptr);
    }
}
#endif

#ifdef TMS_HOOK_MODE_TORCH
extern "C" {
void *tms_torch_malloc(ssize_t size, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] tms_torch_malloc "
              << " size=" << size << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    void *ptr;
    TorchMemorySaver::instance().malloc(
        &ptr, CUDAUtils::cu_device_get(device), size, thread_local_config.current_tag_, thread_local_config.enable_cpu_backup());
    return ptr;
}

void tms_torch_free(void *ptr, ssize_t ssize, int device, cudaStream_t stream) {
#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] tms_torch_free "
              << " ptr=" << ptr << " ssize=" << ssize << " device=" << device << " stream=" << stream
              << std::endl;
#endif
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    TorchMemorySaver::instance().free(ptr);
}
}
#endif

// ------------------------------------------------- entrypoints :: others ------------------------------------------------

extern "C" {
void tms_set_interesting_region(bool is_interesting_region) {
    thread_local_config.set_interesting_region(is_interesting_region);
}

bool tms_get_interesting_region() {
    return thread_local_config.is_interesting_region();
}

void tms_set_current_tag(const char* tag) {
    SIMPLE_CHECK(tag != nullptr, "tag should not be null");
    thread_local_config.current_tag_ = tag;
}

bool tms_get_enable_cpu_backup() {
    return thread_local_config.enable_cpu_backup();
}

void tms_set_enable_cpu_backup(bool enable_cpu_backup) {
    thread_local_config.set_enable_cpu_backup(enable_cpu_backup);
}

void tms_pause(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().pause(tag_str);
}

void tms_resume(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    TorchMemorySaver::instance().resume(tag_str);
}

// C7 -- THE PLANNER'S NEW SIZING INPUT (spec R8).  The per-tag host bytes used
// to be read as an RssShmem delta around the pause (weight_updater.py:738).
// With the ring that instrument is DEAD BY CONSTRUCTION: granules are shared
// tmpfs pages mapped by BOTH co-located processes, so the delta collapses to
// ~0 and a per-process sum double-counts.  These two entries publish the
// saver's OWN accounting instead, which is the only place the bytes are known
// exactly.
uint64_t tms_tag_bytes(const char* tag) {
    std::string tag_str = (tag != nullptr) ? std::string(tag) : "";
    return TorchMemorySaver::instance().tag_bytes(tag_str);
}

//: Returns 0 when this boot published no ring (stock cudaMallocHost path), 1
//: otherwise.  Every out-pointer may be null.  ``card_uuid`` is written NUL
//: terminated and truncated to ``card_uuid_len``.
int tms_ring_stats(char* card_uuid, size_t card_uuid_len,
                   uint64_t* granule_bytes,
                   uint64_t* granules_total,
                   uint64_t* granules_free,
                   uint64_t* granules_free_min,
                   uint64_t* granules_peak_taken,
                   uint64_t* acquires,
                   uint64_t* releases,
                   uint64_t* swept_stale,
                   uint64_t* spans_registered,
                   double* blocked_ms) {
    HostRingStats st;
    std::string uuid;
    if (!TorchMemorySaver::instance().ring_stats(&st, &uuid)) {
        return 0;
    }
    if (card_uuid != nullptr && card_uuid_len > 0) {
        size_t n = uuid.size() < (card_uuid_len - 1) ? uuid.size() : (card_uuid_len - 1);
        memcpy(card_uuid, uuid.c_str(), n);
        card_uuid[n] = '\0';
    }
    if (granule_bytes != nullptr) *granule_bytes = (uint64_t)TMS_RING_GRANULE_BYTES;
    if (granules_total != nullptr) *granules_total = st.granules_total;
    if (granules_free != nullptr) *granules_free = st.granules_free;
    if (granules_free_min != nullptr) *granules_free_min = st.granules_free_min;
    if (granules_peak_taken != nullptr) *granules_peak_taken = st.granules_peak_taken;
    if (acquires != nullptr) *acquires = st.acquires;
    if (releases != nullptr) *releases = st.releases;
    if (swept_stale != nullptr) *swept_stale = st.swept_stale;
    if (spans_registered != nullptr) *spans_registered = st.spans_registered;
    if (blocked_ms != nullptr) *blocked_ms = st.blocked_ms;
    return 1;
}
}