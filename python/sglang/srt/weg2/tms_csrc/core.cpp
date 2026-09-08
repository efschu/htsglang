#include "core.h"
#include "utils.h"
#include "macro.h"
#include "api_forwarder.h"

#include <cstring>
#include <map>
#include <string>

#if defined(USE_ROCM)
#include "hardware_amd_support.h"
#endif

TorchMemorySaver::TorchMemorySaver() {}

TorchMemorySaver &TorchMemorySaver::instance() {
    static TorchMemorySaver instance;
    return instance;
}

cudaError_t TorchMemorySaver::malloc(void **ptr, CUdevice device, size_t size, const std::string& tag, const bool enable_cpu_backup) {
#if defined(USE_ROCM)
    return ROCmHIPImplementation::rocm_malloc(ptr, device, size, tag, enable_cpu_backup, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    CUmemGenericAllocationHandle allocHandle;
    CUDAUtils::cu_mem_create(&allocHandle, size, device);
    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, {}, false, allocHandle}
        );
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_malloc "
              << " ptr=" << ptr << " *ptr=" << *ptr << " size=" << size
              << " allocHandle=" << allocHandle << " tag=" << tag
              << std::endl;
#endif

#else
    #error "USE_PLATFORM is not set"
#endif
    return cudaSuccess;
}

cudaError_t TorchMemorySaver::free(void *ptr) {
#if defined(USE_ROCM)
    return ROCmHIPImplementation::rocm_free(ptr, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    AllocationMetadata metadata;
    {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
        if (allocation_metadata_.count(ptr) == 0) {
            return APIForwarder::call_real_cuda_free(ptr);
        }

        metadata = allocation_metadata_[ptr];
        allocation_metadata_.erase(ptr);
    }

    CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
    CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
    CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));

    // C5: give the granules back to whichever allocator owns them.  A freed
    // allocation whose backup came from the ring MUST release, or the region
    // leaks bytes the ledger has already charged and the peer blocks forever.
    if (!metadata.cpu_backup_granules.empty()) {
        if (metadata.cpu_backup_from_ring) {
            SIMPLE_CHECK(ring_ != nullptr, "cpu_backup_from_ring without an open ring");
            ring_->release(metadata.cpu_backup_granules);
        } else {
            for (size_t k = 0; k < metadata.cpu_backup_granules.size(); ++k) {
                CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup_granules[k]));
            }
        }
        metadata.cpu_backup_granules.clear();
    }

#ifdef TMS_DEBUG_LOG
    std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.cuda_free "
              << " ptr=" << ptr << " metadata.size=" << metadata.size
              << " metadata.allocHandle=" << metadata.allocHandle << " tag=" << metadata.tag
              << std::endl;
#endif

#else
    #error "USE_PLATFORM is not set"
#endif
    return cudaSuccess;
}

void TorchMemorySaver::ensure_backup_stream() {
    if (backup_stream_ != nullptr) {
        return;
    }
    // R12: cudaStreamCreate, NOT cudaStreamCreateWithFlags(cudaStreamNonBlocking).
    // The default (blocking) flags keep the implicit ordering against the legacy
    // default stream that today's cudaMemcpy relies on; dropping it would copy
    // pages a pending kernel is still writing -- silent corruption, no error.
    CUDA_ERROR_CHECK(cudaStreamCreate(&backup_stream_));
}

HostBackupRing* TorchMemorySaver::ensure_ring(CUdevice device) {
    if (!ring_open_attempted_) {
        ring_open_attempted_ = true;
        ring_ = HostBackupRing::open_from_env(device);
    }
    return ring_;
}

void TorchMemorySaver::assert_host_backup_eligible(const AllocationMetadata& metadata) {
    // C6 / R20: kv_cache is paused WITHOUT enable_cpu_backup and stays RELEASED
    // across the flip.  If an allocation with enable_cpu_backup == false ever
    // reached a host allocator, the ring would be sized for weights and asked to
    // hold a KV pool, and the ledger's arithmetic would be wrong by a term
    // nobody printed.  Token-checked by T5.
    SIMPLE_CHECK(metadata.enable_cpu_backup,
                 "R20: an allocation with enable_cpu_backup == false must never reach a host "
                 "backup allocator (kv stays RELEASED, never backed up) tag=" + metadata.tag);
}

uint64_t TorchMemorySaver::tag_bytes(const std::string& tag) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    uint64_t total = 0;
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        if (tag.empty() || it->second.tag == tag) {
            total += static_cast<uint64_t>(it->second.size);
        }
    }
    return total;
}

int TorchMemorySaver::backed_up_tag_bytes(char* out, size_t len) {
    if (out == nullptr || len == 0) {
        return -1;
    }
    std::map<std::string, uint64_t> totals;
    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            // THE POPULATION IS enable_cpu_backup, NOT the tag name.  An
            // allocation without it never reaches a host granule (R20, asserted
            // in pause/resume), so it holds no host bytes the ring must size
            // for -- and charging it would over-size H and refuse the boot.
            if (!it->second.enable_cpu_backup) {
                continue;
            }
            totals[it->second.tag] += static_cast<uint64_t>(it->second.size);
        }
    }
    std::string text;
    for (auto it = totals.begin(); it != totals.end(); ++it) {
        if (!text.empty()) {
            text += ",";
        }
        text += it->first + "=" + std::to_string(it->second);
    }
    if (text.size() + 1 > len) {
        return -1;
    }
    std::memcpy(out, text.c_str(), text.size() + 1);
    return static_cast<int>(totals.size());
}

bool TorchMemorySaver::ring_stats(HostRingStats* out, std::string* card_uuid) {
    if (ring_ == nullptr) {
        return false;
    }
    if (out != nullptr) {
        *out = ring_->stats();
    }
    if (card_uuid != nullptr) {
        *card_uuid = ring_->card_uuid();
    }
    return true;
}

void TorchMemorySaver::pause(const std::string& tag) {
#if defined(USE_ROCM)
    ROCmHIPImplementation::rocm_pause(tag, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    // C3: THREE PASSES.  The old shape copied and unmapped one allocation at a
    // time on the legacy default stream, so every copy was a full host-side
    // synchronisation and the region's bytes were held for the whole loop.
    //
    // (1) acquire host granules and issue the D2H copies on backup_stream_
    // (2) ONE cudaStreamSynchronize for the whole tag
    // (3) the unmap/release/PAUSED loop, now unblocked
    //
    // The ORDER is load-bearing and is what T4 pins: no page may be unmapped
    // before the copy that reads it has completed, so the single sync lies
    // strictly between pass 1 and pass 3.
    std::vector<void*> matched_ptrs;
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::ACTIVE) {
            std::cerr << "[torch_memory_saver.cpp] Cannot pause allocation that is not active."
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }
        matched_ptrs.push_back(ptr);
    }

    // --- pass 1: host bytes + async D2H ---
    bool any_copy = false;
    for (size_t m = 0; m < matched_ptrs.size(); ++m) {
        void* ptr = matched_ptrs[m];
        AllocationMetadata& metadata = allocation_metadata_[ptr];
        if (!metadata.enable_cpu_backup) {
            continue;
        }
        assert_host_backup_eligible(metadata);
        ensure_backup_stream();
        if (metadata.cpu_backup_granules.empty()) {
            HostBackupRing* ring = ensure_ring(metadata.device);
            if (ring != nullptr) {
                metadata.cpu_backup_granules =
                    ring->acquire(metadata.size, TMS_RING_FAMILY_FLIP_BACKUP, metadata.tag);
                metadata.cpu_backup_from_ring = true;
            } else {
                // Stock path, unchanged in behaviour: ONE pinned block, held as
                // a one-element scatter list so both legs have one shape.
                void* block = nullptr;
                CUDA_ERROR_CHECK(cudaMallocHost(&block, metadata.size));
                metadata.cpu_backup_granules.assign(1, block);
                metadata.cpu_backup_from_ring = false;
            }
        }
        SIMPLE_CHECK(!metadata.cpu_backup_granules.empty(), "cpu backup granules should not be empty");
        size_t chunk = metadata.cpu_backup_from_ring ? TMS_RING_GRANULE_BYTES : metadata.size;
        size_t offset = 0;
        for (size_t g = 0; g < metadata.cpu_backup_granules.size(); ++g) {
            size_t n = metadata.size - offset;
            if (n > chunk) {
                n = chunk;
            }
            CUDA_ERROR_CHECK(cudaMemcpyAsync(metadata.cpu_backup_granules[g],
                                             (char*)ptr + offset, n,
                                             cudaMemcpyDeviceToHost, backup_stream_));
            offset += n;
        }
        SIMPLE_CHECK(offset == metadata.size, "D2H granule walk did not cover the allocation");
        any_copy = true;
    }

    // --- pass 2: ONE synchronisation for the whole tag ---
    if (any_copy) {
        CUDA_ERROR_CHECK(cudaStreamSynchronize(backup_stream_));
    }

    // --- pass 3: unmap / release / PAUSED ---
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata& metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }
        if (metadata.state != AllocationState::ACTIVE) {
            continue;
        }

        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));

        metadata.state = AllocationState::PAUSED;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.pause"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size << " metadata.allocHandle="
                  << metadata.allocHandle << " tag=" << metadata.tag << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << " granules=" << metadata.cpu_backup_granules.size()
                  << " from_ring=" << metadata.cpu_backup_from_ring
                  << std::endl;
#endif
    }
#else
    #error "USE_PLATFORM is not set"
#endif
}

void TorchMemorySaver::resume(const std::string& tag) {
#if defined(USE_ROCM)
    ROCmHIPImplementation::rocm_resume(tag, allocation_metadata_, allocator_metadata_mutex_);

#elif defined(USE_CUDA)
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

    // C4: FOUR PASSES -- map all, async H2D per granule, ONE synchronize, then
    // give the host bytes back.  Same ordering law as pause: the release in
    // pass 4 lies strictly AFTER the single cudaStreamSynchronize, which is
    // what makes the peer's acquire safe (R11: a granule is TAKEN until
    // release, and acquire only ever returns bits that are 0, so the peer can
    // never take a granule whose H2D is still in flight).
    std::vector<void*> matched_ptrs;
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        void *ptr = it->first;
        AllocationMetadata &metadata = it->second;

        if (!tag.empty() && metadata.tag != tag) {
            continue;
        }

        if (metadata.state != AllocationState::PAUSED) {
            std::cerr << "[torch_memory_saver.cpp] Cannot resume allocation that is not paused. "
                      << " tag=" << metadata.tag << " ptr=" << std::to_string((uintptr_t)ptr)
                      << " file=" << __FILE__ << " func=" << __func__ << " line=" << __LINE__
                      << std::endl;
            exit(1);
        }
        matched_ptrs.push_back(ptr);
    }

    // --- pass 1: map every allocation of the tag ---
    const auto weg2_map_t0 = std::chrono::steady_clock::now();   // S7 (#1273)
    for (size_t m = 0; m < matched_ptrs.size(); ++m) {
        void* ptr = matched_ptrs[m];
        AllocationMetadata& metadata = allocation_metadata_[ptr];

        CUmemGenericAllocationHandle newAllocHandle;
        CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);
        CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));
        CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);
        metadata.state = AllocationState::ACTIVE;
        metadata.allocHandle = newAllocHandle;

#ifdef TMS_DEBUG_LOG
        std::cout << "[torch_memory_saver.cpp] TorchMemorySaver.resume"
                  << " ptr=" << ptr << " metadata.size=" << metadata.size
                  << " (new)newAllocHandle=" << newAllocHandle << " tag=" << metadata.tag
                  << " filter_tag=" << tag
                  << " metadata.enable_cpu_backup=" << metadata.enable_cpu_backup
                  << std::endl;
#endif
    }

    const auto weg2_map_t1 = std::chrono::steady_clock::now();   // S7 (#1273)

    // --- pass 2: async H2D per granule ---
    bool any_copy = false;
    for (size_t m = 0; m < matched_ptrs.size(); ++m) {
        void* ptr = matched_ptrs[m];
        AllocationMetadata& metadata = allocation_metadata_[ptr];
        if (!metadata.enable_cpu_backup) {
            continue;
        }
        assert_host_backup_eligible(metadata);
        ensure_backup_stream();
        SIMPLE_CHECK(!metadata.cpu_backup_granules.empty(), "cpu backup granules should not be empty");
        size_t chunk = metadata.cpu_backup_from_ring ? TMS_RING_GRANULE_BYTES : metadata.size;
        size_t offset = 0;
        for (size_t g = 0; g < metadata.cpu_backup_granules.size(); ++g) {
            size_t n = metadata.size - offset;
            if (n > chunk) {
                n = chunk;
            }
            CUDA_ERROR_CHECK(cudaMemcpyAsync((char*)ptr + offset,
                                             metadata.cpu_backup_granules[g], n,
                                             cudaMemcpyHostToDevice, backup_stream_));
            offset += n;
        }
        SIMPLE_CHECK(offset == metadata.size, "H2D granule walk did not cover the allocation");
        any_copy = true;
    }

    // --- pass 3: ONE synchronisation for the whole tag ---
    if (any_copy) {
        CUDA_ERROR_CHECK(cudaStreamSynchronize(backup_stream_));
    }
    note_resume(tag, matched_ptrs.size(), weg2_map_t0, weg2_map_t1);   // S7 (#1273)

    // --- pass 4: give the host bytes back ---
    for (size_t m = 0; m < matched_ptrs.size(); ++m) {
        AllocationMetadata& metadata = allocation_metadata_[matched_ptrs[m]];
        if (!metadata.enable_cpu_backup || metadata.cpu_backup_granules.empty()) {
            continue;
        }
        // WEG2 ONE-BACKUP PATCH (#1233, DR-1): upstream 0.0.9.post1 keeps the
        // pinned host image after the restore "to reduce re-alloc time", so a
        // dormant-then-awake group still holds its whole weights image in
        // RssShmem (measured +15.4 GiB per shard, boot weg2s1) and two groups
        // hold two images at the flip (boot weg2ls1b2: host OOM, oom_kill
        // 7->18).  The design statement is "only ONE layout lies in host RAM":
        // the image is released the moment its bytes are back on the device.
        // cudaMemcpy from PINNED host memory is synchronous with respect to
        // the host, so the buffer is dead here; pause() re-allocates it
        // (cudaMallocHost above) on the next sleep -- that re-allocation is
        // the price, and it is paid per chunk tag, never for the whole image.
        //
        // WEG2 RING (C4): the invariant is UNCHANGED and the release is still
        // strictly after the copies have completed -- what changed is only WHO
        // the bytes go back to.  The "synchronous with respect to the host"
        // clause above is now carried by the explicit cudaStreamSynchronize in
        // pass 3 rather than by cudaMemcpy's own semantics, which is the same
        // guarantee stated where a reader can check it.
        if (metadata.cpu_backup_from_ring) {
            SIMPLE_CHECK(ring_ != nullptr, "cpu_backup_from_ring without an open ring");
            ring_->release(metadata.cpu_backup_granules);
        } else {
            for (size_t g = 0; g < metadata.cpu_backup_granules.size(); ++g) {
                CUDA_ERROR_CHECK(cudaFreeHost(metadata.cpu_backup_granules[g]));
            }
        }
        metadata.cpu_backup_granules.clear();
    }
#else
    #error "USE_PLATFORM is not set"
#endif
}
