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
    // REMAP (#1352): the page-granular arm.  Same VA, same accessibility, same
    // metadata key -- the ONLY difference is that the backing is N handles of
    // one page each instead of one handle of ``size``, which is what makes a
    // single page movable at all (see the AllocationMetadata comment).  The
    // legacy arm below is untouched, and an unset ``TMS_REMAP_TAGS`` takes it.
    if (remap_tag(tag)) {
        if (page_bytes_ == 0) {
            page_bytes_ = CUDAUtils::cu_mem_min_granularity(device);
            SIMPLE_CHECK(page_bytes_ > 0, "REMAP: driver reported a zero allocation granularity");
        }
        const size_t page = page_bytes_;
        const size_t aligned = ((size + page - 1) / page) * page;
        const size_t n_pages = aligned / page;

        CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, aligned, 0, 0, 0));
        std::vector<CUmemGenericAllocationHandle> handles(n_pages);
        for (size_t i = 0; i < n_pages; ++i) {
            CUDAUtils::cu_mem_create(&handles[i], page, device, remap_exportable_);
            CURESULT_CHECK(cuMemMap((CUdeviceptr) ((char*) *ptr + i * page), page, 0, handles[i], 0));
        }
        // ONE set_access over the whole run: it takes a VA RANGE, so the N
        // separate mappings cost one call, not N.
        CUDAUtils::cu_mem_set_access(*ptr, aligned, device);

        {
            const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
            allocation_metadata_.emplace(
                *ptr,
                AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup,
                                   {}, false, std::move(handles), aligned, 0}
            );
        }
        return cudaSuccess;
    }

    CUmemGenericAllocationHandle allocHandle;
    CUDAUtils::cu_mem_create(&allocHandle, size, device);
    CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
    CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
    CUDAUtils::cu_mem_set_access(*ptr, size, device);

    {
        const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
        allocation_metadata_.emplace(
            *ptr,
            AllocationMetadata{size, device, tag, AllocationState::ACTIVE, enable_cpu_backup,
                               {}, false, {}, size, allocHandle}
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

    if (!metadata.page_handles.empty()) {
        // REMAP: a zero entry is a page this allocation no longer owns (it was
        // remapped away, or the tag is paused).  Skipping it is not laxity --
        // unmapping a VA that carries no mapping is an error, and releasing a
        // handle the peer now owns would free a page under the peer's feet.
        for (size_t i = 0; i < metadata.page_handles.size(); ++i) {
            if (metadata.page_handles[i] == 0) {
                continue;
            }
            CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ((char*) ptr + i * page_bytes_), page_bytes_));
            CURESULT_CHECK(cuMemRelease(metadata.page_handles[i]));
        }
        CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.reserved_size));
    } else {
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
        CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
        CURESULT_CHECK(cuMemAddressFree((CUdeviceptr) ptr, metadata.size));
    }

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

// =========================================================== REMAP (#1352) ==
//
// The flip stops asking the driver for pages.  A physical page changes owner:
// ``cuMemUnmap`` in the source arena's VA, ``cuMemMap`` of the SAME handle in
// the destination arena's VA.  No ``cuMemCreate``, no ``cuMemRelease``, and
// therefore no credit counter standing in for the physics.

bool TorchMemorySaver::remap_tag(const std::string& tag) {
    if (!remap_prefixes_parsed_) {
        remap_prefixes_parsed_ = true;
        const char* raw = std::getenv("TMS_REMAP_TAGS");
        if (raw != nullptr && *raw != '\0') {
            std::string all(raw);
            size_t start = 0;
            while (start <= all.size()) {
                size_t comma = all.find(',', start);
                if (comma == std::string::npos) {
                    comma = all.size();
                }
                std::string one = all.substr(start, comma - start);
                if (!one.empty()) {
                    remap_prefixes_.push_back(one);
                }
                start = comma + 1;
            }
        }
        const char* exp = std::getenv("TMS_REMAP_EXPORTABLE");
        remap_exportable_ = (exp != nullptr && (*exp == '1' || *exp == 't' || *exp == 'T'));
    }
    for (size_t i = 0; i < remap_prefixes_.size(); ++i) {
        const std::string& p = remap_prefixes_[i];
        if (tag.size() >= p.size() && tag.compare(0, p.size(), p) == 0) {
            return true;
        }
    }
    return false;
}

std::vector<void*> TorchMemorySaver::ordered_ptrs(const std::string& tag) {
    // ADDRESS ORDER, and it is load-bearing rather than tidy: a page index is
    // only a stable name for a page if both ends of the plan derive it the same
    // way, and ``allocation_metadata_`` is an unordered_map whose iteration
    // order is an implementation detail that can differ between two processes
    // holding identical allocations.
    std::vector<void*> out;
    for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
        if (it->second.tag == tag && !it->second.page_handles.empty()) {
            out.push_back(it->first);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool TorchMemorySaver::locate_page(const std::vector<void*>& order, uint64_t index,
                                   void** ptr_out, uint64_t* local_out) {
    uint64_t seen = 0;
    for (size_t m = 0; m < order.size(); ++m) {
        const uint64_t n = (uint64_t) allocation_metadata_[order[m]].page_handles.size();
        if (index < seen + n) {
            *ptr_out = order[m];
            *local_out = index - seen;
            return true;
        }
        seen += n;
    }
    return false;
}

uint64_t TorchMemorySaver::tag_pages(const std::string& tag) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    const std::vector<void*> order = ordered_ptrs(tag);
    uint64_t total = 0;
    for (size_t m = 0; m < order.size(); ++m) {
        total += (uint64_t) allocation_metadata_[order[m]].page_handles.size();
    }
    return total;
}

int TorchMemorySaver::page_stats(const std::string& tag, uint64_t* pages, uint64_t* created,
                                 uint64_t* mapped_in, uint64_t* mapped_out) {
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);
    const std::vector<void*> order = ordered_ptrs(tag);
    if (order.empty()) {
        return 0;   // ABSENCE: the tag was never armed for remap.
    }
    uint64_t total = 0;
    for (size_t m = 0; m < order.size(); ++m) {
        total += (uint64_t) allocation_metadata_[order[m]].page_handles.size();
    }
    if (pages != nullptr) *pages = total;
    if (created != nullptr) *created = pages_created_.count(tag) ? pages_created_[tag] : 0;
    if (mapped_in != nullptr) *mapped_in = pages_mapped_in_.count(tag) ? pages_mapped_in_[tag] : 0;
    if (mapped_out != nullptr) *mapped_out = pages_mapped_out_.count(tag) ? pages_mapped_out_[tag] : 0;
    return 1;
}

int TorchMemorySaver::remap_pages(const std::string& src_tag, uint64_t src_page,
                                  const std::string& dst_tag, uint64_t dst_page,
                                  uint64_t n_pages, char* err, size_t errlen) {
#if defined(USE_CUDA)
    const std::lock_guard<std::mutex> lock(allocator_metadata_mutex_);

    auto fail = [&](int code, const std::string& why) {
        if (err != nullptr && errlen > 0) {
            const std::string msg = "W95 Weg2RemapPageRefused: " + why;
            size_t n = msg.size() < (errlen - 1) ? msg.size() : (errlen - 1);
            memcpy(err, msg.c_str(), n);
            err[n] = '\0';
        }
        return code;
    };

    if (n_pages == 0) {
        return 0;
    }
    if (src_tag == dst_tag) {
        return fail(-1, "src_tag == dst_tag (" + src_tag + ") -- a remap within one tag "
                        "moves a page onto itself and is a plan defect, not a no-op");
    }
    const std::vector<void*> src_order = ordered_ptrs(src_tag);
    const std::vector<void*> dst_order = ordered_ptrs(dst_tag);
    if (src_order.empty()) {
        return fail(-2, "tag '" + src_tag + "' has no page-granular allocation -- it was never "
                        "armed for remap (TMS_REMAP_TAGS), so its pages cannot be named");
    }
    if (dst_order.empty()) {
        return fail(-2, "tag '" + dst_tag + "' has no page-granular allocation -- it was never "
                        "armed for remap (TMS_REMAP_TAGS), so its pages cannot be named");
    }

    // ---- VALIDATE THE WHOLE BATCH BEFORE MOVING ANY PAGE -------------------
    // A half-performed remap leaves pages owned by neither arena, and nothing
    // above this layer can repair that.  So the loop runs twice.
    std::vector<void*> sp(n_pages), dp(n_pages);
    std::vector<uint64_t> sl(n_pages), dl(n_pages);
    for (uint64_t k = 0; k < n_pages; ++k) {
        if (!locate_page(src_order, src_page + k, &sp[k], &sl[k])) {
            return fail(-3, "src page " + std::to_string(src_page + k) + " is past the end of tag '"
                            + src_tag + "'");
        }
        if (!locate_page(dst_order, dst_page + k, &dp[k], &dl[k])) {
            return fail(-3, "dst page " + std::to_string(dst_page + k) + " is past the end of tag '"
                            + dst_tag + "'");
        }
        AllocationMetadata& sm = allocation_metadata_[sp[k]];
        AllocationMetadata& dm = allocation_metadata_[dp[k]];
        if (sm.device != dm.device) {
            return fail(-4, "src page " + std::to_string(src_page + k) + " is on device "
                            + std::to_string((int) sm.device) + " and dst page "
                            + std::to_string(dst_page + k) + " on device "
                            + std::to_string((int) dm.device)
                            + " -- a physical page cannot change card, only VA");
        }
        if (sm.page_handles[sl[k]] == 0) {
            return fail(-5, "src page " + std::to_string(src_page + k) + " of tag '" + src_tag
                            + "' holds no physical page (already moved, or the tag is asleep) "
                              "-- refusing rather than allocating one");
        }
        if (dm.page_handles[dl[k]] != 0) {
            return fail(-6, "dst page " + std::to_string(dst_page + k) + " of tag '" + dst_tag
                            + "' is already backed -- mapping over it would strand the page "
                              "it already owns");
        }
    }

    // ---- PERFORM ------------------------------------------------------------
    for (uint64_t k = 0; k < n_pages; ++k) {
        AllocationMetadata& sm = allocation_metadata_[sp[k]];
        AllocationMetadata& dm = allocation_metadata_[dp[k]];
        const CUmemGenericAllocationHandle h = sm.page_handles[sl[k]];
        CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ((char*) sp[k] + sl[k] * page_bytes_), page_bytes_));
        CURESULT_CHECK(cuMemMap((CUdeviceptr) ((char*) dp[k] + dl[k] * page_bytes_),
                                page_bytes_, 0, h, 0));
        sm.page_handles[sl[k]] = 0;
        dm.page_handles[dl[k]] = h;
    }
    // Accessibility is granted per DESTINATION ALLOCATION, once, over its whole
    // reserved range -- a VA-range call, so a thousand moved pages landing in
    // one allocation cost one call and not a thousand.
    std::vector<void*> touched;
    for (uint64_t k = 0; k < n_pages; ++k) {
        if (std::find(touched.begin(), touched.end(), dp[k]) == touched.end()) {
            touched.push_back(dp[k]);
        }
    }
    for (size_t m = 0; m < touched.size(); ++m) {
        AllocationMetadata& dm = allocation_metadata_[touched[m]];
        CUDAUtils::cu_mem_set_access(touched[m], dm.reserved_size, dm.device);
    }
    pages_mapped_out_[src_tag] += n_pages;
    pages_mapped_in_[dst_tag] += n_pages;
    return 0;
#else
    (void) src_tag; (void) src_page; (void) dst_tag; (void) dst_page; (void) n_pages;
    if (err != nullptr && errlen > 0) {
        const char* msg = "W95 Weg2RemapPageRefused: the page remap is CUDA-only";
        size_t n = strlen(msg) < (errlen - 1) ? strlen(msg) : (errlen - 1);
        memcpy(err, msg, n);
        err[n] = '\0';
    }
    return -7;
#endif
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

        if (!metadata.page_handles.empty()) {
            // REMAP: page-granular sleep.  Each page is unmapped and released
            // on its own, and its slot ZEROED -- that zero is what ``resume``
            // reads as "this page must be created" and what a remapped-in page
            // is NOT, which is the entire funding mechanism.
            for (size_t i = 0; i < metadata.page_handles.size(); ++i) {
                if (metadata.page_handles[i] == 0) {
                    continue;
                }
                CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ((char*) ptr + i * page_bytes_), page_bytes_));
                CURESULT_CHECK(cuMemRelease(metadata.page_handles[i]));
                metadata.page_handles[i] = 0;
            }
        } else {
            CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
            CURESULT_CHECK(cuMemRelease(metadata.allocHandle));
        }

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
    uint64_t created_this_wake = 0;                              // REMAP (#1352)
    for (size_t m = 0; m < matched_ptrs.size(); ++m) {
        void* ptr = matched_ptrs[m];
        AllocationMetadata& metadata = allocation_metadata_[ptr];

        if (!metadata.page_handles.empty()) {
            // REMAP: FILL ONLY THE HOLES.  A page that arrived by
            // ``remap_pages`` is already non-zero and already mapped, so this
            // loop does not touch it -- which is exactly how a remap FUNDS a
            // wake instead of a counter promising that it will.  The design's
            // claim ("the flip asks the driver for no page") is therefore not
            // an assertion anywhere: it is ``created_this_wake == 0``, and the
            // number is published by ``page_stats`` whether it is zero or not.
            for (size_t i = 0; i < metadata.page_handles.size(); ++i) {
                if (metadata.page_handles[i] != 0) {
                    continue;
                }
                CUDAUtils::cu_mem_create(&metadata.page_handles[i], page_bytes_,
                                         metadata.device, remap_exportable_);
                CURESULT_CHECK(cuMemMap((CUdeviceptr) ((char*) ptr + i * page_bytes_),
                                        page_bytes_, 0, metadata.page_handles[i], 0));
                ++created_this_wake;
            }
            CUDAUtils::cu_mem_set_access(ptr, metadata.reserved_size, metadata.device);
            metadata.state = AllocationState::ACTIVE;
            continue;
        }

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
    pages_created_[tag] = created_this_wake;                     // REMAP (#1352)

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
