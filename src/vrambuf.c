#include "vrambuf.h"
#include "thread-plat.h"

#if defined(__HIP_PLATFORM_AMD__) && !defined(_WIN32)
#  define VRAM_CHUNK_SIZE      CUDA_PAGE_SIZE
#else
#  define VRAM_CHUNK_SIZE      (16ULL * 1024 * 1024)
#endif

/* ROCm/Windows mitigation: a HIP-runtime VMM defect faults after a large VA
 * window is repeatedly reserved+freed per node. We keep the reservation alive
 * and reuse it per (device, max_size); physical VRAM is still released on
 * destroy, only the reserve/free pair is elided. */
#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
static VramBuffer *g_va_pool;
static Mutex g_va_pool_lock;
#endif

SHARED_EXPORT
void *vrambuf_create(int device, size_t max_size) {
    VramBuffer *buf;

    if (!set_devctx_for_device(device)) {
        return NULL;
    }

    max_size = CUDA_ALIGN_UP(max_size);

#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
    if (!g_va_pool_lock) {
        g_va_pool_lock = mutex_create();
    }
    mutex_lock(g_va_pool_lock);
    for (VramBuffer **p = &g_va_pool; *p; p = &(*p)->next) {
        if ((*p)->device == device && (*p)->max_size == max_size) {
            buf = *p;
            *p = buf->next;
            mutex_unlock(g_va_pool_lock);
            return (void *)buf;
        }
    }
    mutex_unlock(g_va_pool_lock);
#endif

    buf = (VramBuffer *)calloc(1, sizeof(*buf) + sizeof(CUmemGenericAllocationHandle) * max_size / VRAM_CHUNK_SIZE);
    if (!buf) {
        return NULL;
    }
    buf->device = device;
    buf->max_size = max_size;

    if (!CHECK_CU(cuMemAddressReserve(&buf->base_ptr, max_size, 0, 0, 0))) {
        log(ERROR, "%s: %d %zuk\n", __func__, device, max_size / K);
        free(buf);
        return NULL;
    }

    return (void *)buf;
}

SHARED_EXPORT
bool vrambuf_grow(void *arg, size_t required_size) {
    VramBuffer *buf = (VramBuffer *)arg;
    size_t grow_to;
    CUmemGenericAllocationHandle handle;
    CUresult err;

    if (!buf) {
        return false;
    }
    if (!set_devctx_for_device(buf->device)) {
        return false;
    }
    if (required_size > buf->max_size) {
        return false;
    }
    if (required_size <= buf->allocated) {
        return true;
    }

    grow_to = ALIGN_UP(required_size, VRAM_CHUNK_SIZE);
    if (grow_to > buf->max_size) {
        grow_to = buf->max_size;
    }

    vbars_free(budget_deficit(grow_to - buf->allocated));
    while (buf->allocated < grow_to) {
        size_t to_allocate = grow_to - buf->allocated;
        if (to_allocate > VRAM_CHUNK_SIZE) {
            to_allocate = VRAM_CHUNK_SIZE;
        }
        if ((err = three_stooges(buf->base_ptr + buf->allocated, to_allocate, buf->device, &handle)) != CUDA_SUCCESS) {
            if (err != CUDA_ERROR_OUT_OF_MEMORY) {
                log(ERROR, "VRAM Allocation failed (non OOM)\n");
                return false;
            }
            log(DEBUG, "Pytorch allocator attempt exceeds available VRAM ...\n");
            vbars_free(VRAM_CHUNK_SIZE);
            if ((err = three_stooges(buf->base_ptr + buf->allocated, to_allocate, buf->device, &handle)) != CUDA_SUCCESS) {
                bool is_oom = err == CUDA_ERROR_OUT_OF_MEMORY;
                log(is_oom ? INFO : ERROR, "VRAM Allocation failed (%s)\n", is_oom ? "OOM" : "error");
                return false;
            }
        }

        buf->handles[buf->handle_count++] = handle;
        buf->allocated += to_allocate;
    }

    return true;
}

SHARED_EXPORT
CUdeviceptr vrambuf_get(void *arg) {
    VramBuffer *buf = (VramBuffer *)arg;

    if (!buf) {
        return 0;
    }
    return buf->base_ptr;
}

SHARED_EXPORT
bool vrambuf_destroy(void *arg) {
    VramBuffer *buf = (VramBuffer *)arg;
    size_t i;

    if (!buf || !set_devctx_for_device(buf->device)) {
        return false;
    }

    if (buf->allocated > 0) {
        CHECK_CU(cuMemUnmap(buf->base_ptr, buf->allocated));
        unmap_workaround(buf->base_ptr, buf->allocated);
    }

    for (i = 0; i < buf->handle_count; i++) {
        CHECK_CU(cuMemRelease(buf->handles[i]));
    }

    total_vram_usage -= buf->allocated;

#if defined(__HIP_PLATFORM_AMD__) && defined(_WIN32)
    /* VRAM freed; keep the VA reservation and park it for reuse. */
    buf->allocated = 0;
    buf->handle_count = 0;
    mutex_lock(g_va_pool_lock);
    buf->next = g_va_pool;
    g_va_pool = buf;
    mutex_unlock(g_va_pool_lock);
    return true;
#else
    CHECK_CU(cuMemAddressFree(buf->base_ptr, buf->max_size));
    free(buf);
    return true;
#endif
}
