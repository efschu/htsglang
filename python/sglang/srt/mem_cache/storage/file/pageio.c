/* #1402: batched page I/O for the HiCache file backend.
 *
 * WHY C. Every os.* call in Python releases and re-acquires the GIL, and in a
 * scheduler process whose main thread is busy the re-acquire waits up to the
 * interpreter's switch interval (5 ms by default). Measured 2026-09-15 against
 * boot xsn133's store: a page read of six syscalls costs 13 us alone and
 * 20.6 ms beside a busy Python thread. One ctypes call releases the GIL ONCE
 * for a whole batch, so 128 pages cost one hand-off instead of ~800.
 *
 * Both entry points are plain loops over plain syscalls; nothing here decides
 * anything about keys, windows or readability -- that stays in Python.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/* sizes[i] = st_size of paths[i], or -1 when it cannot be stat'ed.
 * Returns the number of paths found. */
int64_t hicache_stat_sizes(int64_t n, const char **paths, int64_t *sizes) {
    struct stat st;
    int64_t found = 0;
    for (int64_t i = 0; i < n; i++) {
        if (stat(paths[i], &st) == 0) {
            sizes[i] = (int64_t)st.st_size;
            found++;
        } else {
            sizes[i] = -1;
        }
    }
    return found;
}

/* Read the extents of n canonical blobs into n caller buffers.
 *
 * For page i: expect_total[i] >= 0 demands that exact file size (a blob of
 * another width is refused, status 2); n_ext[i] extents follow in ext_off /
 * ext_len (flattened over all pages, in page order) and land back to back in
 * out[i]. touch != 0 bumps the mtime through the open fd (the LRU recency the
 * evictor's sibling owner reads). status[i]: 0 ok, 1 missing, 2 size
 * mismatch, 3 short read, 4 other error. Returns the number of ok pages. */
int64_t hicache_read_pages(int64_t n, const char **paths, const int64_t *expect_total,
                           const int64_t *n_ext, const int64_t *ext_off,
                           const int64_t *ext_len, uint8_t **out, int32_t touch,
                           int8_t *status) {
    int64_t ok = 0, e = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t k = n_ext[i];
        int8_t s = 0;
        int fd = open(paths[i], O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            s = (errno == ENOENT) ? 1 : 4;
        } else {
            struct stat st;
            if (fstat(fd, &st) != 0) {
                s = 4;
            } else if (expect_total[i] >= 0 && (int64_t)st.st_size != expect_total[i]) {
                s = 2;
            } else {
                int64_t taken = 0;
                for (int64_t j = 0; j < k && s == 0; j++) {
                    int64_t off = ext_off[e + j], len = ext_len[e + j], got = 0;
                    while (got < len) {
                        ssize_t r = pread(fd, out[i] + taken + got, (size_t)(len - got),
                                          (off_t)(off + got));
                        if (r < 0) {
                            if (errno == EINTR) continue;
                            s = 4;
                            break;
                        }
                        if (r == 0) {
                            s = 3;
                            break;
                        }
                        got += r;
                    }
                    taken += len;
                }
                if (s == 0 && touch) {
                    struct timespec ts[2] = {{0, UTIME_NOW}, {0, UTIME_NOW}};
                    (void)futimens(fd, ts);
                }
            }
            close(fd);
        }
        status[i] = s;
        if (s == 0) ok++;
        e += k;
    }
    return ok;
}
