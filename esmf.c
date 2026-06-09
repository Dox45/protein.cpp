// /*
//  * esmf.c  —  ESMF v0 format implementation
//  *
//  * Self-contained C99 implementation of the ESMF (ESM Format) binary model
//  * format.  Covers:
//  *
//  *   - Memory-mapped file loading (mmap on POSIX, MapViewOfFile on Windows)
//  *   - Header / metadata / directory parsing
//  *   - Tensor lookup by name (linear scan; small n makes a hash overkill)
//  *   - Safe bounds-checked data pointer retrieval
//  *   - Resource cleanup
//  *
//  * Build (standalone test):
//  *   cc -std=c99 -O2 -DESMF_TEST -o esmf_test esmf.c -lm && ./esmf_test foo.esmf
//  *
//  * Include as a translation unit in larger projects:
//  *   #include "esmf.h"   (auto-generated companion header shown at the bottom)
//  *   ... compile esmf.c alongside your other .c files ...
//  *
//  * Format spec (v0)
//  * ────────────────
//  *   Offset  Size  Field
//  *   0       4     Magic: "ESMF"
//  *   4       4     Version: uint32_le  (must be 1 for v0)
//  *   8       8     metadata_size: uint64_le
//  *   16      var   metadata JSON (UTF-8, not NUL-terminated)
//  *   16+M    4     tensor_count: uint32_le
//  *   20+M    n*120 tensor directory (n entries × 120 bytes each)
//  *   dir_end pad   alignment padding to 64-byte boundary
//  *   blob_start var tensor blob
//  *
//  * Directory entry (120 bytes, little-endian):
//  *   0   64   name (UTF-8, NUL-padded)
//  *   64   4   dtype: uint32  (0=f32, 1=f16, 2=bf16, 3=q4_0, 4=q8_0)
//  *   68   4   ndim: uint32
//  *   72  32   shape[4]: int64_le  (unused dims = 0)
//  *   104  8   data_offset: uint64_le  (offset within blob)
//  *   112  8   data_size: uint64_le   (bytes)
//  *
//  * All multi-byte integers are little-endian.
//  */

// #define _POSIX_C_SOURCE 200809L   /* strnlen, mmap, etc. */

// #include <stdarg.h>
// #include <stddef.h>
// #include <stdint.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <errno.h>

// /* ── Platform-specific memory-mapping ─────────────────────────────────────── */

// #if defined(_WIN32) || defined(_WIN64)
// #  define ESMF_WINDOWS
// #  include <windows.h>
// #else
// #  define ESMF_POSIX
// #  include <fcntl.h>
// #  include <sys/mman.h>
// #  include <sys/stat.h>
// #  include <unistd.h>
// #endif

// /* ── Public types ─────────────────────────────────────────────────────────── */

// #define ESMF_MAGIC_U32   0x464D5345u   /* "ESMF" little-endian */
// #define ESMF_VERSION     1u

// #define ESMF_DTYPE_F32   0u
// #define ESMF_DTYPE_F16   1u
// #define ESMF_DTYPE_BF16  2u
// #define ESMF_DTYPE_Q4_0  3u
// #define ESMF_DTYPE_Q8_0  4u

// #define ESMF_MAX_DIMS       4
// #define ESMF_NAME_MAXLEN    64
// #define ESMF_BLOB_ALIGN     64
// #define ESMF_ENTRY_SIZE     120   /* 64 + 4 + 4 + 32 + 8 + 8 */

// typedef struct {
//     char     name[ESMF_NAME_MAXLEN + 1];  /* +1 for guaranteed NUL */
//     uint32_t dtype;
//     uint32_t ndim;
//     int64_t  shape[ESMF_MAX_DIMS];
//     uint64_t data_offset;   /* offset within blob section */
//     uint64_t data_size;     /* bytes */
// } esmf_tensor_t;

// typedef struct {
//     /* Raw mapped region */
//     const uint8_t *base;
//     size_t         file_size;

//     /* Parsed header */
//     uint32_t version;

//     /* Metadata JSON (points into mapped region, NOT NUL-terminated) */
//     const char *metadata_json;
//     uint64_t    metadata_size;

//     /* Tensor directory (heap-allocated, fully decoded) */
//     uint32_t       tensor_count;
//     esmf_tensor_t *tensors;

//     /* Pointer to start of blob within mapped region */
//     const uint8_t *blob;
//     uint64_t       blob_size;   /* remaining bytes from blob_start to EOF */

//     /* Error string (static or heap; see esmf_last_error) */
//     char err[256];

//     /* Platform handles for cleanup */
// #if defined(ESMF_WINDOWS)
//     HANDLE hFile;
//     HANDLE hMap;
// #else
//     int fd;
// #endif
// } esmf_file_t;

// /* ── Error helpers ────────────────────────────────────────────────────────── */

// static void esmf_set_err(esmf_file_t *ef, const char *msg)
// {
//     snprintf(ef->err, sizeof(ef->err), "%s", msg);
// }

// static void esmf_set_errf(esmf_file_t *ef, const char *fmt, ...)
// {
//     va_list ap;
//     va_start(ap, fmt);
//     vsnprintf(ef->err, sizeof(ef->err), fmt, ap);
//     va_end(ap);
// }

// const char *esmf_last_error(const esmf_file_t *ef)
// {
//     return ef ? ef->err : "null esmf_file_t";
// }

// /* ── Little-endian read helpers ───────────────────────────────────────────── */

// static inline uint32_t read_u32le(const uint8_t *p)
// {
//     return (uint32_t)p[0]
//          | (uint32_t)p[1] <<  8
//          | (uint32_t)p[2] << 16
//          | (uint32_t)p[3] << 24;
// }

// static inline uint64_t read_u64le(const uint8_t *p)
// {
//     return (uint64_t)p[0]
//          | (uint64_t)p[1] <<  8
//          | (uint64_t)p[2] << 16
//          | (uint64_t)p[3] << 24
//          | (uint64_t)p[4] << 32
//          | (uint64_t)p[5] << 40
//          | (uint64_t)p[6] << 48
//          | (uint64_t)p[7] << 56;
// }

// static inline int64_t read_i64le(const uint8_t *p)
// {
//     return (int64_t)read_u64le(p);
// }

// /* ── Bounds check ─────────────────────────────────────────────────────────── */

// /*
//  * Returns non-zero if [offset, offset+len) is fully within the mapped region.
//  * Uses size_t arithmetic to avoid signed overflow.
//  */
// static int in_bounds(const esmf_file_t *ef, size_t offset, size_t len)
// {
//     if (offset > ef->file_size) return 0;
//     if (len > ef->file_size - offset) return 0;
//     return 1;
// }

// /* ── mmap / munmap wrappers ───────────────────────────────────────────────── */

// static int esmf_map(esmf_file_t *ef, const char *path)
// {
// #if defined(ESMF_POSIX)
//     struct stat st;

//     ef->fd = open(path, O_RDONLY);
//     if (ef->fd < 0) {
//         esmf_set_errf(ef, "open(%s): %s", path, strerror(errno));
//         return -1;
//     }
//     if (fstat(ef->fd, &st) < 0) {
//         esmf_set_errf(ef, "fstat: %s", strerror(errno));
//         close(ef->fd);
//         ef->fd = -1;
//         return -1;
//     }
//     ef->file_size = (size_t)st.st_size;
//     if (ef->file_size == 0) {
//         esmf_set_err(ef, "file is empty");
//         close(ef->fd);
//         ef->fd = -1;
//         return -1;
//     }
//     void *ptr = mmap(NULL, ef->file_size, PROT_READ, MAP_PRIVATE, ef->fd, 0);
//     if (ptr == MAP_FAILED) {
//         esmf_set_errf(ef, "mmap: %s", strerror(errno));
//         close(ef->fd);
//         ef->fd = -1;
//         return -1;
//     }
//     ef->base = (const uint8_t *)ptr;
//     return 0;

// #elif defined(ESMF_WINDOWS)
//     /* Convert to wide string for CreateFileW */
//     int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
//     if (wlen <= 0) {
//         esmf_set_err(ef, "MultiByteToWideChar failed");
//         return -1;
//     }
//     wchar_t *wpath = (wchar_t *)malloc(wlen * sizeof(wchar_t));
//     if (!wpath) { esmf_set_err(ef, "OOM"); return -1; }
//     MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen);

//     ef->hFile = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ,
//                             NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
//     free(wpath);
//     if (ef->hFile == INVALID_HANDLE_VALUE) {
//         esmf_set_errf(ef, "CreateFileW: error %lu", GetLastError());
//         return -1;
//     }

//     LARGE_INTEGER sz;
//     if (!GetFileSizeEx(ef->hFile, &sz)) {
//         esmf_set_errf(ef, "GetFileSizeEx: error %lu", GetLastError());
//         CloseHandle(ef->hFile);
//         ef->hFile = INVALID_HANDLE_VALUE;
//         return -1;
//     }
//     ef->file_size = (size_t)sz.QuadPart;

//     ef->hMap = CreateFileMappingW(ef->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
//     if (!ef->hMap) {
//         esmf_set_errf(ef, "CreateFileMappingW: error %lu", GetLastError());
//         CloseHandle(ef->hFile);
//         ef->hFile = INVALID_HANDLE_VALUE;
//         return -1;
//     }

//     void *ptr = MapViewOfFile(ef->hMap, FILE_MAP_READ, 0, 0, 0);
//     if (!ptr) {
//         esmf_set_errf(ef, "MapViewOfFile: error %lu", GetLastError());
//         CloseHandle(ef->hMap);
//         CloseHandle(ef->hFile);
//         ef->hMap  = NULL;
//         ef->hFile = INVALID_HANDLE_VALUE;
//         return -1;
//     }
//     ef->base = (const uint8_t *)ptr;
//     return 0;
// #endif
// }

// static void esmf_unmap(esmf_file_t *ef)
// {
//     if (!ef->base) return;
// #if defined(ESMF_POSIX)
//     munmap((void *)ef->base, ef->file_size);
//     ef->base = NULL;
//     if (ef->fd >= 0) { close(ef->fd); ef->fd = -1; }
// #elif defined(ESMF_WINDOWS)
//     UnmapViewOfFile((LPCVOID)ef->base);
//     ef->base = NULL;
//     if (ef->hMap)  { CloseHandle(ef->hMap);  ef->hMap  = NULL; }
//     if (ef->hFile != INVALID_HANDLE_VALUE) {
//         CloseHandle(ef->hFile);
//         ef->hFile = INVALID_HANDLE_VALUE;
//     }
// #endif
// }

// /* ── Directory parser ─────────────────────────────────────────────────────── */

// /*
//  * Parse one 120-byte directory entry starting at `p`.
//  * Caller must guarantee p[0..119] is readable.
//  */
// static void parse_entry(const uint8_t *p, esmf_tensor_t *out)
// {
//     /* name: 64 bytes, NUL-padded */
//     memcpy(out->name, p, ESMF_NAME_MAXLEN);
//     out->name[ESMF_NAME_MAXLEN] = '\0';
//     /* Trim any embedded NULs that strnlen would stop at */
//     out->name[strnlen(out->name, ESMF_NAME_MAXLEN)] = '\0';

//     out->dtype       = read_u32le(p + 64);
//     out->ndim        = read_u32le(p + 68);
//     for (int d = 0; d < ESMF_MAX_DIMS; d++)
//         out->shape[d] = read_i64le(p + 72 + d * 8);
//     out->data_offset = read_u64le(p + 104);
//     out->data_size   = read_u64le(p + 112);
// }

// /* ── Public API ───────────────────────────────────────────────────────────── */

// /*
//  * esmf_open  —  map and parse an ESMF file.
//  *
//  * Returns a heap-allocated esmf_file_t on success, NULL on failure.
//  * On failure, a static error string is printed to stderr.
//  * On success, call esmf_close() when done.
//  */
// esmf_file_t *esmf_open(const char *path)
// {
//     esmf_file_t *ef = (esmf_file_t *)calloc(1, sizeof(esmf_file_t));
//     if (!ef) {
//         fprintf(stderr, "esmf_open: out of memory\n");
//         return NULL;
//     }

// #if defined(ESMF_POSIX)
//     ef->fd = -1;
// #elif defined(ESMF_WINDOWS)
//     ef->hFile = INVALID_HANDLE_VALUE;
// #endif

//     /* 1. Map file */
//     if (esmf_map(ef, path) != 0) {
//         fprintf(stderr, "esmf_open: %s\n", ef->err);
//         free(ef);
//         return NULL;
//     }

//     /* Minimum viable file: 4+4+8 = 16 bytes for fixed header */
//     if (ef->file_size < 16) {
//         esmf_set_err(ef, "file too small to be ESMF");
//         goto fail;
//     }

//     /* 2. Magic */
//     uint32_t magic = read_u32le(ef->base + 0);
//     if (magic != ESMF_MAGIC_U32) {
//         esmf_set_errf(ef, "bad magic: 0x%08X (expected 0x%08X)",
//                       magic, ESMF_MAGIC_U32);
//         goto fail;
//     }

//     /* 3. Version */
//     ef->version = read_u32le(ef->base + 4);
//     if (ef->version != ESMF_VERSION) {
//         esmf_set_errf(ef, "unsupported ESMF version %u (only v%u supported)",
//                       ef->version, ESMF_VERSION);
//         goto fail;
//     }

//     /* 4. Metadata JSON */
//     ef->metadata_size = read_u64le(ef->base + 8);
//     size_t meta_start = 16;
//     if (!in_bounds(ef, meta_start, (size_t)ef->metadata_size)) {
//         esmf_set_err(ef, "metadata_size overflows file");
//         goto fail;
//     }
//     ef->metadata_json = (const char *)(ef->base + meta_start);

//     /* 5. Tensor count */
//     size_t count_pos = meta_start + (size_t)ef->metadata_size;
//     if (!in_bounds(ef, count_pos, 4)) {
//         esmf_set_err(ef, "truncated at tensor_count field");
//         goto fail;
//     }
//     ef->tensor_count = read_u32le(ef->base + count_pos);

//     /* 6. Directory */
//     size_t dir_start = count_pos + 4;
//     size_t dir_bytes = (size_t)ef->tensor_count * ESMF_ENTRY_SIZE;
//     if (!in_bounds(ef, dir_start, dir_bytes)) {
//         esmf_set_errf(ef, "tensor directory overflows file "
//                       "(count=%u dir_start=%zu dir_bytes=%zu file=%zu)",
//                       ef->tensor_count, dir_start, dir_bytes, ef->file_size);
//         goto fail;
//     }

//     ef->tensors = (esmf_tensor_t *)calloc(ef->tensor_count, sizeof(esmf_tensor_t));
//     if (!ef->tensors) {
//         esmf_set_err(ef, "out of memory allocating tensor directory");
//         goto fail;
//     }

//     for (uint32_t i = 0; i < ef->tensor_count; i++) {
//         parse_entry(ef->base + dir_start + (size_t)i * ESMF_ENTRY_SIZE,
//                     &ef->tensors[i]);

//         /* Validate ndim */
//         if (ef->tensors[i].ndim > ESMF_MAX_DIMS) {
//             esmf_set_errf(ef, "tensor[%u] '%s': ndim=%u exceeds MAX_DIMS=%d",
//                           i, ef->tensors[i].name,
//                           ef->tensors[i].ndim, ESMF_MAX_DIMS);
//             goto fail;
//         }
//     }

//     /* 7. Blob start (align dir_end to BLOB_ALIGN) */
//     size_t dir_end  = dir_start + dir_bytes;
//     size_t remainder = dir_end % ESMF_BLOB_ALIGN;
//     size_t blob_start = remainder == 0
//                         ? dir_end
//                         : dir_end + (ESMF_BLOB_ALIGN - remainder);

//     if (blob_start > ef->file_size) {
//         esmf_set_err(ef, "blob_start past end of file (truncated?)");
//         goto fail;
//     }

//     ef->blob      = ef->base + blob_start;
//     ef->blob_size = ef->file_size - blob_start;

//     /* 8. Cross-validate each tensor's data range against blob_size */
//     for (uint32_t i = 0; i < ef->tensor_count; i++) {
//         const esmf_tensor_t *t = &ef->tensors[i];
//         uint64_t end = t->data_offset + t->data_size;
//         if (end < t->data_offset /* overflow */ || end > ef->blob_size) {
//             esmf_set_errf(ef,
//                 "tensor '%s': data_offset=%llu + data_size=%llu = %llu "
//                 "exceeds blob_size=%llu",
//                 t->name,
//                 (unsigned long long)t->data_offset,
//                 (unsigned long long)t->data_size,
//                 (unsigned long long)end,
//                 (unsigned long long)ef->blob_size);
//             goto fail;
//         }
//     }

//     return ef;

// fail:
//     fprintf(stderr, "esmf_open(%s): %s\n", path, ef->err);
//     esmf_unmap(ef);
//     free(ef->tensors);
//     free(ef);
//     return NULL;
// }

// /*
//  * esmf_close  —  release all resources held by ef.
//  */
// void esmf_close(esmf_file_t *ef)
// {
//     if (!ef) return;
//     esmf_unmap(ef);
//     free(ef->tensors);
//     free(ef);
// }

// /*
//  * esmf_find  —  look up a tensor by ESMF name.
//  *
//  * Returns a pointer into ef->tensors on success, NULL if not found.
//  * The returned pointer is valid for the lifetime of ef.
//  */
// const esmf_tensor_t *esmf_find(const esmf_file_t *ef, const char *name)
// {
//     for (uint32_t i = 0; i < ef->tensor_count; i++) {
//         if (strcmp(ef->tensors[i].name, name) == 0)
//             return &ef->tensors[i];
//     }
//     return NULL;
// }

// /*
//  * esmf_data  —  get a bounds-checked pointer to a tensor's raw bytes.
//  *
//  * Returns a pointer into the memory-mapped blob on success.
//  * Returns NULL if tensor is NULL or its range is invalid (should never happen
//  * after esmf_open succeeds, but checked defensively).
//  *
//  * The pointer is valid (read-only) as long as ef is open.
//  */
// const void *esmf_data(const esmf_file_t *ef, const esmf_tensor_t *tensor)
// {
//     if (!ef || !tensor) return NULL;
//     if (tensor->data_offset + tensor->data_size > ef->blob_size) return NULL;
//     return (const void *)(ef->blob + tensor->data_offset);
// }

// /*
//  * esmf_dtype_size  —  bytes per element for a given dtype enum.
//  *
//  * Returns 0 for unknown dtypes.
//  */
// size_t esmf_dtype_size(uint32_t dtype)
// {
//     switch (dtype) {
//         case ESMF_DTYPE_F32:  return 4;
//         case ESMF_DTYPE_F16:  return 2;
//         case ESMF_DTYPE_BF16: return 2;
//         case ESMF_DTYPE_Q4_0: return 0;  /* variable-width; caller handles */
//         case ESMF_DTYPE_Q8_0: return 1;
//         default:              return 0;
//     }
// }

// /*
//  * esmf_dtype_name  —  human-readable dtype string.
//  */
// const char *esmf_dtype_name(uint32_t dtype)
// {
//     switch (dtype) {
//         case ESMF_DTYPE_F32:  return "f32";
//         case ESMF_DTYPE_F16:  return "f16";
//         case ESMF_DTYPE_BF16: return "bf16";
//         case ESMF_DTYPE_Q4_0: return "q4_0";
//         case ESMF_DTYPE_Q8_0: return "q8_0";
//         default:              return "unknown";
//     }
// }

// /*
//  * esmf_tensor_nelems  —  total element count for a tensor.
//  *
//  * Returns 0 for zero-dim or zero-size tensors.
//  */
// uint64_t esmf_tensor_nelems(const esmf_tensor_t *t)
// {
//     if (!t || t->ndim == 0) return 0;
//     uint64_t n = 1;
//     for (uint32_t d = 0; d < t->ndim; d++) {
//         if (t->shape[d] <= 0) return 0;
//         n *= (uint64_t)t->shape[d];
//     }
//     return n;
// }

// /*
//  * esmf_metadata_copy  —  copy the metadata JSON into a caller-supplied buffer.
//  *
//  * Writes at most buf_size-1 bytes and always NUL-terminates buf.
//  * Returns the number of bytes written (excluding NUL), or (size_t)-1 if
//  * buf_size is 0.
//  */
// size_t esmf_metadata_copy(const esmf_file_t *ef, char *buf, size_t buf_size)
// {
//     if (!ef || !buf || buf_size == 0) return (size_t)-1;
//     size_t n = (size_t)ef->metadata_size;
//     if (n >= buf_size) n = buf_size - 1;
//     memcpy(buf, ef->metadata_json, n);
//     buf[n] = '\0';
//     return n;
// }

// /*
//  * esmf_print_summary  —  dump a human-readable summary to fp.
//  */
// void esmf_print_summary(const esmf_file_t *ef, FILE *fp)
// {
//     if (!ef || !fp) return;
//     fprintf(fp, "ESMF v%u  tensors=%u  blob=%llu bytes\n",
//             ef->version, ef->tensor_count,
//             (unsigned long long)ef->blob_size);
//     fprintf(fp, "Metadata (%llu bytes):\n",
//             (unsigned long long)ef->metadata_size);

//     /* Print metadata JSON with a trailing newline if absent */
//     fwrite(ef->metadata_json, 1, (size_t)ef->metadata_size, fp);
//     fputc('\n', fp);

//     fprintf(fp, "\nTensor directory:\n");
//     fprintf(fp, "  %-42s  %-6s  %-5s  %s\n",
//             "name", "dtype", "ndim", "shape");
//     fprintf(fp, "  %s\n", "----------------------------------------------"
//                            "---------------------------------");
//     for (uint32_t i = 0; i < ef->tensor_count; i++) {
//         const esmf_tensor_t *t = &ef->tensors[i];
//         fprintf(fp, "  %-42s  %-6s  %u     [",
//                 t->name, esmf_dtype_name(t->dtype), t->ndim);
//         for (uint32_t d = 0; d < t->ndim; d++) {
//             if (d) fprintf(fp, ", ");
//             fprintf(fp, "%lld", (long long)t->shape[d]);
//         }
//         fprintf(fp, "]  (%llu bytes @ offset %llu)\n",
//                 (unsigned long long)t->data_size,
//                 (unsigned long long)t->data_offset);
//     }
// }

// /* ── Optional standalone test driver ─────────────────────────────────────── */

// #ifdef ESMF_TEST

// int main(int argc, char **argv)
// {
//     if (argc < 2) {
//         fprintf(stderr, "Usage: %s <file.esmf> [tensor_name]\n", argv[0]);
//         return 1;
//     }

//     esmf_file_t *ef = esmf_open(argv[1]);
//     if (!ef) return 1;

//     esmf_print_summary(ef, stdout);

//     if (argc >= 3) {
//         const char *name = argv[2];
//         const esmf_tensor_t *t = esmf_find(ef, name);
//         if (!t) {
//             fprintf(stderr, "\nTensor '%s' not found.\n", name);
//         } else {
//             const void *data = esmf_data(ef, t);
//             fprintf(stdout, "\nTensor '%s':\n", name);
//             fprintf(stdout, "  dtype:    %s\n", esmf_dtype_name(t->dtype));
//             fprintf(stdout, "  ndim:     %u\n", t->ndim);
//             fprintf(stdout, "  nelems:   %llu\n",
//                     (unsigned long long)esmf_tensor_nelems(t));
//             fprintf(stdout, "  data_ptr: %p\n", data);
//             /* Print first few f32 values as a sanity check */
//             if (t->dtype == ESMF_DTYPE_F32 && data) {
//                 const float *fp32 = (const float *)data;
//                 uint64_t n = esmf_tensor_nelems(t);
//                 uint64_t show = n < 8 ? n : 8;
//                 fprintf(stdout, "  first %llu values:", (unsigned long long)show);
//                 for (uint64_t k = 0; k < show; k++)
//                     fprintf(stdout, " %.6g", fp32[k]);
//                 fprintf(stdout, "\n");
//             }
//         }
//     }

//     esmf_close(ef);
//     return 0;
// }

// #endif /* ESMF_TEST */

/*
 * esmf.c  —  ESMF v0 format implementation
 *
 * Self-contained C99 implementation of the ESMF (ESM Format) binary model
 * format.  Covers:
 *
 *   - Memory-mapped file loading (mmap on POSIX, MapViewOfFile on Windows)
 *   - Header / metadata / directory parsing
 *   - Tensor lookup by name (linear scan; small n makes a hash overkill)
 *   - Safe bounds-checked data pointer retrieval
 *   - Resource cleanup
 *
 * Build (standalone test):
 *   cc -std=c99 -O2 -DESMF_TEST -o esmf_test esmf.c -lm && ./esmf_test foo.esmf
 *
 * Include as a translation unit in larger projects:
 *   #include "esmf.h"
 *   ... compile esmf.c alongside your other .c files ...
 *
 * Format spec (v0)
 * ────────────────
 *   Offset  Size  Field
 *   0       4     Magic: "ESMF"
 *   4       4     Version: uint32_le  (must be 1 for v0)
 *   8       8     metadata_size: uint64_le
 *   16      var   metadata JSON (UTF-8, not NUL-terminated)
 *   16+M    4     tensor_count: uint32_le
 *   20+M    n*120 tensor directory (n entries × 120 bytes each)
 *   dir_end pad   alignment padding to 64-byte boundary
 *   blob_start var tensor blob
 *
 * Directory entry (120 bytes, little-endian):
 *   0   64   name (UTF-8, NUL-padded)
 *   64   4   dtype: uint32  (0=f32, 1=f16, 2=bf16, 3=q4_0, 4=q8_0)
 *   68   4   ndim: uint32
 *   72  32   shape[4]: int64_le  (unused dims = 0)
 *   104  8   data_offset: uint64_le  (offset within blob)
 *   112  8   data_size: uint64_le   (bytes)
 *
 * All multi-byte integers are little-endian.
 */

#define _POSIX_C_SOURCE 200809L   /* strnlen, mmap, etc. */

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* ── Platform-specific memory-mapping ─────────────────────────────────────── */

#if defined(_WIN32) || defined(_WIN64)
#  define ESMF_WINDOWS
#  include <windows.h>
#else
#  define ESMF_POSIX
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

/* ── Public types ─────────────────────────────────────────────────────────── */

#define ESMF_MAGIC_U32   0x464D5345u   /* "ESMF" little-endian */
#define ESMF_VERSION     1u

#define ESMF_DTYPE_F32   0u
#define ESMF_DTYPE_F16   1u
#define ESMF_DTYPE_BF16  2u
#define ESMF_DTYPE_Q4_0  3u
#define ESMF_DTYPE_Q8_0  4u

#define ESMF_MAX_DIMS       4
#define ESMF_NAME_MAXLEN    64
#define ESMF_BLOB_ALIGN     64
#define ESMF_ENTRY_SIZE     120   /* 64 + 4 + 4 + 32 + 8 + 8 */

typedef struct {
    char     name[ESMF_NAME_MAXLEN + 1];  /* +1 for guaranteed NUL */
    uint32_t dtype;
    uint32_t ndim;
    int64_t  shape[ESMF_MAX_DIMS];
    uint64_t data_offset;   /* offset within blob section */
    uint64_t data_size;     /* bytes */
} esmf_tensor_t;

typedef struct {
    /* Raw mapped region */
    const uint8_t *base;
    size_t         file_size;

    /* Parsed header */
    uint32_t version;

    /* Metadata JSON (points into mapped region, NOT NUL-terminated) */
    const char *metadata_json;
    uint64_t    metadata_size;

    /* Tensor directory (heap-allocated, fully decoded) */
    uint32_t       tensor_count;
    esmf_tensor_t *tensors;

    /* Pointer to start of blob within mapped region */
    const uint8_t *blob;
    uint64_t       blob_size;   /* remaining bytes from blob_start to EOF */

    /* Error string */
    char err[256];

    /* Platform handles for cleanup */
#if defined(ESMF_WINDOWS)
    HANDLE hFile;
    HANDLE hMap;
#else
    int fd;
#endif
} esmf_file_t;

/* ── Error helpers ────────────────────────────────────────────────────────── */

static void esmf_set_err(esmf_file_t *ef, const char *msg)
{
    snprintf(ef->err, sizeof(ef->err), "%s", msg);
}

static void esmf_set_errf(esmf_file_t *ef, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ef->err, sizeof(ef->err), fmt, ap);
    va_end(ap);
}

const char *esmf_last_error(const esmf_file_t *ef)
{
    return ef ? ef->err : "null esmf_file_t";
}

/* ── Little-endian read helpers ───────────────────────────────────────────── */

static inline uint32_t read_u32le(const uint8_t *p)
{
    return (uint32_t)p[0]
         | (uint32_t)p[1] <<  8
         | (uint32_t)p[2] << 16
         | (uint32_t)p[3] << 24;
}

static inline uint64_t read_u64le(const uint8_t *p)
{
    return (uint64_t)p[0]
         | (uint64_t)p[1] <<  8
         | (uint64_t)p[2] << 16
         | (uint64_t)p[3] << 24
         | (uint64_t)p[4] << 32
         | (uint64_t)p[5] << 40
         | (uint64_t)p[6] << 48
         | (uint64_t)p[7] << 56;
}

static inline int64_t read_i64le(const uint8_t *p)
{
    return (int64_t)read_u64le(p);
}

/* ── Bounds check ─────────────────────────────────────────────────────────── */

static int in_bounds(const esmf_file_t *ef, size_t offset, size_t len)
{
    if (offset > ef->file_size) return 0;
    if (len > ef->file_size - offset) return 0;
    return 1;
}

/* ── mmap / munmap wrappers ───────────────────────────────────────────────── */

static int esmf_map(esmf_file_t *ef, const char *path)
{
#if defined(ESMF_POSIX)
    struct stat st;

    ef->fd = open(path, O_RDONLY);
    if (ef->fd < 0) {
        esmf_set_errf(ef, "open(%s): %s", path, strerror(errno));
        return -1;
    }
    if (fstat(ef->fd, &st) < 0) {
        esmf_set_errf(ef, "fstat: %s", strerror(errno));
        close(ef->fd);
        ef->fd = -1;
        return -1;
    }
    ef->file_size = (size_t)st.st_size;
    if (ef->file_size == 0) {
        esmf_set_err(ef, "file is empty");
        close(ef->fd);
        ef->fd = -1;
        return -1;
    }
    {
        void *ptr = mmap(NULL, ef->file_size, PROT_READ, MAP_PRIVATE, ef->fd, 0);
        if (ptr == MAP_FAILED) {
            esmf_set_errf(ef, "mmap: %s", strerror(errno));
            close(ef->fd);
            ef->fd = -1;
            return -1;
        }
        ef->base = (const uint8_t *)ptr;
    }
    return 0;

#elif defined(ESMF_WINDOWS)
    {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
        wchar_t *wpath;
        void *ptr;

        if (wlen <= 0) {
            esmf_set_err(ef, "MultiByteToWideChar failed");
            return -1;
        }
        wpath = (wchar_t *)malloc(wlen * sizeof(wchar_t));
        if (!wpath) { esmf_set_err(ef, "OOM"); return -1; }
        MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wlen);

        ef->hFile = CreateFileW(wpath, GENERIC_READ, FILE_SHARE_READ,
                                NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        free(wpath);
        if (ef->hFile == INVALID_HANDLE_VALUE) {
            esmf_set_errf(ef, "CreateFileW: error %lu", GetLastError());
            return -1;
        }

        {
            LARGE_INTEGER sz;
            if (!GetFileSizeEx(ef->hFile, &sz)) {
                esmf_set_errf(ef, "GetFileSizeEx: error %lu", GetLastError());
                CloseHandle(ef->hFile);
                ef->hFile = INVALID_HANDLE_VALUE;
                return -1;
            }
            ef->file_size = (size_t)sz.QuadPart;
        }

        ef->hMap = CreateFileMappingW(ef->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!ef->hMap) {
            esmf_set_errf(ef, "CreateFileMappingW: error %lu", GetLastError());
            CloseHandle(ef->hFile);
            ef->hFile = INVALID_HANDLE_VALUE;
            return -1;
        }

        ptr = MapViewOfFile(ef->hMap, FILE_MAP_READ, 0, 0, 0);
        if (!ptr) {
            esmf_set_errf(ef, "MapViewOfFile: error %lu", GetLastError());
            CloseHandle(ef->hMap);
            CloseHandle(ef->hFile);
            ef->hMap  = NULL;
            ef->hFile = INVALID_HANDLE_VALUE;
            return -1;
        }
        ef->base = (const uint8_t *)ptr;
    }
    return 0;
#endif
}

static void esmf_unmap(esmf_file_t *ef)
{
    if (!ef->base) return;
#if defined(ESMF_POSIX)
    munmap((void *)ef->base, ef->file_size);
    ef->base = NULL;
    if (ef->fd >= 0) { close(ef->fd); ef->fd = -1; }
#elif defined(ESMF_WINDOWS)
    UnmapViewOfFile((LPCVOID)ef->base);
    ef->base = NULL;
    if (ef->hMap)  { CloseHandle(ef->hMap);  ef->hMap  = NULL; }
    if (ef->hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(ef->hFile);
        ef->hFile = INVALID_HANDLE_VALUE;
    }
#endif
}

/* ── Directory parser ─────────────────────────────────────────────────────── */

static void parse_entry(const uint8_t *p, esmf_tensor_t *out)
{
    int d;
    memcpy(out->name, p, ESMF_NAME_MAXLEN);
    out->name[ESMF_NAME_MAXLEN] = '\0';
    out->name[strnlen(out->name, ESMF_NAME_MAXLEN)] = '\0';

    out->dtype       = read_u32le(p + 64);
    out->ndim        = read_u32le(p + 68);
    for (d = 0; d < ESMF_MAX_DIMS; d++)
        out->shape[d] = read_i64le(p + 72 + d * 8);
    out->data_offset = read_u64le(p + 104);
    out->data_size   = read_u64le(p + 112);
}

/* ── Public API ───────────────────────────────────────────────────────────── */

/*
 * esmf_open  —  map and parse an ESMF file.
 *
 * FIX: all local variables are declared at the top of the function body
 * before any goto, eliminating C99 "jump crosses initialization" errors.
 */
esmf_file_t *esmf_open(const char *path)
{
    /* ── All locals declared up front — no initialised declarations below goto ── */
    esmf_file_t *ef;
    uint32_t     magic;
    size_t       meta_start;
    size_t       count_pos;
    size_t       dir_start;
    size_t       dir_bytes;
    size_t       dir_end;
    size_t       remainder;
    size_t       blob_start;
    uint32_t     i;

    ef = (esmf_file_t *)calloc(1, sizeof(esmf_file_t));
    if (!ef) {
        fprintf(stderr, "esmf_open: out of memory\n");
        return NULL;
    }

#if defined(ESMF_POSIX)
    ef->fd = -1;
#elif defined(ESMF_WINDOWS)
    ef->hFile = INVALID_HANDLE_VALUE;
#endif

    /* 1. Map file */
    if (esmf_map(ef, path) != 0) {
        fprintf(stderr, "esmf_open: %s\n", ef->err);
        free(ef);
        return NULL;
    }

    if (ef->file_size < 16) {
        esmf_set_err(ef, "file too small to be ESMF");
        goto fail;
    }

    /* 2. Magic */
    magic = read_u32le(ef->base + 0);
    if (magic != ESMF_MAGIC_U32) {
        esmf_set_errf(ef, "bad magic: 0x%08X (expected 0x%08X)",
                      magic, ESMF_MAGIC_U32);
        goto fail;
    }

    /* 3. Version */
    ef->version = read_u32le(ef->base + 4);
    if (ef->version != ESMF_VERSION) {
        esmf_set_errf(ef, "unsupported ESMF version %u (only v%u supported)",
                      ef->version, ESMF_VERSION);
        goto fail;
    }

    /* 4. Metadata JSON */
    ef->metadata_size = read_u64le(ef->base + 8);
    meta_start = 16;
    if (!in_bounds(ef, meta_start, (size_t)ef->metadata_size)) {
        esmf_set_err(ef, "metadata_size overflows file");
        goto fail;
    }
    ef->metadata_json = (const char *)(ef->base + meta_start);

    /* 5. Tensor count */
    count_pos = meta_start + (size_t)ef->metadata_size;
    if (!in_bounds(ef, count_pos, 4)) {
        esmf_set_err(ef, "truncated at tensor_count field");
        goto fail;
    }
    ef->tensor_count = read_u32le(ef->base + count_pos);

    /* 6. Directory */
    dir_start = count_pos + 4;
    dir_bytes = (size_t)ef->tensor_count * ESMF_ENTRY_SIZE;
    if (!in_bounds(ef, dir_start, dir_bytes)) {
        esmf_set_errf(ef, "tensor directory overflows file "
                      "(count=%u dir_start=%zu dir_bytes=%zu file=%zu)",
                      ef->tensor_count, dir_start, dir_bytes, ef->file_size);
        goto fail;
    }

    ef->tensors = (esmf_tensor_t *)calloc(ef->tensor_count, sizeof(esmf_tensor_t));
    if (!ef->tensors) {
        esmf_set_err(ef, "out of memory allocating tensor directory");
        goto fail;
    }

    for (i = 0; i < ef->tensor_count; i++) {
        parse_entry(ef->base + dir_start + (size_t)i * ESMF_ENTRY_SIZE,
                    &ef->tensors[i]);

        if (ef->tensors[i].ndim > ESMF_MAX_DIMS) {
            esmf_set_errf(ef, "tensor[%u] '%s': ndim=%u exceeds MAX_DIMS=%d",
                          i, ef->tensors[i].name,
                          ef->tensors[i].ndim, ESMF_MAX_DIMS);
            goto fail;
        }
    }

    /* 7. Blob start (align dir_end to BLOB_ALIGN) */
    dir_end   = dir_start + dir_bytes;
    remainder = dir_end % ESMF_BLOB_ALIGN;
    blob_start = (remainder == 0)
                 ? dir_end
                 : dir_end + (ESMF_BLOB_ALIGN - remainder);

    if (blob_start > ef->file_size) {
        esmf_set_err(ef, "blob_start past end of file (truncated?)");
        goto fail;
    }

    ef->blob      = ef->base + blob_start;
    ef->blob_size = ef->file_size - blob_start;

    /* 8. Cross-validate tensor data ranges */
    for (i = 0; i < ef->tensor_count; i++) {
        const esmf_tensor_t *t = &ef->tensors[i];
        uint64_t end = t->data_offset + t->data_size;
        if (end < t->data_offset /* overflow */ || end > ef->blob_size) {
            esmf_set_errf(ef,
                "tensor '%s': data_offset=%llu + data_size=%llu = %llu "
                "exceeds blob_size=%llu",
                t->name,
                (unsigned long long)t->data_offset,
                (unsigned long long)t->data_size,
                (unsigned long long)end,
                (unsigned long long)ef->blob_size);
            goto fail;
        }
    }

    return ef;

fail:
    fprintf(stderr, "esmf_open(%s): %s\n", path, ef->err);
    esmf_unmap(ef);
    free(ef->tensors);
    free(ef);
    return NULL;
}

/*
 * esmf_close  —  release all resources held by ef.
 */
void esmf_close(esmf_file_t *ef)
{
    if (!ef) return;
    esmf_unmap(ef);
    free(ef->tensors);
    free(ef);
}

/*
 * esmf_find  —  look up a tensor by name.
 */
const esmf_tensor_t *esmf_find(const esmf_file_t *ef, const char *name)
{
    uint32_t i;
    for (i = 0; i < ef->tensor_count; i++) {
        if (strcmp(ef->tensors[i].name, name) == 0)
            return &ef->tensors[i];
    }
    return NULL;
}

/*
 * esmf_data  —  get a bounds-checked pointer to a tensor's raw bytes.
 */
const void *esmf_data(const esmf_file_t *ef, const esmf_tensor_t *tensor)
{
    if (!ef || !tensor) return NULL;
    if (tensor->data_offset + tensor->data_size > ef->blob_size) return NULL;
    return (const void *)(ef->blob + tensor->data_offset);
}

/*
 * esmf_dtype_size  —  bytes per element for a given dtype enum.
 */
size_t esmf_dtype_size(uint32_t dtype)
{
    switch (dtype) {
        case ESMF_DTYPE_F32:  return 4;
        case ESMF_DTYPE_F16:  return 2;
        case ESMF_DTYPE_BF16: return 2;
        case ESMF_DTYPE_Q4_0: return 0;  /* variable-width; caller handles */
        case ESMF_DTYPE_Q8_0: return 1;
        default:              return 0;
    }
}

/*
 * esmf_dtype_name  —  human-readable dtype string.
 */
const char *esmf_dtype_name(uint32_t dtype)
{
    switch (dtype) {
        case ESMF_DTYPE_F32:  return "f32";
        case ESMF_DTYPE_F16:  return "f16";
        case ESMF_DTYPE_BF16: return "bf16";
        case ESMF_DTYPE_Q4_0: return "q4_0";
        case ESMF_DTYPE_Q8_0: return "q8_0";
        default:              return "unknown";
    }
}

/*
 * esmf_tensor_nelems  —  total element count for a tensor.
 */
uint64_t esmf_tensor_nelems(const esmf_tensor_t *t)
{
    uint32_t d;
    uint64_t n;
    if (!t || t->ndim == 0) return 0;
    n = 1;
    for (d = 0; d < t->ndim; d++) {
        if (t->shape[d] <= 0) return 0;
        n *= (uint64_t)t->shape[d];
    }
    return n;
}

/*
 * esmf_metadata_copy  —  copy the metadata JSON into a caller buffer.
 */
size_t esmf_metadata_copy(const esmf_file_t *ef, char *buf, size_t buf_size)
{
    size_t n;
    if (!ef || !buf || buf_size == 0) return (size_t)-1;
    n = (size_t)ef->metadata_size;
    if (n >= buf_size) n = buf_size - 1;
    memcpy(buf, ef->metadata_json, n);
    buf[n] = '\0';
    return n;
}

/*
 * esmf_print_summary  —  dump a human-readable summary to fp.
 */
void esmf_print_summary(const esmf_file_t *ef, FILE *fp)
{
    uint32_t i;
    if (!ef || !fp) return;
    fprintf(fp, "ESMF v%u  tensors=%u  blob=%llu bytes\n",
            ef->version, ef->tensor_count,
            (unsigned long long)ef->blob_size);
    fprintf(fp, "Metadata (%llu bytes):\n",
            (unsigned long long)ef->metadata_size);
    fwrite(ef->metadata_json, 1, (size_t)ef->metadata_size, fp);
    fputc('\n', fp);

    fprintf(fp, "\nTensor directory:\n");
    fprintf(fp, "  %-42s  %-6s  %-5s  %s\n",
            "name", "dtype", "ndim", "shape");
    fprintf(fp, "  %s\n",
            "------------------------------------------------------------"
            "-------------------");
    for (i = 0; i < ef->tensor_count; i++) {
        uint32_t d;
        const esmf_tensor_t *t = &ef->tensors[i];
        fprintf(fp, "  %-42s  %-6s  %u     [",
                t->name, esmf_dtype_name(t->dtype), t->ndim);
        for (d = 0; d < t->ndim; d++) {
            if (d) fprintf(fp, ", ");
            fprintf(fp, "%lld", (long long)t->shape[d]);
        }
        fprintf(fp, "]  (%llu bytes @ offset %llu)\n",
                (unsigned long long)t->data_size,
                (unsigned long long)t->data_offset);
    }
}

/* ── Optional standalone test driver ─────────────────────────────────────── */

#ifdef ESMF_TEST

int main(int argc, char **argv)
{
    esmf_file_t *ef;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file.esmf> [tensor_name]\n", argv[0]);
        return 1;
    }

    ef = esmf_open(argv[1]);
    if (!ef) return 1;

    esmf_print_summary(ef, stdout);

    if (argc >= 3) {
        const char *name = argv[2];
        const esmf_tensor_t *t = esmf_find(ef, name);
        if (!t) {
            fprintf(stderr, "\nTensor '%s' not found.\n", name);
        } else {
            const void *data = esmf_data(ef, t);
            fprintf(stdout, "\nTensor '%s':\n", name);
            fprintf(stdout, "  dtype:    %s\n", esmf_dtype_name(t->dtype));
            fprintf(stdout, "  ndim:     %u\n", t->ndim);
            fprintf(stdout, "  nelems:   %llu\n",
                    (unsigned long long)esmf_tensor_nelems(t));
            fprintf(stdout, "  data_ptr: %p\n", data);
            if (t->dtype == ESMF_DTYPE_F32 && data) {
                const float *fp32 = (const float *)data;
                uint64_t n    = esmf_tensor_nelems(t);
                uint64_t show = n < 8 ? n : 8;
                uint64_t k;
                fprintf(stdout, "  first %llu values:", (unsigned long long)show);
                for (k = 0; k < show; k++)
                    fprintf(stdout, " %.6g", fp32[k]);
                fprintf(stdout, "\n");
            }
        }
    }

    esmf_close(ef);
    return 0;
}

#endif /* ESMF_TEST */