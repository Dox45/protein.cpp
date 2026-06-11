#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESMF_DTYPE_F32   0u
#define ESMF_DTYPE_F16   1u
#define ESMF_DTYPE_BF16  2u
#define ESMF_DTYPE_Q4_0  3u
#define ESMF_DTYPE_Q8_0  4u

#define ESMF_MAX_DIMS    4
#define ESMF_NAME_MAXLEN 64

typedef struct {
    char     name[ESMF_NAME_MAXLEN + 1];
    uint32_t dtype;
    uint32_t ndim;
    int64_t  shape[ESMF_MAX_DIMS];
    uint64_t data_offset;
    uint64_t data_size;
} esmf_tensor_t;

typedef struct esmf_file_t esmf_file_t;

esmf_file_t        *esmf_open(const char *path);
void                esmf_close(esmf_file_t *ef);
const esmf_tensor_t*esmf_find(const esmf_file_t *ef, const char *name);
const void         *esmf_data(const esmf_file_t *ef, const esmf_tensor_t *t);
size_t              esmf_dtype_size(uint32_t dtype);
const char         *esmf_dtype_name(uint32_t dtype);
uint64_t            esmf_tensor_nelems(const esmf_tensor_t *t);
size_t              esmf_metadata_copy(const esmf_file_t *ef,
                                        char *buf, size_t buf_size);
void                esmf_print_summary(const esmf_file_t *ef, FILE *fp);
const char         *esmf_last_error(const esmf_file_t *ef);

uint32_t            esmf_tensor_count(const esmf_file_t *ef);
const esmf_tensor_t*esmf_get_tensor(const esmf_file_t *ef, uint32_t index);
uint64_t            esmf_metadata_size(const esmf_file_t *ef);
const char         *esmf_metadata_json(const esmf_file_t *ef);

#ifdef __cplusplus
}
#endif