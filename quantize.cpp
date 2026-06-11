#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include "ggml.h"
#include "esmf.h"

#define TENSOR_NAME_MAXLEN 64
#define BLOB_ALIGNMENT 64

struct packed_entry {
    char name[TENSOR_NAME_MAXLEN];
    uint32_t dtype;
    uint32_t ndim;
    int64_t shape[4];
    uint64_t data_offset;
    uint64_t data_size;
} __attribute__((packed));

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_file.esmf> <output_file.esmf> <type>\n", argv[0]);
        fprintf(stderr, "  type: q4_0, q8_0\n");
        return 1;
    }

    const char *src_path = argv[1];
    const char *dst_path = argv[2];
    const char *type_str = argv[3];

    ggml_type qtype = GGML_TYPE_F32;
    uint32_t esmf_dtype = ESMF_DTYPE_F32;

    if (strcmp(type_str, "q4_0") == 0) {
        qtype = GGML_TYPE_Q4_0;
        esmf_dtype = ESMF_DTYPE_Q4_0;
    } else if (strcmp(type_str, "q8_0") == 0) {
        qtype = GGML_TYPE_Q8_0;
        esmf_dtype = ESMF_DTYPE_Q8_0;
    } else {
        fprintf(stderr, "error: unsupported quantization type '%s'\n", type_str);
        return 1;
    }

    printf("Quantizing '%s' -> '%s' as %s...\n", src_path, dst_path, type_str);

    // Initialize GGML quantization tables
    ggml_quantize_init(qtype);

    esmf_file_t *ef = esmf_open(src_path);
    if (!ef) {
        fprintf(stderr, "error: failed to open input file '%s'\n", src_path);
        return 1;
    }

    uint32_t tensor_count = esmf_tensor_count(ef);
    printf("Loaded model with %u tensors\n", tensor_count);

    struct processed_tensor {
        std::string name;
        uint32_t dtype;
        uint32_t ndim;
        int64_t shape[4];
        std::vector<uint8_t> data;
    };
    std::vector<processed_tensor> processed_tensors;
    processed_tensors.reserve(tensor_count);

    size_t total_orig_size = 0;
    size_t total_new_size = 0;

    for (uint32_t i = 0; i < tensor_count; i++) {
        const esmf_tensor_t *t = esmf_get_tensor(ef, i);
        const void *src_ptr = esmf_data(ef, t);
        if (!src_ptr) {
            fprintf(stderr, "error: failed to get data pointer for tensor '%s'\n", t->name);
            esmf_close(ef);
            return 1;
        }

        total_orig_size += t->data_size;

        processed_tensor pt;
        pt.name = t->name;
        pt.ndim = t->ndim;
        std::memcpy(pt.shape, t->shape, sizeof(pt.shape));

        // Quantize all 2D matrices except token_embd.weight
        bool should_quantize = (t->ndim == 2 && pt.name != "token_embd.weight");

        if (should_quantize) {
            uint64_t nelems = esmf_tensor_nelems(t);
            std::vector<float> fp32_buf(nelems);

            // Convert to FP32 first
            if (t->dtype == ESMF_DTYPE_F32) {
                std::memcpy(fp32_buf.data(), src_ptr, nelems * sizeof(float));
            } else if (t->dtype == ESMF_DTYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *)src_ptr, fp32_buf.data(), nelems);
            } else if (t->dtype == ESMF_DTYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *)src_ptr, fp32_buf.data(), nelems);
            } else {
                fprintf(stderr, "error: cannot quantize tensor '%s' of dtype %u\n", t->name, t->dtype);
                esmf_close(ef);
                return 1;
            }

            int64_t n_per_row = t->shape[0];
            int64_t nrows = t->shape[1];

            size_t max_size = ggml_row_size(qtype, n_per_row) * nrows;
            pt.data.resize(max_size);

            size_t actual_size = ggml_quantize_chunk(
                qtype,
                fp32_buf.data(),
                pt.data.data(),
                0,
                nrows,
                n_per_row,
                nullptr
            );

            if (actual_size == 0) {
                fprintf(stderr, "error: ggml_quantize_chunk failed for tensor '%s'\n", t->name);
                esmf_close(ef);
                return 1;
            }

            pt.data.resize(actual_size);
            pt.dtype = esmf_dtype;

            printf("  %-40s : quantized %s -> %s (size %7.2f MB -> %7.2f MB)\n",
                   t->name, esmf_dtype_name(t->dtype), type_str,
                   (double)t->data_size / (1024 * 1024),
                   (double)actual_size / (1024 * 1024));
        } else {
            // Keep original configuration
            pt.dtype = t->dtype;
            pt.data.assign((const uint8_t *)src_ptr, (const uint8_t *)src_ptr + t->data_size);
            printf("  %-40s : kept %s (size %7.2f MB)\n",
                   t->name, esmf_dtype_name(t->dtype),
                   (double)t->data_size / (1024 * 1024));
        }

        total_new_size += pt.data.size();
        processed_tensors.push_back(std::move(pt));
    }

    // Write the output file
    FILE *fout = std::fopen(dst_path, "wb");
    if (!fout) {
        fprintf(stderr, "error: failed to open output file '%s' for writing\n", dst_path);
        esmf_close(ef);
        return 1;
    }

    // 1. Magic
    std::fwrite("ESMF", 1, 4, fout);

    // 2. Version
    uint32_t version = 1;
    std::fwrite(&version, sizeof(version), 1, fout);

    // 3. Metadata Size & Metadata JSON
    uint64_t metadata_size = esmf_metadata_size(ef);
    std::fwrite(&metadata_size, sizeof(metadata_size), 1, fout);
    std::fwrite(esmf_metadata_json(ef), 1, metadata_size, fout);

    // 4. Tensor Count
    std::fwrite(&tensor_count, sizeof(tensor_count), 1, fout);

    // 5. Compute directory offsets
    std::vector<packed_entry> dir(tensor_count);
    uint64_t offset = 0;
    for (uint32_t i = 0; i < tensor_count; i++) {
        const auto &pt = processed_tensors[i];
        std::memset(&dir[i], 0, sizeof(packed_entry));
        std::strncpy(dir[i].name, pt.name.c_str(), TENSOR_NAME_MAXLEN - 1);
        dir[i].dtype = pt.dtype;
        dir[i].ndim = pt.ndim;
        std::memcpy(dir[i].shape, pt.shape, sizeof(dir[i].shape));
        dir[i].data_offset = offset;
        dir[i].data_size = pt.data.size();

        offset += pt.data.size();
        uint64_t remainder = offset % BLOB_ALIGNMENT;
        if (remainder != 0) {
            offset += BLOB_ALIGNMENT - remainder;
        }
    }

    // 6. Write Directory
    std::fwrite(dir.data(), sizeof(packed_entry), tensor_count, fout);

    // 7. Alignment padding before blob
    uint64_t current_pos = 4 + 4 + 8 + metadata_size + 4 + tensor_count * sizeof(packed_entry);
    uint64_t remainder = current_pos % BLOB_ALIGNMENT;
    uint64_t pad_size = (remainder == 0) ? 0 : (BLOB_ALIGNMENT - remainder);
    std::vector<uint8_t> padding(pad_size, 0);
    if (pad_size > 0) {
        std::fwrite(padding.data(), 1, pad_size, fout);
    }

    // 8. Write Tensor Blob data
    for (uint32_t i = 0; i < tensor_count; i++) {
        const auto &pt = processed_tensors[i];
        std::fwrite(pt.data.data(), 1, pt.data.size(), fout);

        // Pad to next 64B alignment
        uint64_t size_written = pt.data.size();
        uint64_t size_remainder = size_written % BLOB_ALIGNMENT;
        if (size_remainder != 0) {
            uint64_t size_pad = BLOB_ALIGNMENT - size_remainder;
            std::vector<uint8_t> size_padding(size_pad, 0);
            std::fwrite(size_padding.data(), 1, size_pad, fout);
        }
    }

    std::fclose(fout);
    esmf_close(ef);

    printf("Quantization completed successfully!\n");
    printf("Original size: %7.2f MB\n", (double)total_orig_size / (1024 * 1024));
    printf("Quantized size: %7.2f MB (%.1f%% of original)\n",
           (double)total_new_size / (1024 * 1024),
           ((double)total_new_size / total_orig_size) * 100.0);

    ggml_quantize_free();
    return 0;
}
