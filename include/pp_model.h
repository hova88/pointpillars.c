#ifndef PP_MODEL_H
#define PP_MODEL_H

#include <stddef.h>
#include <stdint.h>

#define PP_MAX_TENSORS 256u
#define PP_NAME_BYTES 48u

typedef struct {
    char name[PP_NAME_BYTES];
    uint32_t rank;
    uint32_t dim[4];
    uint64_t offset;
    uint64_t bytes;
    uint32_t crc32;
    uint8_t reserved[20];
} pp_weight_record;

typedef struct {
    void *mapping;
    size_t mapping_bytes;
    const uint8_t *data;
    size_t data_bytes;
    uint32_t count;
    const pp_weight_record *records;
} pp_model;

int pp_model_open(pp_model *model, const char *path, char *error, size_t error_size);
void pp_model_close(pp_model *model);
const float *pp_model_tensor(const pp_model *model, const char *name,
                             const uint32_t *dims, uint32_t rank);

#endif
