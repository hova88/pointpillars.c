#define _POSIX_C_SOURCE 200809L
#include "pp_model.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char magic[8];
    uint32_t version, count, record_size, alignment;
    uint64_t data_offset, data_bytes;
    uint32_t table_crc32;
    uint8_t reserved[20];
} pp_file_header;

_Static_assert(sizeof(pp_file_header) == 64, "container header ABI");
_Static_assert(sizeof(pp_weight_record) == 112, "container record ABI");

static uint32_t crc32_bytes(const void *ptr, size_t n) {
    const uint8_t *p = (const uint8_t *)ptr;
    uint32_t c = ~0u;
    while (n--) {
        c ^= *p++;
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    }
    return ~c;
}

static int fail(char *out, size_t cap, const char *fmt, ...) {
    if (out && cap) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(out, cap, fmt, ap);
        va_end(ap);
    }
    return 0;
}

int pp_model_open(pp_model *m, const char *path, char *err, size_t cap) {
    memset(m, 0, sizeof(*m));
    int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(err, cap, "%s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) || st.st_size < (off_t)sizeof(pp_file_header)) {
        close(fd); return fail(err, cap, "model file is truncated");
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) return fail(err, cap, "mmap: %s", strerror(errno));
    const pp_file_header *h = (const pp_file_header *)map;
    const char expected[8] = {'P','P','W','G','T',0,0,0};
    uint64_t table_end = sizeof(*h) + (uint64_t)h->count * h->record_size;
    int valid = !memcmp(h->magic, expected, 8) &&
        h->version == 2 && h->count <= PP_MAX_TENSORS && h->record_size == sizeof(pp_weight_record) &&
        h->alignment == 64 && h->data_offset >= table_end &&
        h->data_offset <= (uint64_t)st.st_size && h->data_bytes <= (uint64_t)st.st_size - h->data_offset;
    if (!valid) { munmap(map, (size_t)st.st_size); return fail(err, cap, "invalid model header"); }
    const pp_weight_record *r = (const pp_weight_record *)((const uint8_t *)map + sizeof(*h));
    if (crc32_bytes(r, (size_t)h->count * sizeof(*r)) != h->table_crc32) {
        munmap(map, (size_t)st.st_size); return fail(err, cap, "model tensor table checksum mismatch");
    }
    for (uint32_t i = 0; i < h->count; ++i) {
        if (!memchr(r[i].name, 0, PP_NAME_BYTES) || r[i].rank > 4 ||
            r[i].offset > h->data_bytes || r[i].bytes > h->data_bytes - r[i].offset) {
            munmap(map, (size_t)st.st_size); return fail(err, cap, "invalid tensor record %u", i);
        }
        const uint8_t *p = (const uint8_t *)map + h->data_offset + r[i].offset;
        if (crc32_bytes(p, (size_t)r[i].bytes) != r[i].crc32) {
            munmap(map, (size_t)st.st_size); return fail(err, cap, "tensor checksum mismatch: %s", r[i].name);
        }
    }
    m->mapping = map; m->mapping_bytes = (size_t)st.st_size;
    m->data = (const uint8_t *)map + h->data_offset; m->data_bytes = (size_t)h->data_bytes;
    m->count = h->count; m->records = r;
    return 1;
}

void pp_model_close(pp_model *m) {
    if (m && m->mapping) munmap(m->mapping, m->mapping_bytes);
    if (m) memset(m, 0, sizeof(*m));
}

const float *pp_model_tensor(const pp_model *m, const char *name,
                             const uint32_t *dims, uint32_t rank) {
    for (uint32_t i = 0; i < m->count; ++i) if (!strcmp(m->records[i].name, name)) {
        const pp_weight_record *r = &m->records[i];
        if (r->rank != rank) return NULL;
        uint64_t count = 1;
        for (uint32_t d = 0; d < rank; ++d) {
            if (dims && r->dim[d] != dims[d]) return NULL;
            count *= r->dim[d];
        }
        if (count * sizeof(float) != r->bytes) return NULL;
        return (const float *)(m->data + r->offset);
    }
    return NULL;
}
