#include "pp_model.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    pp_model model;
    char error[256];
    if (!pp_model_open(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "%s\n", error); return 1;
    }
    const uint32_t pfn_shape[] = {64, 11};
    if (!pp_model_tensor(&model, "pfn.weight", pfn_shape, 2)) return 3;
    const uint32_t head_shape[] = {8, 64, 3, 3};
    if (!pp_model_tensor(&model, "head.1.cls.out.weight", head_shape, 4)) return 4;
    if (model.count != 190) return 5;
    printf("model container: %u tensors, %.2f MiB mapped\n", model.count,
           model.mapping_bytes / 1048576.0);
    pp_model_close(&model);
    return 0;
}
