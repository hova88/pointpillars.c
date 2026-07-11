#ifndef PP_CUDA_H
#define PP_CUDA_H
#include "pp_infer.h"
#ifdef __cplusplus
extern "C" {
#endif
int pp_cuda_available(char *error, size_t error_size);
int pp_infer_cuda(const pp_model *model, const pp_pillars *pillars,
                  pp_raw_output *out, pp_profile *profile,
                  char *error, size_t error_size);
#ifdef __cplusplus
}
#endif
#endif
