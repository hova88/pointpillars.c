#ifndef PP_CUDNN_H
#define PP_CUDNN_H

#include "pp_model.h"

#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int pp_cudnn_conv(const pp_model *model, const uint8_t *device_weights,
                  const char *name, const float *input, float *output,
                  int input_channels, int output_channels, int input_height,
                  int input_width, int stride, int kernel, int padding,
                  int relu, int precise, cudaStream_t stream,
                  char *error, size_t error_capacity);
int pp_cudnn_conv_raw(const char *plan_name, const float *weights,
                      const float *bias, const float *input, float *output,
                      int input_channels, int output_channels,
                      int input_height, int input_width, int stride,
                      int kernel, int padding, int relu, int precise,
                      cudaStream_t stream, char *error,
                      size_t error_capacity);
int pp_cudnn_deconv(const pp_model *model, const uint8_t *device_weights,
                    const char *name, const float *input, float *output,
                    int input_channels, int output_channels, int input_height,
                    int input_width, int stride, int kernel, int precise,
                    cudaStream_t stream, char *error, size_t error_capacity);
size_t pp_cudnn_workspace_bytes(void);

#ifdef __cplusplus
}
#endif

#endif
