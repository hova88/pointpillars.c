#ifndef PP_GGML_H
#define PP_GGML_H

#include <stddef.h>

int pp_ggml_conv3_relu(const float *input, float *output,
                       const float *weights, const float *bias,
                       int input_channels, int output_channels,
                       int input_height, int input_width, int stride);
size_t pp_ggml_workspace_bytes(void);

#endif
