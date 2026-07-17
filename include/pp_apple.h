#ifndef PP_APPLE_H
#define PP_APPLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Apple Accelerate/BNNS convolution adapter. Returns zero when the platform
 * backend is disabled or cannot represent the requested shape. */
int pp_apple_conv(const float *input, float *output, const float *weights,
                  const float *bias, int input_channels,
                  int output_channels, int input_height, int input_width,
                  int kernel, int stride, int padding, int relu);
int pp_apple_deconv(const float *input, float *output, const float *weights,
                    const float *bias, int input_channels,
                    int output_channels, int input_height, int input_width,
                    int kernel, int stride, int relu);
size_t pp_apple_cache_bytes(void);

#ifdef __cplusplus
}
#endif
#endif
