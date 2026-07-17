#ifndef PP_KERNELS_H
#define PP_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Internal operator boundary used by deterministic differential fixtures. */
void pp_cpu_conv3_relu(const float *x, float *y, const float *weights,
                       const float *bias, int input_channels,
                       int output_channels, int input_height,
                       int input_width, int stride);
void pp_cpu_conv3_plain(const float *x, float *y, const float *weights,
                        const float *bias, int input_channels,
                        int output_channels, int height, int width);

#ifdef __cplusplus
}
#endif
#endif
