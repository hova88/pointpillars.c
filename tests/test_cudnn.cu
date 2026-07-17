#include "pp_cudnn.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static float value(int i)
{
    return std::sin((float)i * 0.37f) * 0.25f;
}

static int check_cuda(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess) {
        return 1;
    }
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return 0;
}

static float compare(const float *actual, const float *expected, size_t count)
{
    float maximum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        maximum = std::fmax(maximum, std::fabs(actual[i] - expected[i]));
    }
    return maximum;
}

static void scalar_conv(const float *input, const float *weights,
                        const float *bias, float *output, int ci, int co,
                        int height, int width, int relu)
{
    for (int oc = 0; oc < co; oc++) {
        for (int oy = 0; oy < height; oy++) {
            for (int ox = 0; ox < width; ox++) {
                float sum = bias[oc];
                for (int ic = 0; ic < ci; ic++) {
                    for (int ky = 0; ky < 3; ky++) {
                        for (int kx = 0; kx < 3; kx++) {
                            int iy = oy + ky - 1;
                            int ix = ox + kx - 1;
                            if ((unsigned)iy < (unsigned)height &&
                                (unsigned)ix < (unsigned)width) {
                                size_t xi = ((size_t)ic * height + iy) * width + ix;
                                size_t wi = (((size_t)oc * ci + ic) * 3 + ky) * 3 + kx;
                                sum += input[xi] * weights[wi];
                            }
                        }
                    }
                }
                output[((size_t)oc * height + oy) * width + ox] =
                    relu ? std::fmax(sum, 0.0f) : sum;
            }
        }
    }
}

static int test_conv(int relu)
{
    enum { CI = 3, CO = 5, H = 7, W = 9 };
    const size_t input_count = (size_t)CI * H * W;
    const size_t weight_count = (size_t)CO * CI * 3 * 3;
    const size_t output_count = (size_t)CO * H * W;
    float *input = (float *)std::malloc(input_count * sizeof(float));
    float *weights = (float *)std::malloc(weight_count * sizeof(float));
    float *bias = (float *)std::malloc(CO * sizeof(float));
    float *expected = (float *)std::malloc(output_count * sizeof(float));
    float *actual = (float *)std::malloc(output_count * sizeof(float));
    float *device_input = nullptr;
    float *device_weights = nullptr;
    float *device_bias = nullptr;
    float *device_output = nullptr;
    if (!input || !weights || !bias || !expected || !actual ||
        !check_cuda(cudaMalloc(&device_input, input_count * sizeof(float)),
                    "cudaMalloc conv input") ||
        !check_cuda(cudaMalloc(&device_weights, weight_count * sizeof(float)),
                    "cudaMalloc conv weights") ||
        !check_cuda(cudaMalloc(&device_bias, CO * sizeof(float)),
                    "cudaMalloc conv bias") ||
        !check_cuda(cudaMalloc(&device_output, output_count * sizeof(float)),
                    "cudaMalloc conv output")) {
        return 0;
    }
    for (size_t i = 0; i < input_count; i++) input[i] = value((int)i + 3);
    for (size_t i = 0; i < weight_count; i++) weights[i] = value((int)i + 17);
    for (int i = 0; i < CO; i++) bias[i] = value(i + 101);
    scalar_conv(input, weights, bias, expected, CI, CO, H, W, relu);
    check_cuda(cudaMemcpy(device_input, input, input_count * sizeof(float),
                          cudaMemcpyHostToDevice), "copy conv input");
    check_cuda(cudaMemcpy(device_weights, weights, weight_count * sizeof(float),
                          cudaMemcpyHostToDevice), "copy conv weights");
    check_cuda(cudaMemcpy(device_bias, bias, CO * sizeof(float),
                          cudaMemcpyHostToDevice), "copy conv bias");
    char error[256] = {0};
    int ok = pp_cudnn_conv_raw("fixture.conv", device_weights, device_bias,
                               device_input, device_output, CI, CO, H, W, 1,
                               3, 1, relu, 1, cudaStreamLegacy, error,
                               sizeof(error));
    if (!ok) std::fprintf(stderr, "%s\n", error);
    ok = ok && check_cuda(cudaMemcpy(actual, device_output,
                                     output_count * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy conv output");
    float error_max = ok ? compare(actual, expected, output_count) : INFINITY;
    std::printf("cudnn conv relu=%d max_error=%.3g\n", relu, error_max);
    cudaFree(device_output);
    cudaFree(device_bias);
    cudaFree(device_weights);
    cudaFree(device_input);
    std::free(actual);
    std::free(expected);
    std::free(bias);
    std::free(weights);
    std::free(input);
    return ok && error_max <= 2e-6f;
}

static void scalar_deconv(const float *input, const float *weights,
                          const float *bias, float *output, int ci, int co,
                          int height, int width, int kernel)
{
    int output_height = height * kernel;
    int output_width = width * kernel;
    for (int oc = 0; oc < co; oc++) {
        for (int oy = 0; oy < output_height; oy++) {
            for (int ox = 0; ox < output_width; ox++) {
                int iy = oy / kernel;
                int ix = ox / kernel;
                int ky = oy % kernel;
                int kx = ox % kernel;
                float sum = bias[oc];
                for (int ic = 0; ic < ci; ic++) {
                    size_t xi = ((size_t)ic * height + iy) * width + ix;
                    size_t wi = (((size_t)ic * co + oc) * kernel + ky) * kernel + kx;
                    sum += input[xi] * weights[wi];
                }
                output[((size_t)oc * output_height + oy) * output_width + ox] =
                    std::fmax(sum, 0.0f);
            }
        }
    }
}

static int test_deconv(void)
{
    enum { CI = 4, CO = 3, H = 5, W = 7, K = 2 };
    const size_t input_count = (size_t)CI * H * W;
    const size_t weight_count = (size_t)CI * CO * K * K;
    const size_t output_count = (size_t)CO * H * K * W * K;
    const size_t weight_bytes = weight_count * sizeof(float);
    const size_t bias_bytes = CO * sizeof(float);
    float *input = (float *)std::malloc(input_count * sizeof(float));
    float *host_weights = (float *)std::malloc(weight_count * sizeof(float));
    float *bias = (float *)std::malloc(bias_bytes);
    float *expected = (float *)std::malloc(output_count * sizeof(float));
    float *actual = (float *)std::malloc(output_count * sizeof(float));
    float *device_input = nullptr;
    float *device_output = nullptr;
    uint8_t *device_model = nullptr;
    if (!input || !host_weights || !bias || !expected || !actual ||
        !check_cuda(cudaMalloc(&device_input, input_count * sizeof(float)),
                    "cudaMalloc deconv input") ||
        !check_cuda(cudaMalloc(&device_output, output_count * sizeof(float)),
                    "cudaMalloc deconv output") ||
        !check_cuda(cudaMalloc(&device_model, weight_bytes + bias_bytes),
                    "cudaMalloc deconv model")) {
        return 0;
    }
    for (size_t i = 0; i < input_count; i++) input[i] = value((int)i + 7);
    for (size_t i = 0; i < weight_count; i++) host_weights[i] = value((int)i + 29);
    for (int i = 0; i < CO; i++) bias[i] = value(i + 131);
    scalar_deconv(input, host_weights, bias, expected, CI, CO, H, W, K);
    check_cuda(cudaMemcpy(device_input, input, input_count * sizeof(float),
                          cudaMemcpyHostToDevice), "copy deconv input");
    check_cuda(cudaMemcpy(device_model, host_weights, weight_bytes,
                          cudaMemcpyHostToDevice), "copy deconv weights");
    check_cuda(cudaMemcpy(device_model + weight_bytes, bias, bias_bytes,
                          cudaMemcpyHostToDevice), "copy deconv bias");
    pp_weight_record records[2] = {};
    std::snprintf(records[0].name, sizeof(records[0].name),
                  "fixture.deconv.weight");
    records[0].offset = 0;
    std::snprintf(records[1].name, sizeof(records[1].name),
                  "fixture.deconv.bias");
    records[1].offset = weight_bytes;
    pp_model model = {};
    model.count = 2;
    model.records = records;
    model.data_bytes = weight_bytes + bias_bytes;
    char error[256] = {0};
    int ok = pp_cudnn_deconv(&model, device_model, "fixture.deconv",
                             device_input, device_output, CI, CO, H, W, K, K,
                             1, cudaStreamLegacy, error, sizeof(error));
    if (!ok) std::fprintf(stderr, "%s\n", error);
    ok = ok && check_cuda(cudaMemcpy(actual, device_output,
                                     output_count * sizeof(float),
                                     cudaMemcpyDeviceToHost),
                          "copy deconv output");
    float error_max = ok ? compare(actual, expected, output_count) : INFINITY;
    std::printf("cudnn deconv max_error=%.3g\n", error_max);
    cudaFree(device_model);
    cudaFree(device_output);
    cudaFree(device_input);
    std::free(actual);
    std::free(expected);
    std::free(bias);
    std::free(host_weights);
    std::free(input);
    return ok && error_max <= 2e-6f;
}

int main(void)
{
    if (!test_conv(1)) return 1;
    if (!test_conv(0)) return 2;
    if (!test_deconv()) return 3;
    std::puts("cudnn fixtures: ok");
    return 0;
}
