#include "pp_cudnn.h"

#include <cudnn.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

enum {
    PP_CUDNN_MAX_PLANS = 128,
    PP_CUDNN_MAX_DECONV_PLANS = 8
};

struct pp_cudnn_plan {
    char name[48];
    int precise;
    int relu;
    int input_channels;
    int output_channels;
    int input_height;
    int input_width;
    int stride;
    int kernel;
    int padding;
    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t convolution_desc;
    cudnnActivationDescriptor_t activation_desc;
    cudnnConvolutionFwdAlgo_t algorithm;
    size_t workspace_bytes;
};

struct pp_cudnn_cache {
    const pp_model *model;
    cudnnHandle_t handle;
    pp_cudnn_plan plans[PP_CUDNN_MAX_PLANS];
    int plan_count;
    struct pp_cudnn_deconv_plan {
        char name[48];
        int precise;
        int input_channels;
        int output_channels;
        int input_height;
        int input_width;
        int stride;
        int kernel;
        cudnnTensorDescriptor_t input_desc;
        cudnnTensorDescriptor_t output_desc;
        cudnnTensorDescriptor_t bias_desc;
        cudnnFilterDescriptor_t filter_desc;
        cudnnConvolutionDescriptor_t convolution_desc;
        cudnnActivationDescriptor_t activation_desc;
        cudnnConvolutionBwdDataAlgo_t algorithm;
        size_t workspace_bytes;
    } deconv_plans[PP_CUDNN_MAX_DECONV_PLANS];
    int deconv_plan_count;
    void *workspace;
    size_t workspace_bytes;
};

static pp_cudnn_cache cache;

static int fail(char *error, size_t capacity, const char *operation,
                cudnnStatus_t status)
{
    if (error && capacity) {
        std::snprintf(error, capacity, "%s: %s", operation,
                      cudnnGetErrorString(status));
    }
    return 0;
}

static int fail_cuda(char *error, size_t capacity, const char *operation,
                     cudaError_t status)
{
    if (error && capacity) {
        std::snprintf(error, capacity, "%s: %s", operation,
                      cudaGetErrorString(status));
    }
    return 0;
}

static const pp_weight_record *find_record(const pp_model *model,
                                           const char *name)
{
    for (uint32_t i = 0; i < model->count; i++) {
        if (!std::strcmp(model->records[i].name, name)) {
            return model->records + i;
        }
    }
    return nullptr;
}

static int same_shape(const pp_cudnn_plan *plan, const char *name, int precise,
                      int relu, int input_channels, int output_channels,
                      int input_height, int input_width, int stride,
                      int kernel, int padding)
{
    (void)name;
    return plan->precise == precise && plan->relu == relu &&
           plan->input_channels == input_channels &&
           plan->output_channels == output_channels &&
           plan->input_height == input_height &&
           plan->input_width == input_width && plan->stride == stride &&
           plan->kernel == kernel && plan->padding == padding;
}

static size_t workspace_limit(void)
{
    static size_t limit;
    static int initialized;
    if (!initialized) {
        const char *value = std::getenv("PP_CUDNN_WORKSPACE_MIB");
        unsigned long long mib = value ? std::strtoull(value, nullptr, 10) : 64;
        if (mib > 4096) {
            mib = 4096;
        }
        limit = (size_t)mib << 20;
        initialized = 1;
    }
    return limit;
}

static int initialize(char *error, size_t capacity)
{
    if (cache.handle) {
        return 1;
    }
    cudnnStatus_t status = cudnnCreate(&cache.handle);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity, "cudnnCreate", status);
    }
    return 1;
}

static int grow_workspace(size_t bytes, char *error, size_t capacity)
{
    if (bytes <= cache.workspace_bytes) {
        return 1;
    }
    if (cache.workspace) {
        cudaError_t status = cudaFree(cache.workspace);
        if (status != cudaSuccess) {
            return fail_cuda(error, capacity, "cudaFree cuDNN workspace", status);
        }
        cache.workspace = nullptr;
        cache.workspace_bytes = 0;
    }
    cudaError_t status = cudaMalloc(&cache.workspace, bytes);
    if (status != cudaSuccess) {
        return fail_cuda(error, capacity, "cudaMalloc cuDNN workspace", status);
    }
    cache.workspace_bytes = bytes;
    return 1;
}

static int create_plan(pp_cudnn_plan **result, const char *name, int precise,
                       int relu, int input_channels, int output_channels,
                       int input_height, int input_width, int stride,
                       int kernel, int padding, char *error, size_t capacity)
{
    if (cache.plan_count == PP_CUDNN_MAX_PLANS) {
        if (error && capacity) {
            std::snprintf(error, capacity, "cuDNN plan cache is full");
        }
        return 0;
    }

    pp_cudnn_plan *plan = cache.plans + cache.plan_count;
    std::snprintf(plan->name, sizeof(plan->name), "%s", name);
    plan->precise = precise;
    plan->relu = relu;
    plan->input_channels = input_channels;
    plan->output_channels = output_channels;
    plan->input_height = input_height;
    plan->input_width = input_width;
    plan->stride = stride;
    plan->kernel = kernel;
    plan->padding = padding;

#define CUDNN_CREATE(call, label) do { \
        cudnnStatus_t create_status = (call); \
        if (create_status != CUDNN_STATUS_SUCCESS) { \
            return fail(error, capacity, (label), create_status); \
        } \
    } while (0)
    CUDNN_CREATE(cudnnCreateTensorDescriptor(&plan->input_desc),
                 "cudnnCreateTensorDescriptor input");
    CUDNN_CREATE(cudnnCreateTensorDescriptor(&plan->output_desc),
                 "cudnnCreateTensorDescriptor output");
    CUDNN_CREATE(cudnnCreateTensorDescriptor(&plan->bias_desc),
                 "cudnnCreateTensorDescriptor bias");
    CUDNN_CREATE(cudnnCreateFilterDescriptor(&plan->filter_desc),
                 "cudnnCreateFilterDescriptor");
    CUDNN_CREATE(cudnnCreateConvolutionDescriptor(&plan->convolution_desc),
                 "cudnnCreateConvolutionDescriptor");
    CUDNN_CREATE(cudnnCreateActivationDescriptor(&plan->activation_desc),
                 "cudnnCreateActivationDescriptor");
#undef CUDNN_CREATE

    int output_height =
        (input_height + 2 * padding - kernel) / stride + 1;
    int output_width =
        (input_width + 2 * padding - kernel) / stride + 1;
    cudnnStatus_t status = cudnnSetTensor4dDescriptor(
        plan->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
        input_channels, input_height, input_width);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity, "cudnnSetTensor4dDescriptor input", status);
    }
    status = cudnnSetTensor4dDescriptor(
        plan->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
        output_channels, output_height, output_width);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity, "cudnnSetTensor4dDescriptor output", status);
    }
    status = cudnnSetTensor4dDescriptor(
        plan->bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
        output_channels, 1, 1);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity, "cudnnSetTensor4dDescriptor bias", status);
    }
    status = cudnnSetFilter4dDescriptor(
        plan->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        output_channels, input_channels, kernel, kernel);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity, "cudnnSetFilter4dDescriptor", status);
    }
    status = cudnnSetConvolution2dDescriptor(
        plan->convolution_desc, padding, padding, stride, stride, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity, "cudnnSetConvolution2dDescriptor", status);
    }
    status = cudnnSetConvolutionMathType(
        plan->convolution_desc,
        precise ? CUDNN_FMA_MATH : CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity, "cudnnSetConvolutionMathType", status);
    }
    status = cudnnSetActivationDescriptor(
        plan->activation_desc,
        relu ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY,
        CUDNN_NOT_PROPAGATE_NAN, 0.0);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity, "cudnnSetActivationDescriptor", status);
    }

    int returned = 0;
    cudnnConvolutionFwdAlgoPerf_t performance[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    status = cudnnGetConvolutionForwardAlgorithm_v7(
        cache.handle, plan->input_desc, plan->filter_desc,
        plan->convolution_desc, plan->output_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, performance);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity,
                    "cudnnGetConvolutionForwardAlgorithm_v7", status);
    }
    int deterministic = std::getenv("PP_CUDNN_NONDETERMINISTIC") == nullptr;
    int selected = -1;
    size_t limit = workspace_limit();
    for (int i = 0; i < returned; i++) {
        if (performance[i].status != CUDNN_STATUS_SUCCESS ||
            performance[i].memory > limit) {
            continue;
        }
        if (deterministic &&
            performance[i].determinism != CUDNN_DETERMINISTIC) {
            continue;
        }
        selected = i;
        break;
    }
    if (selected < 0) {
        if (error && capacity) {
            std::snprintf(error, capacity,
                          "no cuDNN algorithm fits %zu-byte workspace limit",
                          limit);
        }
        return 0;
    }
    plan->algorithm = performance[selected].algo;
    plan->workspace_bytes = performance[selected].memory;
    if (!grow_workspace(plan->workspace_bytes, error, capacity)) {
        return 0;
    }
    cache.plan_count++;
    *result = plan;
    return 1;
}

static int get_plan(pp_cudnn_plan **result, const char *name, int precise,
                    int relu, int input_channels, int output_channels,
                    int input_height, int input_width, int stride, int kernel,
                    int padding, char *error, size_t capacity)
{
    for (int i = 0; i < cache.plan_count; i++) {
        pp_cudnn_plan *plan = cache.plans + i;
        if (same_shape(plan, name, precise, relu, input_channels,
                       output_channels, input_height, input_width, stride,
                       kernel, padding)) {
            *result = plan;
            return 1;
        }
    }
    return create_plan(result, name, precise, relu, input_channels,
                       output_channels, input_height, input_width, stride,
                       kernel, padding, error, capacity);
}

static int run_conv(
    const char *plan_name, const float *weights, const float *bias,
    const float *input, float *output, int input_channels,
    int output_channels, int input_height, int input_width, int stride,
    int kernel, int padding, int relu, int precise, cudaStream_t stream,
    char *error, size_t error_capacity)
{
    if (!initialize(error, error_capacity)) {
        return 0;
    }
    cudnnStatus_t status = cudnnSetStream(cache.handle, stream);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, error_capacity, "cudnnSetStream", status);
    }
    pp_cudnn_plan *plan = nullptr;
    if (!get_plan(&plan, plan_name, precise, relu, input_channels,
                  output_channels, input_height, input_width, stride, kernel,
                  padding, error, error_capacity)) {
        return 0;
    }
    const float one = 1.0f;
    const float zero = 0.0f;
    int separate_bias =
        !relu && std::getenv("PP_CUDNN_SEPARATE_BIAS") != nullptr;
    if (!separate_bias) {
        status = cudnnConvolutionBiasActivationForward(
            cache.handle, &one, plan->input_desc, input, plan->filter_desc,
            weights, plan->convolution_desc, plan->algorithm,
            cache.workspace, plan->workspace_bytes, &zero,
            plan->output_desc, output, plan->bias_desc, bias,
            plan->activation_desc, plan->output_desc, output);
        if (status != CUDNN_STATUS_SUCCESS) {
            return fail(error, error_capacity,
                        "cudnnConvolutionBiasActivationForward", status);
        }
    } else {
        status = cudnnConvolutionForward(
            cache.handle, &one, plan->input_desc, input, plan->filter_desc,
            weights, plan->convolution_desc, plan->algorithm,
            cache.workspace, plan->workspace_bytes, &zero,
            plan->output_desc, output);
        if (status != CUDNN_STATUS_SUCCESS) {
            return fail(error, error_capacity, "cudnnConvolutionForward",
                        status);
        }
        status = cudnnAddTensor(cache.handle, &one, plan->bias_desc, bias,
                                &one, plan->output_desc, output);
        if (status != CUDNN_STATUS_SUCCESS) {
            return fail(error, error_capacity, "cudnnAddTensor bias", status);
        }
    }
    return 1;
}

extern "C" int pp_cudnn_conv(
    const pp_model *model, const uint8_t *device_weights, const char *name,
    const float *input, float *output, int input_channels,
    int output_channels, int input_height, int input_width, int stride,
    int kernel, int padding, int relu, int precise, cudaStream_t stream,
    char *error, size_t error_capacity)
{
    if (!cache.model) {
        cache.model = model;
    } else if (cache.model != model) {
        if (error && error_capacity) {
            std::snprintf(error, error_capacity, "cuDNN model changed");
        }
        return 0;
    }
    char weight_name[48];
    char bias_name[48];
    std::snprintf(weight_name, sizeof(weight_name), "%s.weight", name);
    std::snprintf(bias_name, sizeof(bias_name), "%s.bias", name);
    const pp_weight_record *weight_record = find_record(model, weight_name);
    const pp_weight_record *bias_record = find_record(model, bias_name);
    if (!weight_record || !bias_record) {
        if (error && error_capacity) {
            std::snprintf(error, error_capacity,
                          "missing cuDNN tensor for %s", name);
        }
        return 0;
    }
    const float *weights = reinterpret_cast<const float *>(
        device_weights + weight_record->offset);
    const float *bias = reinterpret_cast<const float *>(
        device_weights + bias_record->offset);

    return run_conv(name, weights, bias, input, output, input_channels,
                    output_channels, input_height, input_width, stride, kernel,
                    padding, relu, precise, stream, error, error_capacity);
}

extern "C" int pp_cudnn_conv_raw(
    const char *plan_name, const float *weights, const float *bias,
    const float *input, float *output, int input_channels,
    int output_channels, int input_height, int input_width, int stride,
    int kernel, int padding, int relu, int precise, cudaStream_t stream,
    char *error, size_t error_capacity)
{
    return run_conv(plan_name, weights, bias, input, output, input_channels,
                    output_channels, input_height, input_width, stride, kernel,
                    padding, relu, precise, stream, error, error_capacity);
}

static int same_deconv_shape(
    const pp_cudnn_cache::pp_cudnn_deconv_plan *plan, const char *name,
    int precise, int input_channels, int output_channels, int input_height,
    int input_width, int stride, int kernel)
{
    (void)name;
    return plan->precise == precise &&
           plan->input_channels == input_channels &&
           plan->output_channels == output_channels &&
           plan->input_height == input_height &&
           plan->input_width == input_width && plan->stride == stride &&
           plan->kernel == kernel;
}

static int create_deconv_plan(
    pp_cudnn_cache::pp_cudnn_deconv_plan **result, const char *name,
    int precise, int input_channels, int output_channels, int input_height,
    int input_width, int stride, int kernel, char *error, size_t capacity)
{
    if (cache.deconv_plan_count == PP_CUDNN_MAX_DECONV_PLANS) {
        if (error && capacity) {
            std::snprintf(error, capacity, "cuDNN deconv plan cache is full");
        }
        return 0;
    }
    pp_cudnn_cache::pp_cudnn_deconv_plan *plan =
        cache.deconv_plans + cache.deconv_plan_count;
    std::snprintf(plan->name, sizeof(plan->name), "%s", name);
    plan->precise = precise;
    plan->input_channels = input_channels;
    plan->output_channels = output_channels;
    plan->input_height = input_height;
    plan->input_width = input_width;
    plan->stride = stride;
    plan->kernel = kernel;

#define CUDNN_CREATE(call, label) do { \
        cudnnStatus_t create_status = (call); \
        if (create_status != CUDNN_STATUS_SUCCESS) { \
            return fail(error, capacity, (label), create_status); \
        } \
    } while (0)
    CUDNN_CREATE(cudnnCreateTensorDescriptor(&plan->input_desc),
                 "cudnnCreateTensorDescriptor deconv input");
    CUDNN_CREATE(cudnnCreateTensorDescriptor(&plan->output_desc),
                 "cudnnCreateTensorDescriptor deconv output");
    CUDNN_CREATE(cudnnCreateTensorDescriptor(&plan->bias_desc),
                 "cudnnCreateTensorDescriptor deconv bias");
    CUDNN_CREATE(cudnnCreateFilterDescriptor(&plan->filter_desc),
                 "cudnnCreateFilterDescriptor deconv");
    CUDNN_CREATE(cudnnCreateConvolutionDescriptor(&plan->convolution_desc),
                 "cudnnCreateConvolutionDescriptor deconv");
    CUDNN_CREATE(cudnnCreateActivationDescriptor(&plan->activation_desc),
                 "cudnnCreateActivationDescriptor deconv");
#undef CUDNN_CREATE

    int output_height = input_height * stride;
    int output_width = input_width * stride;
    cudnnStatus_t status = cudnnSetTensor4dDescriptor(
        plan->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
        input_channels, input_height, input_width);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity,
                    "cudnnSetTensor4dDescriptor deconv input", status);
    }
    status = cudnnSetTensor4dDescriptor(
        plan->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
        output_channels, output_height, output_width);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity,
                    "cudnnSetTensor4dDescriptor deconv output", status);
    }
    status = cudnnSetTensor4dDescriptor(
        plan->bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
        output_channels, 1, 1);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity,
                    "cudnnSetTensor4dDescriptor deconv bias", status);
    }
    status = cudnnSetFilter4dDescriptor(
        plan->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        input_channels, output_channels, kernel, kernel);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity,
                    "cudnnSetFilter4dDescriptor deconv", status);
    }
    status = cudnnSetConvolution2dDescriptor(
        plan->convolution_desc, 0, 0, stride, stride, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity,
                    "cudnnSetConvolution2dDescriptor deconv", status);
    }
    status = cudnnSetConvolutionMathType(
        plan->convolution_desc,
        precise ? CUDNN_FMA_MATH : CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity,
                    "cudnnSetConvolutionMathType deconv", status);
    }
    status = cudnnSetActivationDescriptor(
        plan->activation_desc, CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN, 0.0);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity,
                    "cudnnSetActivationDescriptor deconv", status);
    }

    int returned = 0;
    cudnnConvolutionBwdDataAlgoPerf_t
        performance[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT];
    status = cudnnGetConvolutionBackwardDataAlgorithm_v7(
        cache.handle, plan->filter_desc, plan->input_desc,
        plan->convolution_desc, plan->output_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &returned, performance);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, capacity,
                    "cudnnGetConvolutionBackwardDataAlgorithm_v7", status);
    }
    int deterministic = std::getenv("PP_CUDNN_NONDETERMINISTIC") == nullptr;
    int selected = -1;
    size_t limit = workspace_limit();
    for (int i = 0; i < returned; i++) {
        if (performance[i].status != CUDNN_STATUS_SUCCESS ||
            performance[i].memory > limit) {
            continue;
        }
        if (deterministic &&
            performance[i].determinism != CUDNN_DETERMINISTIC) {
            continue;
        }
        selected = i;
        break;
    }
    if (selected < 0) {
        if (error && capacity) {
            std::snprintf(error, capacity,
                          "no cuDNN deconv algorithm fits %zu-byte limit",
                          limit);
        }
        return 0;
    }
    plan->algorithm = performance[selected].algo;
    plan->workspace_bytes = performance[selected].memory;
    if (!grow_workspace(plan->workspace_bytes, error, capacity)) {
        return 0;
    }
    cache.deconv_plan_count++;
    *result = plan;
    return 1;
}

static int get_deconv_plan(
    pp_cudnn_cache::pp_cudnn_deconv_plan **result, const char *name,
    int precise, int input_channels, int output_channels, int input_height,
    int input_width, int stride, int kernel, char *error, size_t capacity)
{
    for (int i = 0; i < cache.deconv_plan_count; i++) {
        pp_cudnn_cache::pp_cudnn_deconv_plan *plan =
            cache.deconv_plans + i;
        if (same_deconv_shape(plan, name, precise, input_channels,
                              output_channels, input_height, input_width,
                              stride, kernel)) {
            *result = plan;
            return 1;
        }
    }
    return create_deconv_plan(result, name, precise, input_channels,
                              output_channels, input_height, input_width,
                              stride, kernel, error, capacity);
}

extern "C" int pp_cudnn_deconv(
    const pp_model *model, const uint8_t *device_weights, const char *name,
    const float *input, float *output, int input_channels,
    int output_channels, int input_height, int input_width, int stride,
    int kernel, int precise, cudaStream_t stream, char *error,
    size_t error_capacity)
{
    if (!initialize(error, error_capacity)) {
        return 0;
    }
    cudnnStatus_t status = cudnnSetStream(cache.handle, stream);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, error_capacity, "cudnnSetStream deconv", status);
    }
    char weight_name[48];
    char bias_name[48];
    std::snprintf(weight_name, sizeof(weight_name), "%s.weight", name);
    std::snprintf(bias_name, sizeof(bias_name), "%s.bias", name);
    const pp_weight_record *weight_record = find_record(model, weight_name);
    const pp_weight_record *bias_record = find_record(model, bias_name);
    if (!weight_record || !bias_record) {
        if (error && error_capacity) {
            std::snprintf(error, error_capacity,
                          "missing cuDNN deconv tensor for %s", name);
        }
        return 0;
    }
    const float *weights = reinterpret_cast<const float *>(
        device_weights + weight_record->offset);
    const float *bias = reinterpret_cast<const float *>(
        device_weights + bias_record->offset);
    pp_cudnn_cache::pp_cudnn_deconv_plan *plan = nullptr;
    if (!get_deconv_plan(&plan, name, precise, input_channels,
                         output_channels, input_height, input_width, stride,
                         kernel, error, error_capacity)) {
        return 0;
    }
    const float one = 1.0f;
    const float zero = 0.0f;
    status = cudnnConvolutionBackwardData(
        cache.handle, &one, plan->filter_desc, weights,
        plan->input_desc, input, plan->convolution_desc, plan->algorithm,
        cache.workspace, plan->workspace_bytes, &zero,
        plan->output_desc, output);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, error_capacity,
                    "cudnnConvolutionBackwardData", status);
    }
    status = cudnnAddTensor(cache.handle, &one, plan->bias_desc, bias,
                            &one, plan->output_desc, output);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, error_capacity, "cudnnAddTensor deconv bias",
                    status);
    }
    status = cudnnActivationForward(
        cache.handle, plan->activation_desc, &one, plan->output_desc, output,
        &zero, plan->output_desc, output);
    if (status != CUDNN_STATUS_SUCCESS) {
        return fail(error, error_capacity, "cudnnActivationForward deconv",
                    status);
    }
    return 1;
}

extern "C" size_t pp_cudnn_workspace_bytes(void)
{
    return cache.workspace_bytes;
}
