#define _POSIX_C_SOURCE 200809L
#include "pp_apple.h"

#include <Accelerate/Accelerate.h>
#include <stdlib.h>
#include <string.h>

/* The filter API remains available on current macOS and supports the exact
 * NCHW/OIHW contract without graph IR construction. Keep the compatibility
 * adapter isolated here so a future BNNSGraph migration is one-file scoped. */
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

enum { PP_APPLE_MAX_FILTERS = 96 };

typedef struct {
    const float *weights;
    const float *bias;
    int ci, co, hi, wi, kernel, stride, padding, relu, transposed;
    BNNSFilter filter;
    float *packed_weights;
} pp_apple_filter;

/* Filters hold backend-owned plans and are not shared across caller threads.
 * Thread-local caches keep concurrent library users race-free. */
static _Thread_local pp_apple_filter cache[PP_APPLE_MAX_FILTERS];
static _Thread_local size_t cache_count;

static BNNSNDArrayDescriptor image_desc(int width, int height, int channels,
                                        void *data) {
    BNNSNDArrayDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = BNNSDataLayoutImageCHW;
    d.size[0] = (size_t)width;
    d.size[1] = (size_t)height;
    d.size[2] = (size_t)channels;
    d.data = data;
    d.data_type = BNNSDataTypeFloat32;
    return d;
}

static BNNSNDArrayDescriptor weight_desc(int kernel, int ci, int co,
                                         const float *weights, int transposed) {
    BNNSNDArrayDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = transposed ? BNNSDataLayoutConvolutionWeightsIOHrWr
                          : BNNSDataLayoutConvolutionWeightsOIHW;
    d.size[0] = (size_t)kernel;
    d.size[1] = (size_t)kernel;
    d.size[2] = (size_t)(transposed ? co : ci);
    d.size[3] = (size_t)(transposed ? ci : co);
    d.data = (void *)weights;
    d.data_type = BNNSDataTypeFloat32;
    return d;
}

static BNNSNDArrayDescriptor bias_desc(int co, const float *bias) {
    BNNSNDArrayDescriptor d;
    memset(&d, 0, sizeof(d));
    d.layout = BNNSDataLayoutVector;
    d.size[0] = (size_t)co;
    d.data = (void *)bias;
    d.data_type = BNNSDataTypeFloat32;
    return d;
}

static BNNSFilter create_filter(pp_apple_filter *key) {
    int ho = key->transposed ? (key->hi - 1) * key->stride + key->kernel
                             : (key->hi + 2 * key->padding - key->kernel) / key->stride + 1;
    int wo = key->transposed ? (key->wi - 1) * key->stride + key->kernel
                             : (key->wi + 2 * key->padding - key->kernel) / key->stride + 1;
    BNNSLayerParametersConvolution p;
    memset(&p, 0, sizeof(p));
    p.i_desc = image_desc(key->wi, key->hi, key->ci, NULL);
    if (key->transposed) {
        size_t count = (size_t)key->ci * key->co * key->kernel * key->kernel;
        key->packed_weights = (float *)malloc(count * sizeof(float));
        if (!key->packed_weights) return NULL;
        for (int ic = 0; ic < key->ci; ++ic) for (int oc = 0; oc < key->co; ++oc)
            for (int ky = 0; ky < key->kernel; ++ky)
                for (int kx = 0; kx < key->kernel; ++kx) {
                    size_t dst = (((size_t)ic * key->co + oc) * key->kernel + ky) *
                                 key->kernel + kx;
                    size_t src = (((size_t)ic * key->co + oc) * key->kernel +
                                  (key->kernel - 1 - ky)) * key->kernel +
                                 (key->kernel - 1 - kx);
                    key->packed_weights[dst] = key->weights[src];
                }
    }
    p.w_desc = weight_desc(key->kernel, key->ci, key->co,
                           key->packed_weights ? key->packed_weights : key->weights,
                           key->transposed);
    p.o_desc = image_desc(wo, ho, key->co, NULL);
    p.bias = bias_desc(key->co, key->bias);
    p.activation.function = key->relu ? BNNSActivationFunctionRectifiedLinear
                                      : BNNSActivationFunctionIdentity;
    p.x_stride = p.y_stride = (size_t)key->stride;
    p.x_padding = p.y_padding = (size_t)key->padding;
    BNNSFilter result = key->transposed ?
        BNNSFilterCreateLayerTransposedConvolution(&p, NULL) :
        BNNSFilterCreateLayerConvolution(&p, NULL);
    if (!result) { free(key->packed_weights); key->packed_weights = NULL; }
    return result;
}

static int same_filter(const pp_apple_filter *a, const pp_apple_filter *b) {
    return a->weights == b->weights && a->bias == b->bias &&
           a->ci == b->ci && a->co == b->co && a->hi == b->hi &&
           a->wi == b->wi && a->kernel == b->kernel &&
           a->stride == b->stride && a->padding == b->padding &&
           a->relu == b->relu && a->transposed == b->transposed;
}

int pp_apple_conv(const float *input, float *output, const float *weights,
                  const float *bias, int ci, int co, int hi, int wi,
                  int kernel, int stride, int padding, int relu) {
    if (!input || !output || !weights || !bias || getenv("PP_APPLE_DISABLE"))
        return 0;
    const char *minimum_spatial = getenv("PP_APPLE_MIN_SPATIAL");
    if (minimum_spatial && hi < atoi(minimum_spatial)) return 0;
    /* BNNS' 2x2/s2 path is fast but fails the checkpoint-level FP32 oracle on
     * Apple M2. Keep the single deblock downsampler on canonical C unless the
     * explicitly approximate experiment is requested. */
    if (kernel == 2 && getenv("PP_APPLE_CONV2") == NULL) return 0;
    pp_apple_filter key = {weights, bias, ci, co, hi, wi, kernel, stride,
                           padding, relu, 0, NULL, NULL};
    for (size_t i = 0; i < cache_count; ++i)
        if (same_filter(cache + i, &key))
            return BNNSFilterApply(cache[i].filter, input, output) == 0;
    if (cache_count == PP_APPLE_MAX_FILTERS) return 0;
    key.filter = create_filter(&key);
    if (!key.filter) return 0;
    cache[cache_count++] = key;
    return BNNSFilterApply(key.filter, input, output) == 0;
}

int pp_apple_deconv(const float *input, float *output, const float *weights,
                    const float *bias, int ci, int co, int hi, int wi,
                    int kernel, int stride, int relu) {
    if (!input || !output || !weights || !bias || getenv("PP_APPLE_DISABLE") ||
        getenv("PP_APPLE_DECONV_DISABLE")) return 0;
    pp_apple_filter key = {weights, bias, ci, co, hi, wi, kernel, stride,
                           0, relu, 1, NULL, NULL};
    for (size_t i = 0; i < cache_count; ++i)
        if (same_filter(cache + i, &key))
            return BNNSFilterApply(cache[i].filter, input, output) == 0;
    if (cache_count == PP_APPLE_MAX_FILTERS) return 0;
    key.filter = create_filter(&key);
    if (!key.filter) return 0;
    cache[cache_count++] = key;
    return BNNSFilterApply(key.filter, input, output) == 0;
}

size_t pp_apple_cache_bytes(void) {
    size_t bytes = cache_count * sizeof(cache[0]);
    for (size_t i = 0; i < cache_count; ++i) if (cache[i].packed_weights)
        bytes += (size_t)cache[i].ci * cache[i].co * cache[i].kernel *
                 cache[i].kernel * sizeof(float);
    return bytes;
}
