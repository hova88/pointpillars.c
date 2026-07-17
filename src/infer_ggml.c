#include "pp_ggml.h"

#include <ggml-cpu.h>
#include <ggml.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

enum { PP_GGML_SLOTS = 2, PP_GGML_META_BYTES = 1024 * 1024 };

typedef struct {
    struct ggml_context *context;
    struct ggml_cgraph *graph;
    struct ggml_tensor *input;
    struct ggml_tensor *weights;
    struct ggml_tensor *output;
    struct ggml_cplan plan;
    int ci, co, hi, wi, stride, threads;
} pp_ggml_slot;

static _Thread_local pp_ggml_slot slots[PP_GGML_SLOTS];
static _Thread_local uint8_t *work;
static _Thread_local size_t work_bytes;

static int thread_count(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static int slot_index(int ci, int co, int hi, int wi, int stride) {
    if (ci == 64 && co == 128 && hi == 256 && wi == 256 && stride == 2) return 0;
    if (ci == 128 && co == 256 && hi == 128 && wi == 128 && stride == 2) return 1;
    return -1;
}

static int init_slot(pp_ggml_slot *slot, int ci, int co, int hi, int wi,
                     int stride, int threads) {
    struct ggml_init_params params = {
        .mem_size = PP_GGML_META_BYTES,
        .mem_buffer = NULL,
        .no_alloc = true
    };
    slot->context = ggml_init(params);
    if (!slot->context) return 0;
    slot->input = ggml_new_tensor_4d(slot->context, GGML_TYPE_F32,
                                     wi, hi, ci, 1);
    slot->weights = ggml_new_tensor_4d(slot->context, GGML_TYPE_F32,
                                       3, 3, ci, co);
    slot->output = ggml_conv_2d_direct(slot->context, slot->weights,
                                       slot->input, stride, stride,
                                       1, 1, 1, 1);
    slot->graph = ggml_new_graph(slot->context);
    ggml_build_forward_expand(slot->graph, slot->output);
    slot->plan = ggml_graph_plan(slot->graph, threads, NULL);
    if (slot->plan.work_size > work_bytes) {
        uint8_t *grown = realloc(work, slot->plan.work_size);
        if (!grown) {
            ggml_free(slot->context);
            memset(slot, 0, sizeof(*slot));
            return 0;
        }
        work = grown;
        work_bytes = slot->plan.work_size;
    }
    slot->ci = ci; slot->co = co; slot->hi = hi; slot->wi = wi;
    slot->stride = stride; slot->threads = threads;
    return 1;
}

int pp_ggml_conv3_relu(const float *input, float *output,
                       const float *weights, const float *bias,
                       int ci, int co, int hi, int wi, int stride) {
    int index = slot_index(ci, co, hi, wi, stride);
    if (index < 0) return 0;
    int threads = thread_count();
    pp_ggml_slot *slot = slots + index;
    if (!slot->context && !init_slot(slot, ci, co, hi, wi, stride, threads))
        return 0;
    if (slot->threads != threads) return 0;
    slot->input->data = (void *)input;
    slot->weights->data = (void *)weights;
    slot->output->data = output;
    slot->plan.work_data = work;
    if (ggml_graph_compute(slot->graph, &slot->plan) != GGML_STATUS_SUCCESS)
        return 0;
    size_t plane = (size_t)(hi / stride) * (wi / stride);
    #pragma omp parallel for schedule(static)
    for (int oc = 0; oc < co; ++oc) {
        float *dst = output + (size_t)oc * plane;
        for (size_t i = 0; i < plane; ++i) {
            float value = dst[i] + bias[oc];
            dst[i] = value > 0.0f ? value : 0.0f;
        }
    }
    return 1;
}

size_t pp_ggml_workspace_bytes(void) {
    size_t initialized = 0;
    for (int i = 0; i < PP_GGML_SLOTS; ++i)
        initialized += slots[i].context != NULL;
    return work_bytes + initialized * PP_GGML_META_BYTES;
}
