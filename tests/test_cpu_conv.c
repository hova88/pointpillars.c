#include "pp_kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { int ci, co, h, w, stride; } fixture;

static uint32_t random_state = 0x31415926u;

static float next_value(void) {
    random_state = random_state * 1664525u + 1013904223u;
    return ((float)(random_state >> 8) / 8388608.0f - 1.0f) * 0.25f;
}

static void reference_conv3(const float *x, float *y, const float *weights,
                            const float *bias, const fixture *f, int relu) {
    int ho = (f->h + f->stride - 1) / f->stride;
    int wo = (f->w + f->stride - 1) / f->stride;
    size_t in_plane = (size_t)f->h * f->w;
    size_t out_plane = (size_t)ho * wo;
    for (int oc = 0; oc < f->co; ++oc) for (int oy = 0; oy < ho; ++oy)
        for (int ox = 0; ox < wo; ++ox) {
            float sum = bias[oc];
            for (int ic = 0; ic < f->ci; ++ic) for (int ky = 0; ky < 3; ++ky)
                for (int kx = 0; kx < 3; ++kx) {
                    int iy = oy * f->stride + ky - 1;
                    int ix = ox * f->stride + kx - 1;
                    if ((unsigned)iy < (unsigned)f->h &&
                        (unsigned)ix < (unsigned)f->w)
                        sum += x[(size_t)ic * in_plane + (size_t)iy * f->w + ix] *
                               weights[((size_t)oc * f->ci + ic) * 9 + ky * 3 + kx];
                }
            y[(size_t)oc * out_plane + (size_t)oy * wo + ox] = relu && sum < 0.0f ? 0.0f : sum;
        }
}

static int run_fixture(const fixture *f, int id) {
    int ho = (f->h + f->stride - 1) / f->stride;
    int wo = (f->w + f->stride - 1) / f->stride;
    size_t xn = (size_t)f->ci * f->h * f->w;
    size_t wn = (size_t)f->co * f->ci * 9;
    size_t yn = (size_t)f->co * ho * wo;
    float *x = malloc(xn * sizeof(*x));
    float *weights = malloc(wn * sizeof(*weights));
    float *bias = malloc((size_t)f->co * sizeof(*bias));
    float *reference = malloc(yn * sizeof(*reference));
    float *actual = malloc(yn * sizeof(*actual));
    if (!x || !weights || !bias || !reference || !actual) {
        free(x); free(weights); free(bias); free(reference); free(actual);
        return 10 + id;
    }
    for (size_t i = 0; i < xn; ++i) x[i] = next_value();
    for (size_t i = 0; i < wn; ++i) weights[i] = next_value();
    for (int i = 0; i < f->co; ++i) bias[i] = next_value();
    reference_conv3(x, reference, weights, bias, f, 1);
    pp_cpu_conv3_relu(x, actual, weights, bias, f->ci, f->co,
                      f->h, f->w, f->stride);
    float max_abs = 0.0f;
    for (size_t i = 0; i < yn; ++i) {
        float delta = fabsf(reference[i] - actual[i]);
        if (delta > max_abs) max_abs = delta;
        if (!(delta <= 2e-5f + 2e-5f * fabsf(reference[i]))) {
            fprintf(stderr, "fixture %d mismatch at %zu: reference %.9g actual %.9g\n",
                    id, i, reference[i], actual[i]);
            free(x); free(weights); free(bias); free(reference); free(actual);
            return 20 + id;
        }
    }
    float relu_max_abs = max_abs;
    if (f->stride == 1) {
        reference_conv3(x, reference, weights, bias, f, 0);
        pp_cpu_conv3_plain(x, actual, weights, bias, f->ci, f->co, f->h, f->w);
        max_abs = 0.0f;
        for (size_t i = 0; i < yn; ++i) {
            float delta = fabsf(reference[i] - actual[i]);
            if (delta > max_abs) max_abs = delta;
            if (!(delta <= 2e-5f + 2e-5f * fabsf(reference[i]))) {
                fprintf(stderr, "plain fixture %d mismatch at %zu: reference %.9g actual %.9g\n",
                        id, i, reference[i], actual[i]);
                free(x); free(weights); free(bias); free(reference); free(actual);
                return 30 + id;
            }
        }
    }
    printf("cpu conv fixture %d: ci=%d co=%d %dx%d s%d relu=%.3g plain=%.3g\n",
           id, f->ci, f->co, f->h, f->w, f->stride, relu_max_abs,
           f->stride == 1 ? max_abs : 0.0f);
    free(x); free(weights); free(bias); free(reference); free(actual);
    return 0;
}

int main(void) {
    static const fixture fixtures[] = {
        {2, 8, 3, 3, 1},
        {3, 4, 5, 7, 1},
        {3, 8, 5, 13, 1},
        {3, 8, 6, 14, 2},
        {64, 64, 8, 11, 1},
        {64, 64, 9, 12, 2}
    };
    for (size_t i = 0; i < sizeof(fixtures) / sizeof(fixtures[0]); ++i) {
        int result = run_fixture(fixtures + i, (int)i);
        if (result) return result;
    }
    puts("cpu conv fixtures: ok");
    return 0;
}
