#define _POSIX_C_SOURCE 200809L
#include "pointpillars.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const float X_MIN = -51.2f, Y_MIN = -51.2f, Z_MIN = -5.0f;
static const float X_MAX = 51.2f, Y_MAX = 51.2f, Z_MAX = 3.0f;
static const float VX = 0.2f, VY = 0.2f, VZ = 8.0f;

int pp_pillars_alloc(pp_pillars *p) {
    memset(p, 0, sizeof(*p));
    size_t nf = (size_t)PP_MAX_PILLARS * PP_MAX_POINTS * PP_POINT_FEATURES;
    p->features = (float *)calloc(nf, sizeof(float));
    p->coords = (int32_t *)calloc((size_t)PP_MAX_PILLARS * 4, sizeof(int32_t));
    p->point_count = (uint8_t *)calloc(PP_MAX_PILLARS, 1);
    if (p->features && p->coords && p->point_count) return 1;
    pp_pillars_free(p); return 0;
}

void pp_pillars_free(pp_pillars *p) {
    if (!p) return;
    free(p->features); free(p->coords); free(p->point_count);
    memset(p, 0, sizeof(*p));
}

int pp_voxelize(const float *points, size_t count, size_t stride,
                pp_pillars *out, pp_voxel_stats *stats) {
        if (!points || !out || !out->features || !out->coords ||
        !out->point_count || stride != 5) return 0;
    pp_voxel_stats local = {0};
    int32_t *grid = (int32_t *)malloc((size_t)PP_GRID_X * PP_GRID_Y * sizeof(int32_t));
    if (!grid) return 0;
    for (size_t i = 0; i < (size_t)PP_GRID_X * PP_GRID_Y; ++i) grid[i] = -1;
    memset(out->features, 0, (size_t)PP_MAX_PILLARS * PP_MAX_POINTS * PP_POINT_FEATURES * sizeof(float));
    memset(out->coords, 0, (size_t)PP_MAX_PILLARS * 4 * sizeof(int32_t));
    memset(out->point_count, 0, PP_MAX_PILLARS);
    out->pillar_count = 0; local.input_points = count;

    for (size_t n = 0; n < count; ++n) {
        const float *q = points + n * stride;
        float x = q[0], y = q[1], z = q[2];
        if (!(x >= X_MIN && x < X_MAX && y >= Y_MIN && y < Y_MAX && z >= Z_MIN && z < Z_MAX)) {
            ++local.clipped_points; continue;
        }
        int ix = (int)((x - X_MIN) / VX), iy = (int)((y - Y_MIN) / VY);
        size_t cell = (size_t)iy * PP_GRID_X + ix;
        int pi = grid[cell];
        if (pi < 0) {
            if (out->pillar_count == PP_MAX_PILLARS) { ++local.dropped_pillars; continue; }
            pi = out->pillar_count++; grid[cell] = pi;
            int32_t *c = out->coords + (size_t)pi * 4;
            c[0] = 0; c[1] = 0; c[2] = iy; c[3] = ix;
        }
        unsigned pn = out->point_count[pi];
        if (pn == PP_MAX_POINTS) continue;
        out->point_count[pi] = (uint8_t)(pn + 1); ++local.accepted_points;
        float *f = out->features + ((size_t)pi * PP_MAX_POINTS + pn) * PP_POINT_FEATURES;
        f[0] = x; f[1] = y; f[2] = z; f[3] = q[3]; f[4] = q[4];
    }

    for (int pi = 0; pi < out->pillar_count; ++pi) {
        unsigned np = out->point_count[pi];
        float mx = 0, my = 0, mz = 0;
        for (unsigned j = 0; j < np; ++j) {
            float *f = out->features + ((size_t)pi * PP_MAX_POINTS + j) * PP_POINT_FEATURES;
            mx += f[0]; my += f[1]; mz += f[2];
        }
        mx /= np; my /= np; mz /= np;
        int ix = out->coords[(size_t)pi * 4 + 3], iy = out->coords[(size_t)pi * 4 + 2];
        float cx = X_MIN + (ix + 0.5f) * VX;
        float cy = Y_MIN + (iy + 0.5f) * VY;
        float cz = Z_MIN + 0.5f * VZ;
        for (unsigned j = 0; j < np; ++j) {
            float *f = out->features + ((size_t)pi * PP_MAX_POINTS + j) * PP_POINT_FEATURES;
            f[5] = f[0] - mx; f[6] = f[1] - my; f[7] = f[2] - mz;
            f[8] = f[0] - cx; f[9] = f[1] - cy; f[10] = f[2] - cz;
        }
    }
    free(grid);
    if (stats) *stats = local;
    return 1;
}

int pp_load_points(const char *path, size_t stride, float **points,
                   size_t *count, char *error, size_t cap) {
    if (stride != 5 || !points || !count) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) { if (error && cap) snprintf(error, cap, "%s: %s", path, strerror(errno)); return 0; }
    struct stat st;
    if (fstat(fileno(f), &st) || st.st_size < 0 ||
        (uint64_t)st.st_size % (stride * sizeof(float))) {
        fclose(f); if (error && cap) snprintf(error, cap, "invalid point file size/stride"); return 0;
    }
    size_t n = (size_t)st.st_size / (stride * sizeof(float));
    if (n > SIZE_MAX / stride / sizeof(float)) { fclose(f); return 0; }
    float *p = (float *)malloc((size_t)st.st_size);
    if (!p || fread(p, sizeof(float) * stride, n, f) != n) {
        free(p); fclose(f); if (error && cap) snprintf(error, cap, "failed to read point file"); return 0;
    }
    fclose(f); *points = p; *count = n; return 1;
}
