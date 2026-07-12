#ifndef POINTPILLARS_H
#define POINTPILLARS_H

#include <stddef.h>
#include <stdint.h>

enum {
    PP_MAX_PILLARS = 30000,
    PP_MAX_POINTS = 20,
    PP_POINT_FEATURES = 11,
    PP_GRID_X = 512,
    PP_GRID_Y = 512
};

typedef struct {
    float *features;       /* [PP_MAX_PILLARS, PP_MAX_POINTS, 11] */
    int32_t *coords;       /* [PP_MAX_PILLARS, 4] = batch,z,y,x */
    uint8_t *point_count;  /* [PP_MAX_PILLARS] */
    int32_t *grid;         /* persistent [PP_GRID_Y, PP_GRID_X] sparse index */
    int32_t pillar_count;
} pp_pillars;

typedef struct {
    uint64_t input_points, accepted_points, clipped_points, dropped_pillars;
} pp_voxel_stats;

int pp_pillars_alloc(pp_pillars *p);
void pp_pillars_free(pp_pillars *p);
int pp_voxelize(const float *points, size_t count, size_t stride,
                pp_pillars *out, pp_voxel_stats *stats);
int pp_load_points(const char *path, size_t stride, float **points,
                   size_t *count, char *error, size_t error_size);

#endif
