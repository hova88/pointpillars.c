#include "pointpillars.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const float points[][5] = {
        {1.01f, 2.01f, 0.0f, .5f, .1f}, {1.03f, 2.02f, .2f, .7f, .2f},
        {-1, 0, 0, 1, 0}, {68, -30, 0, .2f, 0}
    };
    pp_pillars p; pp_voxel_stats s;
    if (!pp_pillars_alloc(&p) || !pp_voxelize(&points[0][0], 4, 5, &p, &s)) return 1;
    if (p.pillar_count != 2 || s.accepted_points != 3 || s.clipped_points != 1) return 2;
    float *a = p.features;
    if (fabsf(a[5] + .01f) > 1e-5f || fabsf(a[6] + .005f) > 1e-5f ||
        fabsf(a[7] + .1f) > 1e-5f) return 3;
    if (fabsf(a[3]-.5f)>1e-7f||fabsf(a[4]-.1f)>1e-7f) return 5;
    if (p.coords[1] != 0 || p.coords[0] != 0) return 4;
    const float next[][5] = {{20, -10, 1, .3f, .4f}};
    if (!pp_voxelize(&next[0][0], 1, 5, &p, &s) || p.pillar_count != 1 ||
        p.point_count[0] != 1 || p.features[0] != 20 || p.features[11] != 0) return 6;
    pp_pillars_free(&p);
    puts("voxelizer: ok");
    return 0;
}
