#ifndef PP_DECODE_H
#define PP_DECODE_H
#include "pp_infer.h"
#include <stddef.h>

typedef struct { float x,y,z,dx,dy,dz,yaw,vx,vy,score; int class_id; } pp_box;
typedef struct { pp_box *boxes; size_t count, capacity; } pp_detections;

int pp_detections_alloc(pp_detections *d, size_t capacity);
void pp_detections_free(pp_detections *d);
int pp_decode(const pp_raw_output *raw, float score_threshold,
              float nms_threshold, pp_detections *out);
#endif
