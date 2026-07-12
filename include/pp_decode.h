#ifndef PP_DECODE_H
#define PP_DECODE_H
#include "pp_infer.h"
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { float x,y,z,dx,dy,dz,yaw,vx,vy,score; int class_id; } pp_box;
typedef struct { pp_box *boxes; size_t count, capacity; } pp_detections;
typedef struct {
    float code[10], logit;
    uint32_t position;
    uint8_t head, anchor, class_id, reserved;
} pp_compact_candidate;

int pp_detections_alloc(pp_detections *d, size_t capacity);
void pp_detections_free(pp_detections *d);
int pp_decode(const pp_raw_output *raw, float score_threshold,
              float nms_threshold, pp_detections *out);
int pp_decode_compact(const pp_compact_candidate *candidates, size_t count,
                      float score_threshold, float nms_threshold,
                      pp_detections *out);
#ifdef __cplusplus
}
#endif
#endif
