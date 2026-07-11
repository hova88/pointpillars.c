#ifndef PP_TUI_H
#define PP_TUI_H
#include "pp_decode.h"
#include <stddef.h>
void pp_tui_render(const float *points,size_t count,size_t stride,
                   const pp_detections *detections,size_t frame,
                   double inference_ms,const char *backend);
#endif
