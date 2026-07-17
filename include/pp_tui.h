#ifndef PP_TUI_H
#define PP_TUI_H
#include "pp_decode.h"
#include <stddef.h>

enum { PP_TUI_MAX_TRACKS=64, PP_TUI_TRAIL=12 };
typedef struct {
    float x[PP_TUI_TRAIL], y[PP_TUI_TRAIL];
    unsigned char length, class_id, missed;
    int id;
} pp_tui_track;

typedef struct {
    float center_x, center_y, zoom, yaw;
    float score_threshold;
    unsigned class_mask;
    int paused, show_boxes, show_velocity, show_grid, show_tracks, selected;
    pp_tui_track tracks[PP_TUI_MAX_TRACKS];
    size_t track_count, last_frame;
    int next_track_id, have_last_frame;
} pp_tui_state;

enum { PP_TUI_NONE, PP_TUI_REDRAW, PP_TUI_NEXT, PP_TUI_PREV, PP_TUI_QUIT };

int pp_tui_begin(pp_tui_state *state);
void pp_tui_end(void);
int pp_tui_poll(pp_tui_state *state, int timeout_ms);
void pp_tui_update_tracks(pp_tui_state *state,
                          const pp_detections *detections, size_t frame);
void pp_tui_render(const float *points, size_t count, size_t stride,
                   const pp_detections *detections, size_t frame,
                   size_t frame_count, double inference_ms,
                   const char *backend, const pp_tui_state *state);
#endif
