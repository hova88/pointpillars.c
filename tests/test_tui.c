#include "pp_tui.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    pp_tui_state state;
    pp_box boxes[2] = {{.x=1,.y=2,.class_id=0}, {.x=-4,.y=3,.class_id=8}};
    pp_detections detections = {boxes, 2, 2};
    memset(&state, 0, sizeof(state));
    pp_tui_update_tracks(&state, &detections, 0);
    if (state.track_count != 2 || state.tracks[0].length != 1) return 1;
    boxes[0].x += 1; boxes[1].y += 1;
    pp_tui_update_tracks(&state, &detections, 1);
    if (state.track_count != 2 || state.tracks[0].length != 2) return 2;
    pp_tui_update_tracks(&state, &detections, 0);
    if (state.track_count != 2 || state.tracks[0].length != 1) return 3;
    detections.count = 0;
    for (size_t frame = 1; frame <= 4; ++frame)
        pp_tui_update_tracks(&state, &detections, frame);
    if (state.track_count) return 4;
    puts("tui tracks: ok");
    return 0;
}
