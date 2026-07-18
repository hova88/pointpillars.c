#define _DEFAULT_SOURCE
#include "pp_tui.h"
#include <errno.h>
#include <poll.h>
#ifdef __APPLE__
#include <util.h>
#else
#include <pty.h>
#endif
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

static int same_termios(const struct termios *left, const struct termios *right) {
    return left->c_iflag == right->c_iflag && left->c_oflag == right->c_oflag &&
           left->c_cflag == right->c_cflag && left->c_lflag == right->c_lflag &&
           !memcmp(left->c_cc, right->c_cc, sizeof(left->c_cc));
}

static int test_terminal_session(void) {
    int master = -1;
    struct winsize size = {.ws_row = 24, .ws_col = 80};
    pid_t child = forkpty(&master, NULL, NULL, &size);
    if (child < 0) return 0;
    if (!child) {
        struct termios before;
        struct termios after;
        pp_tui_state state;
        if (tcgetattr(STDIN_FILENO, &before) || !pp_tui_begin(&state)) _exit(20);
        int action;
        do action = pp_tui_poll(&state, 3000); while (action == PP_TUI_NONE);
        if (action != PP_TUI_NEXT) _exit(22);
        do action = pp_tui_poll(&state, 3000); while (action == PP_TUI_NONE);
        pp_tui_end();
        if (tcgetattr(STDIN_FILENO, &after) || !same_termios(&before, &after)) _exit(21);
        _exit(action == PP_TUI_QUIT ? 0 : 23);
    }

    char transcript[4096];
    size_t length = 0;
    int input_stage = 0;
    for (int attempt = 0; attempt < 40; ++attempt) {
        struct pollfd descriptor = {master, POLLIN, 0};
        int ready = poll(&descriptor, 1, 100);
        if (ready > 0 && (descriptor.revents & (POLLIN | POLLHUP))) {
            ssize_t count = read(master, transcript + length,
                                 sizeof(transcript) - length - 1);
            if (count > 0) {
                length += (size_t)count;
                transcript[length] = '\0';
            } else if (count < 0 && errno != EIO && errno != EINTR) {
                break;
            }
        }
        if (!input_stage && length && strstr(transcript, "\033[?1049h")) {
            if (write(master, "\033", 1) != 1) break;
            input_stage = 1;
        } else if (input_stage == 1) {
            if (write(master, "[Cq", 3) != 3) break;
            input_stage = 2;
        }
        int status = 0;
        pid_t result = waitpid(child, &status, WNOHANG);
        if (result == child) {
            while (length + 1 < sizeof(transcript)) {
                ssize_t count = read(master, transcript + length,
                                     sizeof(transcript) - length - 1);
                if (count <= 0) break;
                length += (size_t)count;
                transcript[length] = '\0';
            }
            close(master);
            return input_stage == 2 && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
                   strstr(transcript, "\033[?25l") &&
                   strstr(transcript, "\033[0m\033[?25h\033[?1049l");
        }
    }
    kill(child, SIGKILL);
    waitpid(child, NULL, 0);
    close(master);
    return 0;
}

static int validate_layout(const pp_tui_frame *frame, int columns, int rows) {
    int row = 0;
    int width = 0;
    const unsigned char *cursor = (const unsigned char *)frame->data;
    const unsigned char *end = cursor + frame->length;
    while (cursor < end) {
        if (*cursor == 27 && cursor + 1 < end && cursor[1] == '[') {
            cursor += 2;
            while (cursor < end && !(*cursor >= '@' && *cursor <= '~')) ++cursor;
            if (cursor < end) ++cursor;
            continue;
        }
        if (*cursor == '\n') {
            if (width > columns) {
                fprintf(stderr, "TUI row %d is %d cells, expected at most %d\n",
                        row, width, columns);
                return 0;
            }
            ++row;
            width = 0;
            ++cursor;
            continue;
        }
        if ((*cursor & 0xc0u) != 0x80u) ++width;
        ++cursor;
    }
    if (row + 1 != rows || width > columns)
        fprintf(stderr, "TUI final layout rows=%d width=%d\n", row + 1, width);
    return row + 1 == rows && width <= columns;
}

int main(void) {
    if (!test_terminal_session()) {
        fputs("TUI PTY session did not restore terminal state\n", stderr);
        return 1;
    }
    pp_tui_state state;
    pp_box boxes[2] = {
        {.x=1,.y=2,.z=-.4f,.dx=4.2f,.dy=1.8f,.dz=1.6f,
         .yaw=.2f,.vx=1.2f,.vy=.1f,.score=.91f,.class_id=0},
        {.x=-4,.y=3,.z=-.5f,.dx=.8f,.dy=.7f,.dz=1.7f,
         .yaw=-.3f,.vx=.2f,.vy=.3f,.score=.78f,.class_id=8}
    };
    pp_detections detections = {boxes, 2, 2};
    pp_tui_state_init(&state);
    if (!state.show_points || !state.show_boxes || !state.show_velocity ||
        !state.show_grid || !state.show_tracks || state.class_mask != 0x3ffu ||
        !state.perspective || state.show_sidebar ||
        state.zoom != 1.0f) return 2;
    pp_tui_update_tracks(&state, &detections, 0);
    if (state.track_count != 2 || state.tracks[0].length != 1) return 3;
    boxes[0].x += 1; boxes[1].y += 1;
    pp_tui_update_tracks(&state, &detections, 1);
    if (state.track_count != 2 || state.tracks[0].length != 2) return 4;
    pp_tui_update_tracks(&state, &detections, 0);
    if (state.track_count != 2 || state.tracks[0].length != 1) return 5;

    float points[] = {
        0, 0, -1.5f, 1, 0, 4, 2, -.2f, .8f, 0,
        -6, 3, 1.2f, .5f, 0, 8, -1, -.8f, .4f, 0
    };
    pp_tui_frame wide = {0};
    pp_tui_frame repeat = {0};
    if (!pp_tui_compose(points, 4, 5, &detections, 6, 404, 12.16,
                        "cuDNN FP32", &state, 120, 40, &wide)) return 6;
    if (!validate_layout(&wide, 120, 40) ||
        !strstr(wide.data, "POINTPILLARS") ||
        !strstr(wide.data, "LIVE LIDAR") ||
        !strstr(wide.data, "3D LIDAR") ||
        strstr(wide.data, "SWEEP FLOW") ||
        strstr(wide.data, "SCENE OBJECTS") ||
        !strstr(wide.data, "\342\240")) {
        fprintf(stderr, "wide TUI fixture: layout=%d title=%d lidar=%d static=%d sidebar=%d braille=%d\n",
                validate_layout(&wide, 120, 40), !!strstr(wide.data, "POINTPILLARS"),
                !!strstr(wide.data, "LIVE LIDAR"), !strstr(wide.data, "SWEEP FLOW"),
                !!strstr(wide.data, "SCENE OBJECTS"), !!strstr(wide.data, "\342\240"));
        return 7;
    }
    if (!pp_tui_compose(points, 4, 5, &detections, 6, 404, 12.16,
                        "cuDNN FP32", &state, 120, 40, &repeat)) return 8;
    if (wide.length != repeat.length || memcmp(wide.data, repeat.data, wide.length))
        return 9;
    pp_tui_frame_free(&repeat);

    pp_tui_frame_free(&wide);

    float height_a[] = {8, 2, 1, 1, 0};
    float height_b[] = {8, 2, 2, 1, 0};
    pp_tui_frame depth_a = {0};
    pp_tui_frame depth_b = {0};
    if (!pp_tui_compose(height_a, 1, 5, NULL, 0, 1, 1.0, "CPU",
                        &state, 80, 24, &depth_a) ||
        !pp_tui_compose(height_b, 1, 5, NULL, 0, 1, 1.0, "CPU",
                        &state, 80, 24, &depth_b) ||
        (depth_a.length == depth_b.length &&
         !memcmp(depth_a.data, depth_b.data, depth_a.length))) return 22;
    pp_tui_frame_free(&depth_a);
    pp_tui_frame_free(&depth_b);
    state.perspective = 0;
    if (!pp_tui_compose(height_a, 1, 5, NULL, 0, 1, 1.0, "CPU",
                        &state, 80, 24, &depth_a) ||
        !pp_tui_compose(height_b, 1, 5, NULL, 0, 1, 1.0, "CPU",
                        &state, 80, 24, &depth_b) ||
        depth_a.length != depth_b.length ||
        memcmp(depth_a.data, depth_b.data, depth_a.length)) return 23;
    pp_tui_frame_free(&depth_a);
    pp_tui_frame_free(&depth_b);
    state.perspective = 1;

    state.show_grid = 0;
    float dense_points[16 * 5];
    for (size_t index = 0; index < 16; ++index)
        memcpy(dense_points + index * 5, height_b, 5 * sizeof(float));
    pp_tui_frame sparse_density = {0};
    pp_tui_frame high_density = {0};
    if (!pp_tui_compose(height_b, 1, 5, NULL, 0, 1, 1.0, "CPU",
                        &state, 80, 24, &sparse_density) ||
        !pp_tui_compose(dense_points, 16, 5, NULL, 0, 1, 1.0, "CPU",
                        &state, 80, 24, &high_density)) return 24;
    if (!strstr(sparse_density.data, "\033[2;38;5;118;48;5;233m") ||
        !strstr(high_density.data, "\033[1;38;5;118;48;5;233m")) return 25;
    pp_tui_frame_free(&sparse_density);
    pp_tui_frame_free(&high_density);
    state.show_grid = 1;

    state.show_sidebar = 1;
    pp_tui_frame inspector = {0};
    if (!pp_tui_compose(points, 4, 5, &detections, 6, 404, 12.16,
                        "cuDNN FP32", &state, 120, 40, &inspector) ||
        !validate_layout(&inspector, 120, 40) ||
        !strstr(inspector.data, "SCENE OBJECTS") ||
        !strstr(inspector.data, "FOCUS") || !strstr(inspector.data, "car")) return 19;
    pp_tui_frame_free(&inspector);
    state.show_sidebar = 0;

    pp_tui_frame compact = {0};
    if (!pp_tui_compose(points, 4, 5, &detections, 6, 404, 12.16,
                        "CPU", &state, 72, 20, &compact)) return 10;
    if (!validate_layout(&compact, 72, 20) ||
        strstr(compact.data, "SCENE OBJECTS") ||
        !strstr(compact.data, "Space pause")) return 11;
    pp_tui_frame_free(&compact);

    const int widths[] = {32, 40, 64, 79, 80, 90, 103, 104, 105, 120, 160};
    const int heights[] = {10, 20, 29, 30, 40};
    for (size_t wi = 0; wi < sizeof(widths) / sizeof(widths[0]); ++wi) {
        for (size_t hi = 0; hi < sizeof(heights) / sizeof(heights[0]); ++hi) {
            pp_tui_frame responsive = {0};
            if (!pp_tui_compose(points, 4, 5, &detections, 6, 404, 12.16,
                                "CUDA WMMA", &state, widths[wi], heights[hi],
                                &responsive)) return 12;
            if (!validate_layout(&responsive, widths[wi], heights[hi])) {
                fprintf(stderr, "responsive TUI fixture: %dx%d\n",
                        widths[wi], heights[hi]);
                return 13;
            }
            pp_tui_frame_free(&responsive);
        }
    }

    pp_tui_frame minimal = {0};
    if (!pp_tui_compose(NULL, 0, 0, NULL, 0, 0, 0.0, "CPU AVX2",
                        &state, 20, 5, &minimal) ||
        !validate_layout(&minimal, 20, 5) ||
        !strstr(minimal.data, "Terminal too small")) return 14;
    pp_tui_frame_free(&minimal);
    if (pp_tui_compose(points, 1, 2, &detections, 0, 1, 1.0, "CPU",
                       &state, 80, 24, &minimal)) return 15;

    detections.count = 0;
    for (size_t frame = 1; frame <= 4; ++frame)
        pp_tui_update_tracks(&state, &detections, frame);
    if (state.track_count) return 16;
    pp_tui_state_init(&state);
    memcpy(state.input, "wn", 2);
    state.input_length = 2;
    if (pp_tui_poll(&state, 0) != PP_TUI_REDRAW || state.center_x != 5.0f ||
        pp_tui_poll(&state, 0) != PP_TUI_NEXT || state.input_length) return 17;
    puts("tui terminal, input queue, compose, and tracks: ok");
    return 0;
}
