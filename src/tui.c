#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "pp_tui.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/signal.h>
#include <termios.h>
#include <unistd.h>

enum {
    PIX_GRID = 1,
    PIX_AXIS,
    PIX_POINT_OLD,
    PIX_POINT_MID,
    PIX_POINT_GROUND,
    PIX_POINT_LOW,
    PIX_POINT_HIGH,
    PIX_TRACK_BASE = 10,
    PIX_VELOCITY_BASE = 20,
    PIX_BOX_BASE = 30,
    PIX_SELECTED = 60,
    PIX_EGO
};

static const char *const class_name[10] = {
    "car", "truck", "construction", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic cone"
};
static const int class_color[10] = {51, 208, 220, 39, 141, 244, 201, 45, 82, 214};
static const char ansi_base[] = "\033[0;48;5;233m";

static struct termios saved_termios;
static int terminal_active;
static const char restore_sequence[] = "\033[0m\033[?25h\033[?1049l";
static volatile sig_atomic_t caught_signal;
static volatile sig_atomic_t resize_pending;

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
    int failed;
} text_buffer;

static int buffer_reserve(text_buffer *buffer, size_t extra) {
    if (buffer->failed) return 0;
    if (extra > SIZE_MAX - buffer->length - 1) {
        buffer->failed = 1;
        return 0;
    }
    size_t needed = buffer->length + extra + 1;
    if (needed <= buffer->capacity) return 1;
    size_t capacity = buffer->capacity ? buffer->capacity : 4096;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }
    char *grown = (char *)realloc(buffer->data, capacity);
    if (!grown) {
        buffer->failed = 1;
        return 0;
    }
    buffer->data = grown;
    buffer->capacity = capacity;
    return 1;
}

static void buffer_write(text_buffer *buffer, const char *text, size_t length) {
    if (!buffer_reserve(buffer, length)) return;
    memcpy(buffer->data + buffer->length, text, length);
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
}

static void buffer_puts(text_buffer *buffer, const char *text) {
    buffer_write(buffer, text, strlen(text));
}

static void buffer_putc(text_buffer *buffer, char value) {
    buffer_write(buffer, &value, 1);
}

static void buffer_printf(text_buffer *buffer, const char *format, ...) {
    if (buffer->failed) return;
    va_list args;
    va_start(args, format);
    va_list copy;
    va_copy(copy, args);
    int count = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (count < 0 || !buffer_reserve(buffer, (size_t)count)) {
        buffer->failed = 1;
        va_end(args);
        return;
    }
    vsnprintf(buffer->data + buffer->length,
              buffer->capacity - buffer->length, format, args);
    va_end(args);
    buffer->length += (size_t)count;
}

static void buffer_spaces(text_buffer *buffer, int count) {
    for (int i = 0; i < count; ++i) buffer_putc(buffer, ' ');
}

static void buffer_repeat(text_buffer *buffer, const char *glyph, int count) {
    for (int i = 0; i < count; ++i) buffer_puts(buffer, glyph);
}

static int text_cells(const char *text) {
    int cells = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p)
        if ((*p & 0xc0u) != 0x80u) ++cells;
    return cells;
}

static void buffer_text_cells(text_buffer *buffer, const char *text, int limit) {
    const unsigned char *begin = (const unsigned char *)text;
    const unsigned char *p = begin;
    int cells = 0;
    while (*p && cells < limit) {
        const unsigned char *next = p + 1;
        while ((*next & 0xc0u) == 0x80u) ++next;
        buffer_write(buffer, (const char *)p, (size_t)(next - p));
        p = next;
        ++cells;
    }
}

static void finish_line(text_buffer *buffer, int newline) {
    buffer_puts(buffer, ansi_base);
    buffer_puts(buffer, "\033[K");
    if (newline) buffer_putc(buffer, '\n');
}

static void restore_terminal(void) {
    if (!terminal_active) return;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved_termios);
    size_t remaining = sizeof(restore_sequence) - 1;
    const char *cursor = restore_sequence;
    while (remaining) {
        ssize_t written = write(STDOUT_FILENO, cursor, remaining);
        if (written > 0) {
            cursor += written;
            remaining -= (size_t)written;
        } else if (written < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
    terminal_active = 0;
}

static void interrupted(int signal_number) {
    caught_signal = signal_number;
}

static void resized(int signal_number) {
    (void)signal_number;
    resize_pending = 1;
}

static void reset_view(pp_tui_state *state) {
    state->center_x = 0.0f;
    state->center_y = 0.0f;
    state->yaw = 0.0f;
    state->zoom = 1.0f;
}

void pp_tui_state_init(pp_tui_state *state) {
    if (!state) return;
    memset(state, 0, sizeof(*state));
    reset_view(state);
    state->show_points = 1;
    state->show_boxes = 1;
    state->show_velocity = 1;
    state->show_grid = 1;
    state->show_tracks = 1;
    state->perspective = 1;
    state->class_mask = 0x3ffu;
    state->score_threshold = 0.2f;
    state->next_track_id = 1;
}

int pp_tui_begin(pp_tui_state *state) {
    if (!state || !isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return 0;
    pp_tui_state_init(state);
    if (tcgetattr(STDIN_FILENO, &saved_termios)) return 0;
    struct termios raw = saved_termios;
    raw.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    raw.c_iflag &= (tcflag_t)~(IXON | ICRNL);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)) return 0;
    terminal_active = 1;
    caught_signal = 0;
    resize_pending = 0;
    atexit(restore_terminal);
    signal(SIGINT, interrupted);
    signal(SIGTERM, interrupted);
    signal(SIGHUP, interrupted);
    signal(SIGQUIT, interrupted);
    signal(SIGTSTP, interrupted);
    signal(SIGWINCH, resized);
    fputs("\033[?1049h\033[?25l\033[2J", stdout);
    fflush(stdout);
    return 1;
}

void pp_tui_end(void) {
    restore_terminal();
}

static void consume_input(pp_tui_state *state, size_t count) {
    state->input_length -= count;
    memmove(state->input, state->input + count, state->input_length);
}

static int incomplete_escape(const pp_tui_state *state) {
    return state->input_length && state->input[0] == 27 &&
        (state->input_length == 1 ||
         (state->input_length == 2 && state->input[1] == '['));
}

int pp_tui_poll(pp_tui_state *state, int timeout_ms) {
    if (!state) return PP_TUI_QUIT;
    if (caught_signal) return PP_TUI_QUIT;
    if (resize_pending) {
        resize_pending = 0;
        return PP_TUI_REDRAW;
    }
    if (!state->input_length || incomplete_escape(state)) {
        struct pollfd descriptor = {STDIN_FILENO, POLLIN, 0};
        int result;
        do {
            result = poll(&descriptor, 1, timeout_ms);
        } while (result < 0 && errno == EINTR && !caught_signal && !resize_pending);
        if (caught_signal) return PP_TUI_QUIT;
        if (resize_pending) {
            resize_pending = 0;
            return PP_TUI_REDRAW;
        }
        if (result <= 0 || !(descriptor.revents & POLLIN)) {
            if (incomplete_escape(state)) state->input_length = 0;
            return PP_TUI_NONE;
        }
        size_t available = sizeof(state->input) - state->input_length;
        if (!available) consume_input(state, 1);
        available = sizeof(state->input) - state->input_length;
        ssize_t count = read(STDIN_FILENO, state->input + state->input_length,
                             available);
        if (count <= 0) return PP_TUI_NONE;
        state->input_length += (size_t)count;
    }
    if (incomplete_escape(state)) return PP_TUI_NONE;
    unsigned char value = state->input[0];
    unsigned char arrow = 0;
    if (value == 27 && state->input_length >= 3 && state->input[1] == '[') {
        arrow = state->input[2];
        consume_input(state, 3);
    } else {
        consume_input(state, 1);
    }
    if (value == 3 || value == 'q' || value == 'Q') return PP_TUI_QUIT;
    if (value == ' ') {
        state->paused = !state->paused;
        return PP_TUI_REDRAW;
    }
    if (value == 'n' || value == 'N' || arrow == 'C') return PP_TUI_NEXT;
    if (value == 'p' || value == 'P' || arrow == 'D') return PP_TUI_PREV;
    float pan = 5.0f / state->zoom;
    if (value == 'w' || arrow == 'A') state->center_x += pan;
    else if (value == 's' || arrow == 'B') state->center_x -= pan;
    else if (value == 'a') state->center_y += pan;
    else if (value == 'd') state->center_y -= pan;
    else if (value == 'e') state->yaw += 0.1308997f;
    else if (value == 'z') state->yaw -= 0.1308997f;
    else if (value == '+' || value == '=') {
        if (state->zoom < 8.0f) state->zoom *= 1.25f;
    } else if (value == '-' || value == '_') {
        if (state->zoom > 0.3f) state->zoom /= 1.25f;
    } else if (value == 'l' || value == 'L') state->show_points = !state->show_points;
    else if (value == 'b' || value == 'B') state->show_boxes = !state->show_boxes;
    else if (value == 'v' || value == 'V') state->show_velocity = !state->show_velocity;
    else if (value == 'g' || value == 'G') state->show_grid = !state->show_grid;
    else if (value == 't' || value == 'T') state->show_tracks = !state->show_tracks;
    else if (value == 'i' || value == 'I') state->show_sidebar = !state->show_sidebar;
    else if (value == 'm' || value == 'M') state->perspective = !state->perspective;
    else if (value == 'h' || value == 'H' || value == '?') {
        state->show_help = !state->show_help;
        if (state->show_help) state->show_sidebar = 1;
    }
    else if (value == 'c' || value == 'C') state->class_mask = 0x3ffu;
    else if (value >= '0' && value <= '9') state->class_mask ^= 1u << (value - '0');
    else if (value == '[') {
        if (state->selected > 0) --state->selected;
    } else if (value == ']') ++state->selected;
    else if (value == ',' || value == '<') {
        state->score_threshold = fmaxf(0.1f, state->score_threshold - 0.05f);
    } else if (value == '.' || value == '>') {
        state->score_threshold = fminf(0.95f, state->score_threshold + 0.05f);
    } else if (value == 'r' || value == 'R') reset_view(state);
    else return PP_TUI_NONE;
    return PP_TUI_REDRAW;
}

static void plot(unsigned char *pixels, int width, int height,
                 int x, int y, unsigned char value) {
    if ((unsigned)x >= (unsigned)width || (unsigned)y >= (unsigned)height) return;
    size_t index = (size_t)y * (size_t)width + (size_t)x;
    if (pixels[index] < value) pixels[index] = value;
}

static void plot_point(unsigned char *pixels, unsigned char *density,
                       int width, int height, int x, int y,
                       unsigned char value) {
    if ((unsigned)x >= (unsigned)width || (unsigned)y >= (unsigned)height) return;
    size_t index = (size_t)y * (size_t)width + (size_t)x;
    if (density[index] != UCHAR_MAX) ++density[index];
    if (pixels[index] < value) pixels[index] = value;
}

static int line_code(int x, int y, int width, int height) {
    return (x < 0 ? 1 : x >= width ? 2 : 0) |
           (y < 0 ? 4 : y >= height ? 8 : 0);
}

static int clip_line(int *x0, int *y0, int *x1, int *y1,
                     int width, int height) {
    int code0 = line_code(*x0, *y0, width, height);
    int code1 = line_code(*x1, *y1, width, height);
    for (;;) {
        if (!(code0 | code1)) return 1;
        if (code0 & code1) return 0;
        int code = code0 ? code0 : code1;
        double x = 0.0;
        double y = 0.0;
        if (code & 8) {
            y = height - 1;
            x = *x0 + (*x1 - *x0) * (y - *y0) / (double)(*y1 - *y0);
        } else if (code & 4) {
            y = 0.0;
            x = *x0 + (*x1 - *x0) * (y - *y0) / (double)(*y1 - *y0);
        } else if (code & 2) {
            x = width - 1;
            y = *y0 + (*y1 - *y0) * (x - *x0) / (double)(*x1 - *x0);
        } else {
            x = 0.0;
            y = *y0 + (*y1 - *y0) * (x - *x0) / (double)(*x1 - *x0);
        }
        if (code == code0) {
            *x0 = (int)lrint(x);
            *y0 = (int)lrint(y);
            code0 = line_code(*x0, *y0, width, height);
        } else {
            *x1 = (int)lrint(x);
            *y1 = (int)lrint(y);
            code1 = line_code(*x1, *y1, width, height);
        }
    }
}

static void draw_line(unsigned char *pixels, int width, int height,
                      int x0, int y0, int x1, int y1, unsigned char value) {
    if (!clip_line(&x0, &y0, &x1, &y1, width, height)) return;
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    for (;;) {
        plot(pixels, width, height, x0, y0, value);
        if (x0 == x1 && y0 == y1) break;
        int twice = 2 * error;
        if (twice >= dy) {
            error += dy;
            x0 += sx;
        }
        if (twice <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

static void encode_braille(unsigned codepoint, char text[4]) {
    text[0] = (char)(0xe0u | (codepoint >> 12));
    text[1] = (char)(0x80u | ((codepoint >> 6) & 63u));
    text[2] = (char)(0x80u | (codepoint & 63u));
    text[3] = '\0';
}

typedef struct {
    const pp_tui_state *state;
    float cosine, sine;
} projection;

static void project3(float x, float y, float z, const projection *view,
                     int width, int height, int *pixel_x, int *pixel_y) {
    const pp_tui_state *state = view->state;
    float dx = x - state->center_x;
    float dy = y - state->center_y;
    float forward = dx * view->cosine - dy * view->sine;
    float lateral = dx * view->sine + dy * view->cosine;
    if (state->perspective) {
        const float camera_distance = 72.0f;
        const float camera_height = 34.0f;
        float depth_axis = forward + camera_distance;
        float vertical_axis = z - camera_height;
        float depth = depth_axis * 0.904819f - vertical_axis * 0.425797f;
        float vertical = depth_axis * 0.425797f + vertical_axis * 0.904819f;
        if (depth <= 1.0f) {
            *pixel_x = -4 * width;
            *pixel_y = -4 * height;
            return;
        }
        float focal_x = (float)width * 0.70f;
        float focal_y = (float)height * 1.25f;
        float focal = (focal_x < focal_y ? focal_x : focal_y) * state->zoom;
        *pixel_x = (int)lrintf((float)(width - 1) * 0.5f + lateral * focal / depth);
        *pixel_y = (int)lrintf((float)(height - 1) * 0.48f - vertical * focal / depth);
        return;
    }
    float half_range = 51.2f / state->zoom;
    float sx = (float)(width - 1) / (2.0f * half_range);
    float sy = (float)(height - 1) / (2.0f * half_range);
    float scale = sx < sy ? sx : sy;
    *pixel_x = (int)lrintf((float)(width - 1) * 0.5f + lateral * scale);
    *pixel_y = (int)lrintf((float)(height - 1) * 0.5f - forward * scale);
}

static void project(float x, float y, const projection *view,
                    int width, int height, int *pixel_x, int *pixel_y) {
    project3(x, y, 0.0f, view, width, height, pixel_x, pixel_y);
}

void pp_tui_update_tracks(pp_tui_state *state,
                          const pp_detections *detections, size_t frame) {
    if (!state || !detections) return;
    if (!state->have_last_frame || frame != state->last_frame + 1) {
        state->track_count = 0;
        state->next_track_id = 1;
    }
    state->have_last_frame = 1;
    state->last_frame = frame;
    unsigned char used[PP_TUI_MAX_TRACKS] = {0};
    for (size_t box_index = 0; box_index < detections->count; ++box_index) {
        const pp_box *box = detections->boxes + box_index;
        int best = -1;
        float best_distance = 25.0f;
        for (size_t track_index = 0; track_index < state->track_count; ++track_index) {
            pp_tui_track *track = state->tracks + track_index;
            if (used[track_index] || track->class_id != box->class_id || !track->length)
                continue;
            float dx = track->x[track->length - 1] - box->x;
            float dy = track->y[track->length - 1] - box->y;
            float distance = dx * dx + dy * dy;
            if (distance < best_distance) {
                best_distance = distance;
                best = (int)track_index;
            }
        }
        if (best < 0) {
            if (state->track_count == PP_TUI_MAX_TRACKS) continue;
            best = (int)state->track_count++;
            memset(state->tracks + best, 0, sizeof(state->tracks[best]));
            state->tracks[best].class_id = (unsigned char)box->class_id;
            state->tracks[best].id = state->next_track_id++;
        }
        pp_tui_track *track = state->tracks + best;
        if (track->length == PP_TUI_TRAIL) {
            memmove(track->x, track->x + 1, (PP_TUI_TRAIL - 1) * sizeof(float));
            memmove(track->y, track->y + 1, (PP_TUI_TRAIL - 1) * sizeof(float));
            --track->length;
        }
        track->x[track->length] = box->x;
        track->y[track->length] = box->y;
        ++track->length;
        track->missed = 0;
        used[best] = 1;
    }
    for (size_t index = 0; index < state->track_count;) {
        if (!used[index] && ++state->tracks[index].missed > 3) {
            state->tracks[index] = state->tracks[--state->track_count];
            used[index] = used[state->track_count];
        } else {
            ++index;
        }
    }
}

static void detection_summary(const pp_detections *detections,
                              const pp_tui_state *state, size_t counts[10],
                              size_t *visible, int *selected) {
    memset(counts, 0, 10 * sizeof(*counts));
    *visible = 0;
    *selected = -1;
    if (!detections) return;
    for (size_t index = 0; index < detections->count; ++index) {
        const pp_box *box = detections->boxes + index;
        if (box->class_id < 0 || box->class_id >= 10 ||
            box->score < state->score_threshold) continue;
        ++counts[box->class_id];
        if (state->class_mask & (1u << box->class_id)) ++*visible;
    }
    if (!*visible) return;
    size_t target = (size_t)state->selected % *visible;
    size_t ordinal = 0;
    for (size_t index = 0; index < detections->count; ++index) {
        const pp_box *box = detections->boxes + index;
        if (box->class_id < 0 || box->class_id >= 10 ||
            box->score < state->score_threshold ||
            !(state->class_mask & (1u << box->class_id))) continue;
        if (ordinal++ == target) {
            *selected = (int)index;
            return;
        }
    }
}

static const pp_tui_track *track_for_box(const pp_tui_state *state,
                                         const pp_box *box) {
    const pp_tui_track *best = NULL;
    float best_distance = 0.25f;
    for (size_t index = 0; index < state->track_count; ++index) {
        const pp_tui_track *track = state->tracks + index;
        if (!track->length || track->class_id != box->class_id) continue;
        float dx = track->x[track->length - 1] - box->x;
        float dy = track->y[track->length - 1] - box->y;
        float distance = dx * dx + dy * dy;
        if (distance < best_distance) {
            best_distance = distance;
            best = track;
        }
    }
    return best;
}

static void draw_grid(unsigned char *pixels, int width, int height,
                      const projection *view) {
    const float pi = 3.14159265358979323846f;
    for (int radius = 10; radius <= 50; radius += 10) {
        int previous_x = 0;
        int previous_y = 0;
        for (int step = 0; step <= 128; ++step) {
            float angle = 2.0f * pi * (float)step / 128.0f;
            int x;
            int y;
            project((float)radius * cosf(angle), (float)radius * sinf(angle),
                    view, width, height, &x, &y);
            if (step) draw_line(pixels, width, height, previous_x, previous_y,
                                x, y, PIX_GRID);
            previous_x = x;
            previous_y = y;
        }
    }
    int x0;
    int y0;
    int x1;
    int y1;
    project(-55.0f, 0.0f, view, width, height, &x0, &y0);
    project(55.0f, 0.0f, view, width, height, &x1, &y1);
    draw_line(pixels, width, height, x0, y0, x1, y1, PIX_AXIS);
    project(0.0f, -70.0f, view, width, height, &x0, &y0);
    project(0.0f, 70.0f, view, width, height, &x1, &y1);
    draw_line(pixels, width, height, x0, y0, x1, y1, PIX_AXIS);
}

static void draw_ego(unsigned char *pixels, int width, int height,
                     const projection *view) {
    const float x[3] = {2.2f, -1.6f, -1.6f};
    const float y[3] = {0.0f, -1.1f, 1.1f};
    int px[3];
    int py[3];
    for (int index = 0; index < 3; ++index)
        project(x[index], y[index], view, width, height, px + index, py + index);
    for (int index = 0; index < 3; ++index)
        draw_line(pixels, width, height, px[index], py[index],
                  px[(index + 1) % 3], py[(index + 1) % 3], PIX_EGO);
}

static void draw_box(unsigned char *pixels, int width, int height,
                     const pp_box *box, const projection *view,
                     unsigned char ink) {
    const pp_tui_state *state = view->state;
    float cosine = cosf(box->yaw);
    float sine = sinf(box->yaw);
    float half_x = box->dx * 0.5f;
    float half_y = box->dy * 0.5f;
    int x[2][4];
    int y[2][4];
    for (int level = 0; level < 2; ++level) {
        float z = box->z + (level ? 0.5f : -0.5f) * box->dz;
        for (int corner = 0; corner < 4; ++corner) {
            float local_x = (corner == 0 || corner == 3) ? -half_x : half_x;
            float local_y = corner < 2 ? -half_y : half_y;
            project3(box->x + local_x * cosine - local_y * sine,
                     box->y + local_x * sine + local_y * cosine, z,
                     view, width, height, x[level] + corner, y[level] + corner);
        }
        for (int corner = 0; corner < 4; ++corner)
            draw_line(pixels, width, height, x[level][corner], y[level][corner],
                      x[level][(corner + 1) & 3], y[level][(corner + 1) & 3], ink);
    }
    if (state->perspective)
        for (int corner = 0; corner < 4; ++corner)
            draw_line(pixels, width, height, x[0][corner], y[0][corner],
                      x[1][corner], y[1][corner], ink);
    int center_x;
    int center_y;
    int nose_x;
    int nose_y;
    project3(box->x, box->y, box->z, view, width, height, &center_x, &center_y);
    project3(box->x + half_x * cosine, box->y + half_x * sine, box->z,
             view, width, height, &nose_x, &nose_y);
    draw_line(pixels, width, height, center_x, center_y, nose_x, nose_y, ink);
}

static unsigned char point_ink(const float *point, size_t stride) {
    float intensity = stride >= 4 ? fmaxf(0.0f, fminf(1.0f, point[3])) : 1.0f;
    float lag = stride >= 5 ? fmaxf(0.0f, fminf(1.0f, point[4])) : 0.0f;
    if (lag > 0.55f) return PIX_POINT_OLD;
    if (lag > 0.15f || intensity < 0.08f) return PIX_POINT_MID;
    return point[2] < -1.0f ? PIX_POINT_GROUND :
           point[2] < 0.75f ? PIX_POINT_LOW : PIX_POINT_HIGH;
}

static void draw_scene(unsigned char *pixels, unsigned char *density,
                       int width, int height,
                       const float *points, size_t count, size_t stride,
                       const pp_detections *detections, int selected,
                       const pp_tui_state *state) {
    projection view = {state, cosf(state->yaw), sinf(state->yaw)};
    if (state->show_grid) draw_grid(pixels, width, height, &view);
    if (state->show_points) {
        for (size_t index = 0; index < count; ++index) {
            const float *point = points + index * stride;
            unsigned char ink = point_ink(point, stride);
            int x;
            int y;
            project3(point[0], point[1], point[2], &view, width, height, &x, &y);
            plot_point(pixels, density, width, height, x, y, ink);
        }
    }
    if (state->show_tracks) {
        for (size_t index = 0; index < state->track_count; ++index) {
            const pp_tui_track *track = state->tracks + index;
            if (track->class_id >= 10 ||
                !(state->class_mask & (1u << track->class_id))) continue;
            for (unsigned char sample = 1; sample < track->length; ++sample) {
                int x0;
                int y0;
                int x1;
                int y1;
                project(track->x[sample - 1], track->y[sample - 1], &view,
                        width, height, &x0, &y0);
                project(track->x[sample], track->y[sample], &view,
                        width, height, &x1, &y1);
                draw_line(pixels, width, height, x0, y0, x1, y1,
                          (unsigned char)(PIX_TRACK_BASE + track->class_id));
            }
        }
    }
    if (detections) {
        for (size_t index = 0; index < detections->count; ++index) {
            const pp_box *box = detections->boxes + index;
            if (box->class_id < 0 || box->class_id >= 10 ||
                box->score < state->score_threshold ||
                !(state->class_mask & (1u << box->class_id))) continue;
            if (state->show_boxes) {
                unsigned char ink = index == (size_t)selected ? PIX_SELECTED :
                    (unsigned char)(PIX_BOX_BASE + box->class_id);
                draw_box(pixels, width, height, box, &view, ink);
            }
            if (state->show_velocity && box->vx * box->vx + box->vy * box->vy > 0.01f) {
                int x0;
                int y0;
                int x1;
                int y1;
                project3(box->x, box->y, box->z, &view, width, height, &x0, &y0);
                project3(box->x + box->vx, box->y + box->vy, box->z, &view,
                         width, height, &x1, &y1);
                unsigned char ink = index == (size_t)selected ? PIX_SELECTED :
                    (unsigned char)(PIX_VELOCITY_BASE + box->class_id);
                draw_line(pixels, width, height, x0, y0, x1, y1, ink);
                plot(pixels, width, height, x1, y1, ink);
            }
        }
    }
    draw_ego(pixels, width, height, &view);
}

static void set_pixel_style(text_buffer *buffer, int kind, int density_grade) {
    int color = 250;
    int mode = 0;
    if (kind == 0) {
        buffer_puts(buffer, ansi_base);
        return;
    }
    if (kind == PIX_GRID) {
        color = 239;
        mode = 2;
    } else if (kind == PIX_AXIS) {
        color = 243;
        mode = 2;
    } else if (kind == PIX_POINT_OLD) {
        color = 24;
        mode = 2;
    } else if (kind == PIX_POINT_MID) {
        color = 31;
        mode = 2;
    } else if (kind == PIX_POINT_GROUND) {
        color = 246;
        mode = 2;
    } else if (kind == PIX_POINT_LOW) {
        color = 45;
        mode = 2;
    } else if (kind == PIX_POINT_HIGH) {
        color = 118;
    } else if (kind >= PIX_TRACK_BASE && kind < PIX_VELOCITY_BASE) {
        color = class_color[kind - PIX_TRACK_BASE];
        mode = 2;
    } else if (kind >= PIX_VELOCITY_BASE && kind < PIX_BOX_BASE) {
        color = class_color[kind - PIX_VELOCITY_BASE];
    } else if (kind >= PIX_BOX_BASE && kind < PIX_BOX_BASE + 10) {
        color = class_color[kind - PIX_BOX_BASE];
        mode = 1;
    } else if (kind == PIX_SELECTED) {
        color = 15;
        mode = 1;
    } else if (kind == PIX_EGO) {
        color = 214;
        mode = 1;
    }
    if (kind >= PIX_POINT_OLD && kind <= PIX_POINT_HIGH) {
        mode = density_grade <= 1 ? 2 : density_grade == 2 ? 0 : 1;
    }
    buffer_printf(buffer, "\033[%d;38;5;%d;48;5;233m", mode, color);
}

static void render_braille_row(text_buffer *buffer, const unsigned char *pixels,
                                const unsigned char *density,
                                int pixel_width, int character_row,
                                int character_width) {
    static const int dot[4][2] = {{1, 8}, {2, 16}, {4, 32}, {64, 128}};
    int previous_style = -1;
    for (int column = 0; column < character_width; ++column) {
        int mask = 0;
        int kind = 0;
        unsigned density_total = 0;
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 2; ++x) {
                size_t index =
                    (size_t)(character_row * 4 + y) * (size_t)pixel_width +
                    (size_t)(column * 2 + x);
                unsigned char value = pixels[index];
                if (value) mask |= dot[y][x];
                if (value > kind) kind = value;
                density_total += density[index];
            }
        }
        int grade = density_total <= 2 ? 1 : density_total <= 8 ? 2 : 3;
        if (kind < PIX_POINT_OLD || kind > PIX_POINT_HIGH) grade = 0;
        int style = kind * 4 + grade;
        if (style != previous_style) {
            set_pixel_style(buffer, kind, grade);
            previous_style = style;
        }
        if (mask) {
            char encoded[4];
            encode_braille(0x2800u + (unsigned)mask, encoded);
            buffer_puts(buffer, encoded);
        } else {
            buffer_putc(buffer, ' ');
        }
    }
}

static void append_plain_line(text_buffer *buffer, const char *style,
                              const char *text, int width, int newline) {
    int cells = text_cells(text);
    buffer_puts(buffer, style);
    buffer_text_cells(buffer, text, width);
    if (cells < width) buffer_spaces(buffer, width - cells);
    finish_line(buffer, newline);
}

static void append_header(text_buffer *buffer, int columns, size_t frame,
                          size_t frame_count, double inference_ms,
                          const char *backend, const pp_tui_state *state,
                          size_t points, size_t visible, size_t decoded) {
    buffer_puts(buffer, "\033[H");
    if (columns < 80) {
        char line[256];
        snprintf(line, sizeof(line), " POINTPILLARS  [%s]  %s  %.2f ms",
                 backend, state->paused ? "PAUSED" : "PLAY", inference_ms);
        append_plain_line(buffer, "\033[1;38;5;51;48;5;233m", line, columns, 1);
        snprintf(line, sizeof(line), " FRAME %zu/%zu  %zu points  %zu/%zu objects",
                 frame + 1, frame_count, points, visible, decoded);
        append_plain_line(buffer, "\033[38;5;250;48;5;233m", line, columns, 1);
        return;
    }
    char backend_text[64];
    char state_text[32];
    char metric_text[64];
    snprintf(backend_text, sizeof(backend_text), "  [ %s ]", backend);
    snprintf(state_text, sizeof(state_text), "  ● %s", state->paused ? "PAUSED" : "PLAY");
    snprintf(metric_text, sizeof(metric_text), "INFER %7.2f ms ", inference_ms);
    int left = text_cells(" POINTPILLARS") + text_cells("  /  LIVE LIDAR ") +
               text_cells(backend_text) + text_cells(state_text);
    int right = text_cells(metric_text);
    buffer_puts(buffer, "\033[1;38;5;51;48;5;233m POINTPILLARS");
    buffer_puts(buffer, "\033[2;38;5;248;48;5;233m  /  LIVE LIDAR ");
    buffer_puts(buffer, "\033[38;5;255;48;5;233m");
    buffer_puts(buffer, backend_text);
    buffer_printf(buffer, "\033[1;38;5;%d;48;5;233m%s",
                  state->paused ? 214 : 82, state_text);
    buffer_spaces(buffer, columns > left + right ? columns - left - right : 1);
    buffer_puts(buffer, "\033[38;5;250;48;5;233m");
    buffer_puts(buffer, metric_text);
    finish_line(buffer, 1);

    char frame_text[64];
    char object_text[96];
    snprintf(frame_text, sizeof(frame_text), " FRAME %03zu / %03zu  ",
             frame + 1, frame_count);
    snprintf(object_text, sizeof(object_text), "  %zu POINTS   %zu / %zu OBJECTS ",
             points, visible, decoded);
    int available = columns - text_cells(frame_text) - text_cells(object_text);
    int bar_width = available > 36 ? 36 : available;
    if (bar_width < 4) bar_width = 4;
    double ratio = frame_count ? (double)(frame + 1) / (double)frame_count : 0.0;
    int filled = (int)lrint(ratio * bar_width);
    if (filled < 1) filled = 1;
    if (filled > bar_width) filled = bar_width;
    buffer_puts(buffer, "\033[38;5;250;48;5;233m");
    buffer_puts(buffer, frame_text);
    buffer_puts(buffer, "\033[1;38;5;51;48;5;233m");
    buffer_repeat(buffer, "━", filled);
    buffer_puts(buffer, "\033[2;38;5;239;48;5;233m");
    buffer_repeat(buffer, "━", bar_width - filled);
    buffer_puts(buffer, "\033[38;5;248;48;5;233m");
    buffer_puts(buffer, object_text);
    finish_line(buffer, 1);
}

static void side_line(text_buffer *buffer, int width, const char *style,
                      const char *text) {
    int cells = text_cells(text);
    buffer_puts(buffer, style);
    buffer_text_cells(buffer, text, width);
    if (cells < width) buffer_spaces(buffer, width - cells);
}

static void append_side(text_buffer *buffer, int row, int width,
                        const size_t counts[10], size_t visible, int selected,
                        const pp_detections *detections,
                        const pp_tui_state *state) {
    char text[128] = "";
    char dynamic_style[48];
    const char *style = "\033[38;5;248;48;5;233m";
    if (state->show_help) {
        static const char *const help[] = {
            "  CONTROLS", "", "  Space      play / pause",
            "  ← / →      previous / next", "  W A S D    move view",
            "  z / e      rotate", "  + / -      zoom",
            "  [ / ]      select object", "  , / .      score gate",
            "  0–9 / c    classes / all", "  l b v g t  scene layers",
            "  i          toggle inspector", "  m          3D / BEV mode",
            "  r          reset camera",
            "  h / ?      close help",
            "  q          quit", "", "  Changes redraw instantly;",
            "  paused frames never rerun", "  model inference."
        };
        int help_count = (int)(sizeof(help) / sizeof(help[0]));
        if (row >= 0 && row < help_count) {
            style = row == 0 ? "\033[1;38;5;51;48;5;233m" :
                    row >= 2 && row <= 13 ? "\033[38;5;252;48;5;233m" : style;
            snprintf(text, sizeof(text), "%s", help[row]);
        }
        side_line(buffer, width, style, text);
        return;
    }
    if (row == 0) {
        style = "\033[1;38;5;51;48;5;233m";
        snprintf(text, sizeof(text), "  SCENE OBJECTS");
    } else if (row == 1) {
        snprintf(text, sizeof(text), "  score ≥ %.2f  ·  %zu visible",
                 state->score_threshold, visible);
    } else if (row >= 3 && row < 13) {
        int class_id = row - 3;
        int enabled = !!(state->class_mask & (1u << class_id));
        snprintf(text, sizeof(text), "  ●  %d %-13s %4zu",
                 class_id, class_name[class_id], counts[class_id]);
        snprintf(dynamic_style, sizeof(dynamic_style), "\033[%d;38;5;%d;48;5;233m",
                 enabled ? 0 : 2, enabled ? class_color[class_id] : 240);
        style = dynamic_style;
    } else if (row == 14 || row == 24) {
        style = "\033[2;38;5;239;48;5;233m";
        snprintf(text, sizeof(text), "  ──────────────────────────");
    } else if (row == 15) {
        style = "\033[1;38;5;51;48;5;233m";
        if (selected >= 0)
            snprintf(text, sizeof(text), "  FOCUS  %02d / %02zu",
                     state->selected % (int)visible + 1, visible);
        else snprintf(text, sizeof(text), "  FOCUS");
    } else if (selected >= 0 && detections) {
        const pp_box *box = detections->boxes + selected;
        const pp_tui_track *track = track_for_box(state, box);
        if (row == 16) {
            snprintf(dynamic_style, sizeof(dynamic_style),
                     "\033[1;38;5;%d;48;5;233m", class_color[box->class_id]);
            style = dynamic_style;
            snprintf(text, sizeof(text), "  %s  %.1f%%",
                     class_name[box->class_id], box->score * 100.0f);
        } else if (row == 17) {
            snprintf(text, sizeof(text), "  track  #%02d  ·  history %02d",
                     track ? track->id : 0, track ? track->length : 0);
        } else if (row == 19) {
            snprintf(text, sizeof(text), "  position  %+.1f  %+.1f m", box->x, box->y);
        } else if (row == 20) {
            snprintf(text, sizeof(text), "  elevation %+.1f m", box->z);
        } else if (row == 21) {
            snprintf(text, sizeof(text), "  size      %.1f × %.1f × %.1f", box->dx, box->dy, box->dz);
        } else if (row == 22) {
            snprintf(text, sizeof(text), "  heading   %+.0f°", box->yaw * 57.2957795f);
        } else if (row == 23) {
            snprintf(text, sizeof(text), "  velocity  %+.1f  %+.1f m/s", box->vx, box->vy);
        }
    } else if (row == 16) {
        snprintf(text, sizeof(text), "  No object passes the filter");
    }
    if (row == 25) {
        style = "\033[1;38;5;51;48;5;233m";
        snprintf(text, sizeof(text), "  VIEW");
    } else if (row == 26) {
        snprintf(text, sizeof(text), "  center  %+.1f  %+.1f m",
                 state->center_x, state->center_y);
    } else if (row == 27) {
        snprintf(text, sizeof(text), "  zoom    %.2fx  ·  yaw %+.0f°",
                 state->zoom, state->yaw * 57.2957795f);
    } else if (row == 28) {
        snprintf(text, sizeof(text), "  mode    %s · 10 m rings",
                 state->perspective ? "perspective 3D" : "top-down BEV");
    }
    side_line(buffer, width, style, text);
}

static void append_canvas_border(text_buffer *buffer, int width, int top,
                                 const pp_tui_state *state) {
    const char *title = state->perspective ?
        " 3D LIDAR · HEIGHT + AGE " : " BEV · 10 m RANGE RINGS ";
    int title_width = text_cells(title);
    buffer_puts(buffer, "\033[2;38;5;240;48;5;233m");
    buffer_puts(buffer, top ? "╭" : "╰");
    if (top && width >= title_width + 1) {
        buffer_puts(buffer, "─");
        buffer_puts(buffer, "\033[38;5;246;48;5;233m");
        buffer_puts(buffer, title);
        buffer_puts(buffer, "\033[2;38;5;240;48;5;233m");
        buffer_repeat(buffer, "─", width - title_width - 1);
    } else {
        buffer_repeat(buffer, "─", width);
    }
    buffer_puts(buffer, top ? "╮" : "╯");
}

static void append_footer(text_buffer *buffer, int columns,
                          const pp_tui_state *state) {
    char view[512];
    snprintf(view, sizeof(view),
             " VIEW  %s  ·  center %+.1f %+.1f m  ·  zoom %.2fx  ·  yaw %+.0f°  ·  score ≥ %.2f  ·  layers %c %c %c %c %c",
             state->perspective ? "3D" : "BEV",
             state->center_x, state->center_y, state->zoom,
             state->yaw * 57.2957795f, state->score_threshold,
             state->show_points ? 'P' : '-', state->show_boxes ? 'B' : '-',
             state->show_velocity ? 'V' : '-', state->show_grid ? 'G' : '-',
             state->show_tracks ? 'T' : '-');
    append_plain_line(buffer, "\033[38;5;246;48;5;233m", view, columns, 1);
    const char *controls = state->show_help ?
        " h / ? close help   ·   Space play / pause   ·   ← / → frames   ·   q quit" :
        " Space pause  ·  ← / → frames  ·  WASD move  ·  m 3D/BEV  ·  i inspector  ·  h help  ·  q quit";
    append_plain_line(buffer, "\033[38;5;252;48;5;233m", controls, columns, 0);
}

static int compose_minimal(text_buffer *buffer, const char *backend,
                           double inference_ms, int columns, int rows) {
    for (int row = 0; row < rows; ++row) {
        char text[160] = "";
        const char *style = "\033[38;5;248;48;5;233m";
        if (row == 0) {
            snprintf(text, sizeof(text), " POINTPILLARS  [%s]", backend);
            style = "\033[1;38;5;51;48;5;233m";
        } else if (row == 2) {
            snprintf(text, sizeof(text), " Terminal too small for BEV");
        } else if (row == 3) {
            snprintf(text, sizeof(text), " Need at least 32 × 10 · %.2f ms", inference_ms);
        } else if (row == rows - 1) {
            snprintf(text, sizeof(text), " q quit");
        }
        if (row == 0) buffer_puts(buffer, "\033[H");
        append_plain_line(buffer, style, text, columns, row + 1 < rows);
    }
    return !buffer->failed;
}

int pp_tui_compose(const float *points, size_t count, size_t stride,
                   const pp_detections *detections, size_t frame,
                   size_t frame_count, double inference_ms,
                   const char *backend, const pp_tui_state *state,
                   int columns, int rows, pp_tui_frame *output) {
    if (!state || !backend || !output || columns < 1 || rows < 1 ||
        (count && (!points || stride < 3))) return 0;
    output->data = NULL;
    output->length = 0;
    text_buffer buffer = {0};
    if (columns < 32 || rows < 10) {
        if (!compose_minimal(&buffer, backend, inference_ms, columns, rows)) {
            free(buffer.data);
            return 0;
        }
        output->data = buffer.data;
        output->length = buffer.length;
        return 1;
    }

    size_t counts[10];
    size_t visible;
    int selected;
    detection_summary(detections, state, counts, &visible, &selected);
    append_header(&buffer, columns, frame, frame_count, inference_ms, backend,
                  state, count, visible, detections ? detections->count : 0);

    int use_sidebar = state->show_sidebar && columns >= 104 && rows >= 30;
    int side_width = use_sidebar ? 30 : 0;
    int canvas_total = columns - (use_sidebar ? side_width + 1 : 0);
    int canvas_width = canvas_total - 2;
    int canvas_height = rows - 6;
    int pixel_width = canvas_width * 2;
    int pixel_height = canvas_height * 4;
    size_t pixel_count = (size_t)pixel_width * (size_t)pixel_height;
    unsigned char *pixels = (unsigned char *)calloc(pixel_count, 1);
    unsigned char *density = (unsigned char *)calloc(pixel_count, 1);
    if (!pixels || !density) {
        free(density);
        free(pixels);
        free(buffer.data);
        return 0;
    }
    draw_scene(pixels, density, pixel_width, pixel_height, points, count, stride,
               detections, selected, state);

    append_canvas_border(&buffer, canvas_width, 1, state);
    if (use_sidebar) {
        buffer_putc(&buffer, ' ');
        append_side(&buffer, -1, side_width, counts, visible, selected,
                    detections, state);
    }
    finish_line(&buffer, 1);
    for (int row = 0; row < canvas_height; ++row) {
        buffer_puts(&buffer, "\033[2;38;5;240;48;5;233m│");
        render_braille_row(&buffer, pixels, density, pixel_width, row,
                           canvas_width);
        buffer_puts(&buffer, "\033[2;38;5;240;48;5;233m│");
        if (use_sidebar) {
            buffer_putc(&buffer, ' ');
            append_side(&buffer, row, side_width, counts, visible, selected,
                        detections, state);
        }
        finish_line(&buffer, 1);
    }
    append_canvas_border(&buffer, canvas_width, 0, state);
    if (use_sidebar) {
        buffer_putc(&buffer, ' ');
        append_side(&buffer, canvas_height, side_width, counts, visible,
                    selected, detections, state);
    }
    finish_line(&buffer, 1);
    free(density);
    free(pixels);
    append_footer(&buffer, columns, state);
    if (buffer.failed) {
        free(buffer.data);
        return 0;
    }
    output->data = buffer.data;
    output->length = buffer.length;
    return 1;
}

void pp_tui_frame_free(pp_tui_frame *frame) {
    if (!frame) return;
    free(frame->data);
    frame->data = NULL;
    frame->length = 0;
}

static void write_frame(const char *data, size_t length) {
    while (length) {
        ssize_t written = write(STDOUT_FILENO, data, length);
        if (written > 0) {
            data += written;
            length -= (size_t)written;
        } else if (written < 0 && errno == EINTR) {
            continue;
        } else {
            return;
        }
    }
}

void pp_tui_render(const float *points, size_t count, size_t stride,
                   const pp_detections *detections, size_t frame,
                   size_t frame_count, double inference_ms,
                   const char *backend, const pp_tui_state *state) {
    struct winsize size = {0};
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);
    int columns = size.ws_col ? size.ws_col : 100;
    int rows = size.ws_row ? size.ws_row : 36;
    pp_tui_frame output = {0};
    if (!pp_tui_compose(points, count, stride, detections, frame, frame_count,
                        inference_ms, backend, state, columns, rows, &output))
        return;
    write_frame(output.data, output.length);
    pp_tui_frame_free(&output);
}
