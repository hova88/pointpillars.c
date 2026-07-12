#define _POSIX_C_SOURCE 200809L
#include "pp_tui.h"
#include <errno.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

static struct termios saved_termios;
static int terminal_active;
static const char restore_sequence[] = "\033[0m\033[?25h\033[?1049l";
static volatile sig_atomic_t caught_signal;
static volatile sig_atomic_t resize_pending;

static void restore_terminal(void) {
    if (!terminal_active) return;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved_termios);
    ssize_t written = write(STDOUT_FILENO, restore_sequence, sizeof(restore_sequence) - 1);
    (void)written;
    terminal_active = 0;
}

static void interrupted(int sig) {
    caught_signal = sig;
}
static void resized(int sig) { (void)sig; resize_pending = 1; }

static void reset_view(pp_tui_state *s) {
    s->center_x = s->center_y = s->yaw = 0.0f;
    s->zoom = 1.0f;
}

int pp_tui_begin(pp_tui_state *s) {
    if (!s || !isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return 0;
    memset(s, 0, sizeof(*s)); reset_view(s); s->show_boxes = 1;
    s->show_velocity = s->show_grid = s->show_tracks = 1;
    s->class_mask = 0x3ffu; s->score_threshold = .1f;
    if (tcgetattr(STDIN_FILENO, &saved_termios)) return 0;
    struct termios raw = saved_termios;
    raw.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    raw.c_iflag &= (tcflag_t)~(IXON | ICRNL);
    raw.c_cc[VMIN] = 0; raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw)) return 0;
    terminal_active = 1; caught_signal = resize_pending = 0; atexit(restore_terminal);
    signal(SIGINT, interrupted); signal(SIGTERM, interrupted); signal(SIGHUP, interrupted);
    signal(SIGQUIT, interrupted); signal(SIGTSTP, interrupted); signal(SIGWINCH, resized);
    fputs("\033[?1049h\033[?25l\033[2J", stdout); fflush(stdout);
    return 1;
}

void pp_tui_end(void) { restore_terminal(); }

int pp_tui_poll(pp_tui_state *s, int timeout_ms) {
    if (caught_signal) return PP_TUI_QUIT;
    if (resize_pending) { resize_pending = 0; return PP_TUI_REDRAW; }
    struct pollfd p = {STDIN_FILENO, POLLIN, 0};
    int rc; do rc = poll(&p, 1, timeout_ms); while (rc < 0 && errno == EINTR && !caught_signal && !resize_pending);
    if (caught_signal) return PP_TUI_QUIT;
    if (resize_pending) { resize_pending = 0; return PP_TUI_REDRAW; }
    if (rc <= 0 || !(p.revents & POLLIN)) return PP_TUI_NONE;
    unsigned char key[8]; ssize_t n = read(STDIN_FILENO, key, sizeof(key));
    if (n <= 0) return PP_TUI_NONE;
    unsigned char k = key[0];
    if (k == 3 || k == 'q' || k == 'Q') return PP_TUI_QUIT;
    if (k == ' ') { s->paused = !s->paused; return PP_TUI_REDRAW; }
    if (k == 'n' || k == 'N' || (n >= 3 && k == 27 && key[2] == 'C')) return PP_TUI_NEXT;
    if (k == 'p' || k == 'P' || (n >= 3 && k == 27 && key[2] == 'D')) return PP_TUI_PREV;
    float pan = 5.0f / s->zoom;
    if (k == 'w' || (n >= 3 && k == 27 && key[2] == 'A')) s->center_x += pan;
    else if (k == 's' || (n >= 3 && k == 27 && key[2] == 'B')) s->center_x -= pan;
    else if (k == 'a') s->center_y += pan;
    else if (k == 'd') s->center_y -= pan;
    else if (k == 'e') s->yaw += 0.1308997f;
    else if (k == 'z') s->yaw -= 0.1308997f;
    else if (k == '+' || k == '=') { if (s->zoom < 8.0f) s->zoom *= 1.25f; }
    else if (k == '-' || k == '_') { if (s->zoom > 0.3f) s->zoom /= 1.25f; }
    else if (k == 'b' || k == 'B') s->show_boxes = !s->show_boxes;
    else if (k == 'v' || k == 'V') s->show_velocity = !s->show_velocity;
    else if (k == 'g' || k == 'G') s->show_grid = !s->show_grid;
    else if (k == 't' || k == 'T') s->show_tracks = !s->show_tracks;
    else if (k == 'c' || k == 'C') s->class_mask = 0x3ffu;
    else if (k >= '0' && k <= '9') s->class_mask ^= 1u << (k - '0');
    else if (k == '[') { if (s->selected > 0) --s->selected; }
    else if (k == ']') ++s->selected;
    else if (k == ',' || k == '<') { if (s->score_threshold > .05f) s->score_threshold -= .05f; }
    else if (k == '.' || k == '>') { if (s->score_threshold < .95f) s->score_threshold += .05f; }
    else if (k == 'r' || k == 'R') reset_view(s);
    else return PP_TUI_NONE;
    return PP_TUI_REDRAW;
}

static void plot(unsigned char *p,int w,int h,int x,int y,unsigned char v){if((unsigned)x<(unsigned)w&&(unsigned)y<(unsigned)h&&p[(size_t)y*w+x]<v)p[(size_t)y*w+x]=v;}
static void line(unsigned char*p,int w,int h,int x0,int y0,int x1,int y1,unsigned char v){int dx=abs(x1-x0),sx=x0<x1?1:-1,dy=-abs(y1-y0),sy=y0<y1?1:-1,e=dx+dy;for(;;){plot(p,w,h,x0,y0,v);if(x0==x1&&y0==y1)break;int q=2*e;if(q>=dy){e+=dy;x0+=sx;}if(q<=dx){e+=dx;y0+=sy;}}}
static void utf8(unsigned cp,char s[4]){s[0]=(char)(0xe0|(cp>>12));s[1]=(char)(0x80|((cp>>6)&63));s[2]=(char)(0x80|(cp&63));s[3]=0;}
static void project(float x,float y,const pp_tui_state*s,int pw,int ph,int*px,int*py){float dx=x-s->center_x,dy=y-s->center_y,c=cosf(s->yaw),q=sinf(s->yaw),fx=dx*c-dy*q,fy=dx*q+dy*c,range=51.2f/s->zoom;*px=(int)((fy/range*.5f+.5f)*(pw-1));*py=(int)((.5f-fx/range*.5f)*(ph-1));}

void pp_tui_update_tracks(pp_tui_state*s,const pp_detections*d,size_t frame){if(!s||!d)return;if(!s->have_last_frame||frame!=s->last_frame+1){s->track_count=0;s->next_track_id=1;}s->have_last_frame=1;s->last_frame=frame;unsigned char used[PP_TUI_MAX_TRACKS]={0};for(size_t j=0;j<d->count;j++){const pp_box*b=&d->boxes[j];int best=-1;float best_d2=25;for(size_t i=0;i<s->track_count;i++){pp_tui_track*t=&s->tracks[i];if(used[i]||t->class_id!=b->class_id||!t->length)continue;float dx=t->x[t->length-1]-b->x,dy=t->y[t->length-1]-b->y,d2=dx*dx+dy*dy;if(d2<best_d2){best_d2=d2;best=(int)i;}}if(best<0){if(s->track_count==PP_TUI_MAX_TRACKS)continue;best=(int)s->track_count++;memset(&s->tracks[best],0,sizeof(s->tracks[best]));s->tracks[best].class_id=(unsigned char)b->class_id;s->tracks[best].id=s->next_track_id++;}pp_tui_track*t=&s->tracks[best];if(t->length==PP_TUI_TRAIL){memmove(t->x,t->x+1,(PP_TUI_TRAIL-1)*sizeof(float));memmove(t->y,t->y+1,(PP_TUI_TRAIL-1)*sizeof(float));t->length--;}t->x[t->length]=b->x;t->y[t->length]=b->y;t->length++;t->missed=0;used[best]=1;}for(size_t i=0;i<s->track_count;){if(!used[i]&&++s->tracks[i].missed>3){s->tracks[i]=s->tracks[--s->track_count];used[i]=used[s->track_count];}else i++;}}

void pp_tui_render(const float*pts,size_t n,size_t stride,const pp_detections*d,size_t frame,size_t frames,double infer,const char*backend,const pp_tui_state*s){
 static const char*class_name[10]={"car","truck","construction","bus","trailer","barrier","motorcycle","bicycle","pedestrian","cone"};
 static const int class_color[10]={51,208,220,39,141,244,201,45,82,214};
 struct winsize z={0};ioctl(STDOUT_FILENO,TIOCGWINSZ,&z);int cols=z.ws_col?z.ws_col:100,rows=z.ws_row?z.ws_row:36;if(cols<32)cols=32;if(rows<11)rows=11;int ch=rows-5,pw=cols*2,ph=ch*4;
 unsigned char*pix=(unsigned char*)calloc((size_t)pw*ph,1);if(!pix)return;
 if(s->show_grid)for(int m=-50;m<=50;m+=10){int x0,y0,x1,y1;project((float)m,-50,s,pw,ph,&x0,&y0);project((float)m,50,s,pw,ph,&x1,&y1);line(pix,pw,ph,x0,y0,x1,y1,1);project(-50,(float)m,s,pw,ph,&x0,&y0);project(50,(float)m,s,pw,ph,&x1,&y1);line(pix,pw,ph,x0,y0,x1,y1,1);}
 for(size_t i=0;i<n;i++){const float*q=pts+i*stride;int px,py;project(q[0],q[1],s,pw,ph,&px,&py);plot(pix,pw,ph,px,py,q[2]>.5f?3:2);}
 if(s->show_tracks)for(size_t i=0;i<s->track_count;i++){const pp_tui_track*t=&s->tracks[i];if(!(s->class_mask&(1u<<t->class_id)))continue;for(int j=1;j<t->length;j++){int x0,y0,x1,y1;project(t->x[j-1],t->y[j-1],s,pw,ph,&x0,&y0);project(t->x[j],t->y[j],s,pw,ph,&x1,&y1);line(pix,pw,ph,x0,y0,x1,y1,(unsigned char)(10+t->class_id));}}
 size_t visible=0;if(d)for(size_t j=0;j<d->count;j++)if(d->boxes[j].score>=s->score_threshold&&(s->class_mask&(1u<<d->boxes[j].class_id)))visible++;int selected=-1;if(visible){size_t target=(size_t)s->selected%visible,ordinal=0;for(size_t j=0;j<d->count;j++)if(d->boxes[j].score>=s->score_threshold&&(s->class_mask&(1u<<d->boxes[j].class_id))){if(ordinal++==target){selected=(int)j;break;}}}
 if(s->show_boxes&&d)for(size_t j=0;j<d->count;j++){const pp_box*b=d->boxes+j;if(b->score<s->score_threshold||!(s->class_mask&(1u<<b->class_id)))continue;float c=cosf(b->yaw),q=sinf(b->yaw),hx=b->dx*.5f,hy=b->dy*.5f;int x[4],y[4];unsigned char ink=j==(size_t)selected?30:(unsigned char)(10+b->class_id);for(int k=0;k<4;k++){float lx=(k==0||k==3)?-hx:hx,ly=k<2?-hy:hy;project(b->x+lx*c-ly*q,b->y+lx*q+ly*c,s,pw,ph,&x[k],&y[k]);}for(int k=0;k<4;k++)line(pix,pw,ph,x[k],y[k],x[(k+1)&3],y[(k+1)&3],ink);if(s->show_velocity&&(b->vx*b->vx+b->vy*b->vy)>.01f){int x0,y0,x1,y1;project(b->x,b->y,s,pw,ph,&x0,&y0);project(b->x+b->vx,b->y+b->vy,s,pw,ph,&x1,&y1);line(pix,pw,ph,x0,y0,x1,y1,ink);}}
 int ex,ey;project(0,0,s,pw,ph,&ex,&ey);for(int k=-2;k<=2;k++){plot(pix,pw,ph,ex+k,ey,30);plot(pix,pw,ph,ex,ey+k,30);}
 printf("\033[H\033[2K\033[1;38;5;51m PointPillars\033[0m  %s  frame %zu/%zu  %7.2f ms  %zu pts  %zu/%zu boxes  %s\n",backend,frame+1,frames,infer,n,visible,d?d->count:0,s->paused?"\033[33mPAUSED\033[0m":"\033[32mPLAY\033[0m");
 static const int bit[4][2]={{1,8},{2,16},{4,32},{64,128}};
 for(int cy=0;cy<ch;cy++){fputs("\033[2K",stdout);for(int cx=0;cx<cols;cx++){int mask=0,kind=0;for(int yy=0;yy<4;yy++)for(int xx=0;xx<2;xx++){unsigned char v=pix[(size_t)(cy*4+yy)*pw+cx*2+xx];if(v)mask|=bit[yy][xx];if(v>kind)kind=v;}if(mask){char u[4];utf8(0x2800u+(unsigned)mask,u);int color=kind==30?15:kind>=10?class_color[kind-10]:kind==3?118:kind==2?45:240;printf("\033[38;5;%dm%s",color,u);}else fputc(' ',stdout);}fputs("\033[0m\n",stdout);}
 printf("\033[2K\033[38;5;244m view x=%+.1f y=%+.1f zoom=%.2fx yaw=%+.0f° score≥%.2f box:%s vel:%s grid:%s trail:%s classes:%03x\033[0m\n",s->center_x,s->center_y,s->zoom,s->yaw*57.29578f,s->score_threshold,s->show_boxes?"on":"off",s->show_velocity?"on":"off",s->show_grid?"on":"off",s->show_tracks?"on":"off",s->class_mask);
 if(selected>=0){const pp_box*b=&d->boxes[selected];int track_id=0,age=0;float nearest=.01f;for(size_t i=0;i<s->track_count;i++){const pp_tui_track*t=&s->tracks[i];if(!t->length||t->class_id!=b->class_id)continue;float dx=t->x[t->length-1]-b->x,dy=t->y[t->length-1]-b->y,d2=dx*dx+dy*dy;if(d2<nearest){nearest=d2;track_id=t->id;age=t->length;}}printf("\033[2K\033[38;5;%dm selected %d/%zu track #%d age %d %-12s score %.3f xyz %+.1f %+.1f %+.1f size %.1f %.1f %.1f yaw %+.0f° vel %.1f %.1f\033[0m\n",class_color[b->class_id],s->selected%(int)visible+1,visible,track_id,age,class_name[b->class_id],b->score,b->x,b->y,b->z,b->dx,b->dy,b->dz,b->yaw*57.29578f,b->vx,b->vy);}else fputs("\033[2K\n",stdout);
 fputs("\033[2K\033[38;5;250m Space play · ←/→ frame · WASD pan · z/e rotate · +/- zoom · 0-9 classes · ,/. score · [/] select · b/v/g/t layers · r reset · q quit\033[0m",stdout);fflush(stdout);free(pix);
}
