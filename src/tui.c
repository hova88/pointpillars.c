#include "pp_tui.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

static void plot(unsigned char*p,int w,int h,int x,int y,unsigned char v){if((unsigned)x<(unsigned)w&&(unsigned)y<(unsigned)h&&p[(size_t)y*w+x]<v)p[(size_t)y*w+x]=v;}
static void line(unsigned char*p,int w,int h,int x0,int y0,int x1,int y1){int dx=abs(x1-x0),sx=x0<x1?1:-1,dy=-abs(y1-y0),sy=y0<y1?1:-1,e=dx+dy;for(;;){plot(p,w,h,x0,y0,2);if(x0==x1&&y0==y1)break;int q=2*e;if(q>=dy){e+=dy;x0+=sx;}if(q<=dx){e+=dx;y0+=sy;}}}
static void utf8(unsigned cp,char s[5]){s[0]=(char)(0xe0|(cp>>12));s[1]=(char)(0x80|((cp>>6)&63));s[2]=(char)(0x80|(cp&63));s[3]=0;}

void pp_tui_render(const float*pts,size_t n,size_t stride,const pp_detections*d,size_t frame,double infer,const char*backend){
 struct winsize z={0};ioctl(STDOUT_FILENO,TIOCGWINSZ,&z);int cols=z.ws_col?z.ws_col:100,rows=z.ws_row?z.ws_row:36;if(cols<20)cols=20;if(rows<8)rows=8;int ch=rows-3,pw=cols*2,ph=ch*4;
 unsigned char*pix=(unsigned char*)calloc((size_t)pw*ph,1);if(!pix)return;
 for(size_t i=0;i<n;i++){const float*q=pts+i*stride;float x=q[0],y=q[1];int px=(int)((y+51.2f)/102.4f*(pw-1)),py=(int)((51.2f-x)/102.4f*(ph-1));plot(pix,pw,ph,px,py,1);}
 if(d)for(size_t j=0;j<d->count;j++){const pp_box*b=d->boxes+j;float c=cosf(b->yaw),s=sinf(b->yaw),hx=b->dx*.5f,hy=b->dy*.5f;int x[4],y[4];for(int k=0;k<4;k++){float lx=(k==0||k==3)?-hx:hx,ly=k<2?-hy:hy,gx=b->x+lx*c-ly*s,gy=b->y+lx*s+ly*c;x[k]=(int)((gy+51.2f)/102.4f*(pw-1));y[k]=(int)((51.2f-gx)/102.4f*(ph-1));}for(int k=0;k<4;k++)line(pix,pw,ph,x[k],y[k],x[(k+1)&3],y[(k+1)&3]);}
 printf("\033[?25l\033[H\033[38;5;51m PointPillars\033[0m  frame %-5zu  %-4s %7.2f ms  points %-6zu  boxes %-3zu  \033[2mCtrl-C to exit\033[0m\n",frame,backend,infer,n,d?d->count:0);
 static const int bit[4][2]={{1,8},{2,16},{4,32},{64,128}};
 for(int cy=0;cy<ch;cy++){for(int cx=0;cx<cols;cx++){int mask=0,box=0;for(int yy=0;yy<4;yy++)for(int xx=0;xx<2;xx++){unsigned char v=pix[(size_t)(cy*4+yy)*pw+cx*2+xx];if(v)mask|=bit[yy][xx];if(v==2)box=1;}if(mask){char u[5];utf8(0x2800u+(unsigned)mask,u);printf("\033[%sm%s",box?"38;5;208":"38;5;45",u);}else fputc(' ',stdout);}printf("\033[0m\n");}
 printf("\033[38;5;240m rear/ego ───────────────────── y(left/right) ───────────────────── forward at top\033[0m\n");fflush(stdout);free(pix);
}
