#include "pp_decode.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define PP_PI 3.14159265358979323846f
enum{PRE_MAX=1000,POST_PER_CLASS=83};typedef struct{pp_box b;}candidate;typedef struct{float x,y;}p2;
static const int head_class0[6]={0,1,3,5,6,8};
static const float anchors[10][4]={{4.63f,1.97f,1.74f,-.95f},{6.93f,2.51f,2.84f,-.6f},{6.37f,2.85f,3.19f,-.225f},{10.5f,2.94f,3.47f,-.085f},{12.29f,2.90f,3.87f,.115f},{.50f,2.53f,.98f,-1.33f},{2.11f,.77f,1.47f,-1.085f},{1.70f,.60f,1.28f,-1.18f},{.73f,.67f,1.77f,-.935f},{.41f,.41f,1.07f,-1.285f}};
int pp_detections_alloc(pp_detections*d,size_t n){memset(d,0,sizeof(*d));d->boxes=calloc(n,sizeof(pp_box));if(!d->boxes)return 0;d->capacity=n;return 1;}void pp_detections_free(pp_detections*d){if(d){free(d->boxes);memset(d,0,sizeof(*d));}}
static float sigmoid(float x){return x>=0?1/(1+expf(-x)):expf(x)/(1+expf(x));}static int desc(const void*a,const void*b){float x=((const candidate*)a)->b.score,y=((const candidate*)b)->b.score;return x<y?1:x>y?-1:0;}
static void corners(const pp_box*b,p2*p){float c=cosf(b->yaw),s=sinf(b->yaw),hx=b->dx*.5f,hy=b->dy*.5f;const float q[4][2]={{-hx,-hy},{hx,-hy},{hx,hy},{-hx,hy}};for(int i=0;i<4;i++){p[i].x=b->x+q[i][0]*c-q[i][1]*s;p[i].y=b->y+q[i][0]*s+q[i][1]*c;}}
static float cross(p2 a,p2 b,p2 c){return(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);}static p2 inter(p2 a,p2 b,p2 c,p2 d){float x=b.x-a.x,y=b.y-a.y,u=d.x-c.x,v=d.y-c.y,t=((c.x-a.x)*v-(c.y-a.y)*u)/(x*v-y*u);p2 p={a.x+t*x,a.y+t*y};return p;}
static float iou(const pp_box*a,const pp_box*b){p2 s[16],tmp[16],cl[4];corners(a,s);corners(b,cl);int n=4;for(int e=0;e<4&&n;e++){int m=0;p2 c=cl[e],d=cl[(e+1)&3],prev=s[n-1];int pin=cross(c,d,prev)>=0;for(int i=0;i<n;i++){p2 cur=s[i];int in=cross(c,d,cur)>=0;if(in!=pin)tmp[m++]=inter(prev,cur,c,d);if(in)tmp[m++]=cur;prev=cur;pin=in;}memcpy(s,tmp,m*sizeof(p2));n=m;}float area=0;for(int i=0;i<n;i++)area+=s[i].x*s[(i+1)%n].y-s[i].y*s[(i+1)%n].x;area=fabsf(area)*.5f;float u=a->dx*a->dy+b->dx*b->dy-area;return u>0?area/u:0;}
static float code(const pp_raw_output*r,int h,int anchor,size_t p,int q){int channel=anchor*10+q,prefix=0;for(int br=1;br<6;br++){int n=pp_branch_channels(h,br);if(channel<prefix+n)return pp_output_branch(r,h,br)[(size_t)(channel-prefix)*128*128+p];prefix+=n;}return 0;}
int pp_decode(const pp_raw_output*r,float score_thr,float nms_thr,pp_detections*out){if(!r||!r->data||!out||!out->boxes)return 0;size_t hw=128u*128;candidate*all=malloc(hw*20*2*sizeof(*all));if(!all)return 0;out->count=0;
 for(int global=0;global<10;global++){size_t n=0;for(int h=0;h<6;h++){int c=pp_head_classes(h),local=global-head_class0[h];if(local<0||local>=c)continue;int A=2*c;const float*cls=pp_output_branch(r,h,0);for(int a=0;a<A;a++)for(size_t p=0;p<hw;p++){float score=sigmoid(cls[((size_t)a*c+local)*hw+p]);if(score<score_thr)continue;int ac=head_class0[h]+a/2,rot=a&1;const float*an=anchors[ac];float xa=-51.2f+(p%128)*(102.4f/127),ya=-51.2f+(p/128)*(102.4f/127),za=an[3]+an[2]*.5f,ra=rot?1.57f:0,diag=sqrtf(an[0]*an[0]+an[1]*an[1]);float xt=code(r,h,a,p,0),yt=code(r,h,a,p,1),zt=code(r,h,a,p,2),cost=code(r,h,a,p,6),sint=code(r,h,a,p,7);pp_box b={xt*diag+xa,yt*diag+ya,zt*an[2]+za,expf(code(r,h,a,p,3))*an[0],expf(code(r,h,a,p,4))*an[1],expf(code(r,h,a,p,5))*an[2],atan2f(sint+sinf(ra),cost+cosf(ra)),code(r,h,a,p,8),code(r,h,a,p,9),score,global};all[n++].b=b;}}
  qsort(all,n,sizeof(*all),desc);if(n>PRE_MAX)n=PRE_MAX;size_t before=out->count,limit=before+POST_PER_CLASS;if(limit>out->capacity)limit=out->capacity;for(size_t i=0;i<n&&out->count<limit;i++){int keep=1;for(size_t j=before;j<out->count;j++)if(iou(&all[i].b,&out->boxes[j])>nms_thr){keep=0;break;}if(keep)out->boxes[out->count++]=all[i].b;}
 }
 free(all);return 1;}
