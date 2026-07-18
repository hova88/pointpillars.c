#define _POSIX_C_SOURCE 200809L
#include "pp_infer.h"
#include "pp_kernels.h"
#ifdef PP_WITH_ACCELERATE
#include "pp_apple.h"
#endif

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

static int fail(char *e, size_t n, const char *fmt, ...) {
    if (e && n) { va_list ap; va_start(ap, fmt); vsnprintf(e, n, fmt, ap); va_end(ap); }
    return 0;
}

static float *falloc(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(float))) return NULL;
    return (float *)p;
}

#ifdef __AVX2__
/* Eight output accumulators fit in the sixteen-register AVX2 file.  Splitting
 * work by (output-channel block, row) keeps all workers busy when co=64. */
static void conv3s1_relu8(const float *restrict x,float *restrict y,
                          const float *restrict w,const float *restrict b,
                          int ci,int co,int h,int width){
 size_t plane=(size_t)h*width;
 #pragma omp parallel for collapse(2) schedule(static)
 for(int og=0;og<co;og+=8)for(int oy=0;oy<h;oy++){
  float *y0=y+(size_t)(og+0)*plane,*y1=y+(size_t)(og+1)*plane;
  float *y2=y+(size_t)(og+2)*plane,*y3=y+(size_t)(og+3)*plane;
  float *y4=y+(size_t)(og+4)*plane,*y5=y+(size_t)(og+5)*plane;
  float *y6=y+(size_t)(og+6)*plane,*y7=y+(size_t)(og+7)*plane;
  for(int ox=0;ox<width;){
   if(ox>=1&&ox+7<width-1){
    __m256 a0=_mm256_set1_ps(b[og]),a1=_mm256_set1_ps(b[og+1]);
    __m256 a2=_mm256_set1_ps(b[og+2]),a3=_mm256_set1_ps(b[og+3]);
    __m256 a4=_mm256_set1_ps(b[og+4]),a5=_mm256_set1_ps(b[og+5]);
    __m256 a6=_mm256_set1_ps(b[og+6]),a7=_mm256_set1_ps(b[og+7]);
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){
     int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;
     const float *row=x+((size_t)ic*h+iy)*width;
     for(int kx=0;kx<3;kx++){
      __m256 xv=_mm256_loadu_ps(row+ox+kx-1);int k=ky*3+kx;
      a0=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+0)*ci+ic)*9+k]),a0);
      a1=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+1)*ci+ic)*9+k]),a1);
      a2=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+2)*ci+ic)*9+k]),a2);
      a3=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+3)*ci+ic)*9+k]),a3);
      a4=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+4)*ci+ic)*9+k]),a4);
      a5=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+5)*ci+ic)*9+k]),a5);
      a6=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+6)*ci+ic)*9+k]),a6);
      a7=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+7)*ci+ic)*9+k]),a7);
     }
    }
    __m256 z=_mm256_setzero_ps();
    _mm256_storeu_ps(y0+(size_t)oy*width+ox,_mm256_max_ps(a0,z));
    _mm256_storeu_ps(y1+(size_t)oy*width+ox,_mm256_max_ps(a1,z));
    _mm256_storeu_ps(y2+(size_t)oy*width+ox,_mm256_max_ps(a2,z));
    _mm256_storeu_ps(y3+(size_t)oy*width+ox,_mm256_max_ps(a3,z));
    _mm256_storeu_ps(y4+(size_t)oy*width+ox,_mm256_max_ps(a4,z));
    _mm256_storeu_ps(y5+(size_t)oy*width+ox,_mm256_max_ps(a5,z));
    _mm256_storeu_ps(y6+(size_t)oy*width+ox,_mm256_max_ps(a6,z));
    _mm256_storeu_ps(y7+(size_t)oy*width+ox,_mm256_max_ps(a7,z));ox+=8;
   }else{
    float a[8]={b[og],b[og+1],b[og+2],b[og+3],b[og+4],b[og+5],b[og+6],b[og+7]};
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;for(int kx=0;kx<3;kx++){int ix=ox+kx-1;if((unsigned)ix>=(unsigned)width)continue;float v=x[((size_t)ic*h+iy)*width+ix];int k=ky*3+kx;for(int q=0;q<8;q++)a[q]+=v*w[((size_t)(og+q)*ci+ic)*9+k];}}
    size_t p=(size_t)oy*width+ox;y0[p]=a[0]>0?a[0]:0;y1[p]=a[1]>0?a[1]:0;y2[p]=a[2]>0?a[2]:0;y3[p]=a[3]>0?a[3]:0;y4[p]=a[4]>0?a[4]:0;y5[p]=a[5]>0?a[5]:0;y6[p]=a[6]>0?a[6]:0;y7[p]=a[7]>0?a[7]:0;ox++;
   }
  }
 }
}

/* Stride-1 dominates the backbone. Four output channels share every loaded
 * input vector while their accumulators remain in registers for the full K
 * reduction. Boundary pixels use a compact scalar path. */
static void conv3s1_relu4(const float *restrict x,float *restrict y,
                          const float *restrict w,const float *restrict b,
                          int ci,int co,int h,int width){
 size_t plane=(size_t)h*width;
 #pragma omp parallel for schedule(static)
 for(int og=0;og<co;og+=4){
  float *y0=y+(size_t)(og+0)*plane,*y1=y+(size_t)(og+1)*plane;
  float *y2=y+(size_t)(og+2)*plane,*y3=y+(size_t)(og+3)*plane;
  for(int oy=0;oy<h;oy++){
   int ox=0;
   for(;ox<width;){
    if(ox>=1&&ox+7<width-1){
     __m256 a0=_mm256_set1_ps(b[og]),a1=_mm256_set1_ps(b[og+1]);
     __m256 a2=_mm256_set1_ps(b[og+2]),a3=_mm256_set1_ps(b[og+3]);
     for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){
      int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;
      const float *row=x+((size_t)ic*h+iy)*width;
      for(int kx=0;kx<3;kx++){
       __m256 xv=_mm256_loadu_ps(row+ox+kx-1);int k=ky*3+kx;
       a0=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+0)*ci+ic)*9+k]),a0);
       a1=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+1)*ci+ic)*9+k]),a1);
       a2=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+2)*ci+ic)*9+k]),a2);
       a3=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+3)*ci+ic)*9+k]),a3);
      }
     }
     __m256 z=_mm256_setzero_ps();_mm256_storeu_ps(y0+(size_t)oy*width+ox,_mm256_max_ps(a0,z));
     _mm256_storeu_ps(y1+(size_t)oy*width+ox,_mm256_max_ps(a1,z));
     _mm256_storeu_ps(y2+(size_t)oy*width+ox,_mm256_max_ps(a2,z));
     _mm256_storeu_ps(y3+(size_t)oy*width+ox,_mm256_max_ps(a3,z));ox+=8;
    }else{
     float a[4]={b[og],b[og+1],b[og+2],b[og+3]};
     for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;for(int kx=0;kx<3;kx++){int ix=ox+kx-1;if((unsigned)ix>=(unsigned)width)continue;float v=x[((size_t)ic*h+iy)*width+ix];int k=ky*3+kx;for(int q=0;q<4;q++)a[q]+=v*w[((size_t)(og+q)*ci+ic)*9+k];}}
     size_t p=(size_t)oy*width+ox;y0[p]=a[0]>0?a[0]:0;y1[p]=a[1]>0?a[1]:0;y2[p]=a[2]>0?a[2]:0;y3[p]=a[3]>0?a[3]:0;ox++;
    }
   }
  }
 }
}

static void conv3s2_relu8(const float *restrict x,float *restrict y,
                          const float *restrict w,const float *restrict b,
                          int ci,int co,int hi,int wi){
 int h=(hi+1)/2,width=(wi+1)/2;size_t inplane=(size_t)hi*wi,outplane=(size_t)h*width;
 const __m256i gi=_mm256_setr_epi32(0,2,4,6,8,10,12,14);
 #pragma omp parallel for collapse(2) schedule(static)
 for(int og=0;og<co;og+=8)for(int oy=0;oy<h;oy++){
  float *yo[8]={y+(size_t)(og+0)*outplane,y+(size_t)(og+1)*outplane,
                y+(size_t)(og+2)*outplane,y+(size_t)(og+3)*outplane,
                y+(size_t)(og+4)*outplane,y+(size_t)(og+5)*outplane,
                y+(size_t)(og+6)*outplane,y+(size_t)(og+7)*outplane};
  for(int ox=0;ox<width;){
   if(ox>=1&&ox+7<width){
    __m256 a0=_mm256_set1_ps(b[og]),a1=_mm256_set1_ps(b[og+1]);
    __m256 a2=_mm256_set1_ps(b[og+2]),a3=_mm256_set1_ps(b[og+3]);
    __m256 a4=_mm256_set1_ps(b[og+4]),a5=_mm256_set1_ps(b[og+5]);
    __m256 a6=_mm256_set1_ps(b[og+6]),a7=_mm256_set1_ps(b[og+7]);
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){
     int iy=oy*2+ky-1;if((unsigned)iy>=(unsigned)hi)continue;
     const float *row=x+(size_t)ic*inplane+(size_t)iy*wi;
     for(int kx=0;kx<3;kx++){
      __m256 xv=_mm256_i32gather_ps(row+ox*2+kx-1,gi,4);int k=ky*3+kx;
      a0=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+0)*ci+ic)*9+k]),a0);
      a1=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+1)*ci+ic)*9+k]),a1);
      a2=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+2)*ci+ic)*9+k]),a2);
      a3=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+3)*ci+ic)*9+k]),a3);
      a4=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+4)*ci+ic)*9+k]),a4);
      a5=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+5)*ci+ic)*9+k]),a5);
      a6=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+6)*ci+ic)*9+k]),a6);
      a7=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+7)*ci+ic)*9+k]),a7);
     }
    }
    __m256 z=_mm256_setzero_ps();
    _mm256_storeu_ps(yo[0]+(size_t)oy*width+ox,_mm256_max_ps(a0,z));
    _mm256_storeu_ps(yo[1]+(size_t)oy*width+ox,_mm256_max_ps(a1,z));
    _mm256_storeu_ps(yo[2]+(size_t)oy*width+ox,_mm256_max_ps(a2,z));
    _mm256_storeu_ps(yo[3]+(size_t)oy*width+ox,_mm256_max_ps(a3,z));
    _mm256_storeu_ps(yo[4]+(size_t)oy*width+ox,_mm256_max_ps(a4,z));
    _mm256_storeu_ps(yo[5]+(size_t)oy*width+ox,_mm256_max_ps(a5,z));
    _mm256_storeu_ps(yo[6]+(size_t)oy*width+ox,_mm256_max_ps(a6,z));
    _mm256_storeu_ps(yo[7]+(size_t)oy*width+ox,_mm256_max_ps(a7,z));ox+=8;
   }else{
    float a[8]={b[og],b[og+1],b[og+2],b[og+3],b[og+4],b[og+5],b[og+6],b[og+7]};
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){int iy=oy*2+ky-1;if((unsigned)iy>=(unsigned)hi)continue;for(int kx=0;kx<3;kx++){int ix=ox*2+kx-1;if((unsigned)ix>=(unsigned)wi)continue;float v=x[(size_t)ic*inplane+(size_t)iy*wi+ix];int k=ky*3+kx;for(int q=0;q<8;q++)a[q]+=v*w[((size_t)(og+q)*ci+ic)*9+k];}}
    size_t p=(size_t)oy*width+ox;for(int q=0;q<8;q++)yo[q][p]=a[q]>0?a[q]:0;ox++;
   }
  }
 }
}

static void conv3s2_relu4(const float *restrict x,float *restrict y,
                          const float *restrict w,const float *restrict b,
                          int ci,int co,int hi,int wi){
 int h=(hi+1)/2,width=(wi+1)/2;size_t inplane=(size_t)hi*wi,outplane=(size_t)h*width;
 const __m256i gi=_mm256_setr_epi32(0,2,4,6,8,10,12,14);
 #pragma omp parallel for schedule(static)
 for(int og=0;og<co;og+=4){
  float *yo[4]={y+(size_t)(og+0)*outplane,y+(size_t)(og+1)*outplane,y+(size_t)(og+2)*outplane,y+(size_t)(og+3)*outplane};
  for(int oy=0;oy<h;oy++)for(int ox=0;ox<width;){
   if(ox>=1&&ox+7<width){
    __m256 a[4]={_mm256_set1_ps(b[og]),_mm256_set1_ps(b[og+1]),_mm256_set1_ps(b[og+2]),_mm256_set1_ps(b[og+3])};
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){int iy=oy*2+ky-1;if((unsigned)iy>=(unsigned)hi)continue;const float *row=x+(size_t)ic*inplane+(size_t)iy*wi;for(int kx=0;kx<3;kx++){__m256 xv=_mm256_i32gather_ps(row+ox*2+kx-1,gi,4);int k=ky*3+kx;for(int q=0;q<4;q++)a[q]=_mm256_fmadd_ps(xv,_mm256_set1_ps(w[((size_t)(og+q)*ci+ic)*9+k]),a[q]);}}
    __m256 z=_mm256_setzero_ps();for(int q=0;q<4;q++)_mm256_storeu_ps(yo[q]+(size_t)oy*width+ox,_mm256_max_ps(a[q],z));ox+=8;
   }else{
    float a[4]={b[og],b[og+1],b[og+2],b[og+3]};
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){int iy=oy*2+ky-1;if((unsigned)iy>=(unsigned)hi)continue;for(int kx=0;kx<3;kx++){int ix=ox*2+kx-1;if((unsigned)ix>=(unsigned)wi)continue;float v=x[(size_t)ic*inplane+(size_t)iy*wi+ix];int k=ky*3+kx;for(int q=0;q<4;q++)a[q]+=v*w[((size_t)(og+q)*ci+ic)*9+k];}}
    size_t p=(size_t)oy*width+ox;for(int q=0;q<4;q++)yo[q][p]=a[q]>0?a[q]:0;ox++;
   }
  }
 }
}
#endif

int pp_output_alloc(pp_raw_output *o) {
    memset(o, 0, sizeof(*o));
    size_t channels=0;for(int h=0;h<PP_HEADS;h++)for(int b=0;b<PP_BRANCHES;b++)channels+=pp_branch_channels(h,b);
    o->floats=channels*(size_t)PP_OUT_H*PP_OUT_W;o->data=falloc(o->floats);
    return o->data!=NULL;
}

void pp_output_free(pp_raw_output *o) {
    if (!o) return;
    free(o->data);memset(o,0,sizeof(*o));
}
void pp_cpu_workspace_free(pp_cpu_workspace *workspace) {
    if (!workspace) return;
    free(workspace->data);
    memset(workspace, 0, sizeof(*workspace));
}
int pp_head_classes(int h){static const int n[6]={1,2,2,1,2,2};return (unsigned)h<6?n[h]:0;}
int pp_branch_channels(int h,int b){static const int mul[6]={0,4,2,6,4,4};int c=pp_head_classes(h);return b==0?2*c*c:(unsigned)b<6?c*mul[b]:0;}
float *pp_output_branch(const pp_raw_output*o,int h,int b){size_t off=0,hw=(size_t)PP_OUT_H*PP_OUT_W;for(int i=0;i<h;i++)for(int j=0;j<6;j++)off+=(size_t)pp_branch_channels(i,j)*hw;for(int j=0;j<b;j++)off+=(size_t)pp_branch_channels(h,j)*hw;return o&&o->data?o->data+off:NULL;}

/* OIHW 3x3 convolution, NCHW, pad=1, fused ReLU. Parallel output channels
 * own disjoint planes. The inner x loop is contiguous and auto-vectorizes. */
void pp_cpu_conv3_relu(const float *restrict x, float *restrict y,
                       const float *restrict w, const float *restrict b,
                       int ci, int co, int hi, int wi, int stride) {
#ifdef PP_WITH_ACCELERATE
    if (pp_apple_conv(x, y, w, b, ci, co, hi, wi, 3, stride, 1, 1)) return;
#endif
#ifdef __AVX2__
    static int use_oc8 = -1;
    static int use_s2oc8 = -1;
    if (use_oc8 < 0) use_oc8 = getenv("PP_CPU_OC4") == NULL;
    if (use_s2oc8 < 0) use_s2oc8 = use_oc8 && getenv("PP_CPU_S2OC4") == NULL;
    if(stride==1&&!(co&7)&&use_oc8){conv3s1_relu8(x,y,w,b,ci,co,hi,wi);return;}
    if(stride==2&&!(co&7)&&use_s2oc8){conv3s2_relu8(x,y,w,b,ci,co,hi,wi);return;}
    if(stride==1&&!(co&3)){conv3s1_relu4(x,y,w,b,ci,co,hi,wi);return;}
    if(stride==2&&!(co&3)){conv3s2_relu4(x,y,w,b,ci,co,hi,wi);return;}
#endif
    int ho = (hi + stride - 1) / stride, wo = (wi + stride - 1) / stride;
    size_t ni = (size_t)hi * wi, no = (size_t)ho * wo;
    #pragma omp parallel for schedule(static)
    for (int oc = 0; oc < co; ++oc) {
        float *yo = y + (size_t)oc * no;
        /* Eight rows are 6.75 KiB at the widest layer and remain in L1 while
         * all input channels/kernel taps accumulate into them. */
        for (int yb = 0; yb < ho; yb += 8) {
            int ye = yb + 8 < ho ? yb + 8 : ho;
            for (int oy = yb; oy < ye; ++oy) {
                float *dst = yo + (size_t)oy * wo;
                #pragma omp simd
                for (int ox = 0; ox < wo; ++ox) dst[ox] = b[oc];
            }
            for (int ic = 0; ic < ci; ++ic) {
                const float *xi = x + (size_t)ic * ni;
                const float *wk = w + ((size_t)oc * ci + ic) * 9;
                for (int ky = 0; ky < 3; ++ky) {
                    for (int oy = yb; oy < ye; ++oy) {
                        int iy = oy * stride + ky - 1;
                        if ((unsigned)iy >= (unsigned)hi) continue;
                        const float *row = xi + (size_t)iy * wi;
                        float *dst = yo + (size_t)oy * wo;
                        for (int kx = 0; kx < 3; ++kx) {
                            float a = wk[ky * 3 + kx];
                            int first = kx ? 0 : 1;
                            int last = wo;
                            if (first * stride + kx - 1 < 0) ++first;
                            while (last > first && (last - 1) * stride + kx - 1 >= wi) --last;
                            #pragma omp simd
                            for (int ox = first; ox < last; ++ox)
                                dst[ox] += row[ox * stride + kx - 1] * a;
                        }
                    }
                }
            }
            for (int oy = yb; oy < ye; ++oy) {
                float *dst = yo + (size_t)oy * wo;
                #pragma omp simd
                for (int ox = 0; ox < wo; ++ox) if (dst[ox] < 0.0f) dst[ox] = 0.0f;
            }
        }
    }
}

static void conv3_plain_accum(const float*x,float*y,const float*w,const float*b,int ci,int co,int h,int wi){size_t n=(size_t)h*wi;
 #pragma omp parallel for schedule(static)
 for(int oc=0;oc<co;oc++){float*yo=y+(size_t)oc*n;for(size_t p=0;p<n;p++)yo[p]=b[oc];for(int ic=0;ic<ci;ic++){const float*xi=x+(size_t)ic*n;const float*wk=w+((size_t)oc*ci+ic)*9;for(int ky=0;ky<3;ky++)for(int oy=0;oy<h;oy++){int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;for(int kx=0;kx<3;kx++){int first=kx?0:1,last=wi;if(kx==2)last--;float a=wk[ky*3+kx];
   #pragma omp simd
   for(int ox=first;ox<last;ox++)yo[(size_t)oy*wi+ox]+=xi[(size_t)iy*wi+ox+kx-1]*a;}}}}
}

#ifdef __AVX2__
static void conv3_plain_direct1(const float *restrict x,float *restrict y,
                                const float *restrict w,const float *restrict b,
                                int ci,int co,int h,int width){
 size_t plane=(size_t)h*width;
 #pragma omp parallel for collapse(2) schedule(static)
 for(int oc=0;oc<co;oc++)for(int oy=0;oy<h;oy++){
  float *yo=y+(size_t)oc*plane;
  for(int ox=0;ox<width;){
   if(ox>=1&&ox+7<width-1){
    __m256 a=_mm256_set1_ps(b[oc]);
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){
     int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;
     const float *row=x+((size_t)ic*h+iy)*width;
     const float *wk=w+((size_t)oc*ci+ic)*9+ky*3;
     for(int kx=0;kx<3;kx++)a=_mm256_fmadd_ps(_mm256_loadu_ps(row+ox+kx-1),_mm256_set1_ps(wk[kx]),a);
    }
    _mm256_storeu_ps(yo+(size_t)oy*width+ox,a);ox+=8;
   }else{
    float a=b[oc];
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;for(int kx=0;kx<3;kx++){int ix=ox+kx-1;if((unsigned)ix>=(unsigned)width)continue;a+=x[((size_t)ic*h+iy)*width+ix]*w[((size_t)oc*ci+ic)*9+ky*3+kx];}}
    yo[(size_t)oy*width+ox]=a;ox++;
   }
  }
 }
}

static void conv3_plain_direct2(const float *restrict x,float *restrict y,
                                const float *restrict w,const float *restrict b,
                                int ci,int co,int h,int width){
 size_t plane=(size_t)h*width;
 #pragma omp parallel for collapse(2) schedule(static)
 for(int og=0;og<co;og+=2)for(int oy=0;oy<h;oy++){
  float *y0=y+(size_t)og*plane,*y1=y+(size_t)(og+1)*plane;
  for(int ox=0;ox<width;){
   if(ox>=1&&ox+7<width-1){
    __m256 a0=_mm256_set1_ps(b[og]),a1=_mm256_set1_ps(b[og+1]);
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){
     int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;
     const float *row=x+((size_t)ic*h+iy)*width;
     const float *w0=w+((size_t)og*ci+ic)*9+ky*3;
     const float *w1=w+((size_t)(og+1)*ci+ic)*9+ky*3;
     for(int kx=0;kx<3;kx++){__m256 xv=_mm256_loadu_ps(row+ox+kx-1);a0=_mm256_fmadd_ps(xv,_mm256_set1_ps(w0[kx]),a0);a1=_mm256_fmadd_ps(xv,_mm256_set1_ps(w1[kx]),a1);}
    }
    _mm256_storeu_ps(y0+(size_t)oy*width+ox,a0);_mm256_storeu_ps(y1+(size_t)oy*width+ox,a1);ox+=8;
   }else{
    float a0=b[og],a1=b[og+1];
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;for(int kx=0;kx<3;kx++){int ix=ox+kx-1;if((unsigned)ix>=(unsigned)width)continue;float v=x[((size_t)ic*h+iy)*width+ix];a0+=v*w[((size_t)og*ci+ic)*9+ky*3+kx];a1+=v*w[((size_t)(og+1)*ci+ic)*9+ky*3+kx];}}
    size_t p=(size_t)oy*width+ox;y0[p]=a0;y1[p]=a1;ox++;
   }
  }
 }
}

static void conv3_plain_direct4(const float *restrict x,float *restrict y,
                                const float *restrict w,const float *restrict b,
                                int ci,int co,int h,int width){
 size_t plane=(size_t)h*width;
 #pragma omp parallel for collapse(2) schedule(static)
 for(int og=0;og<co;og+=4)for(int oy=0;oy<h;oy++){
  float *y0=y+(size_t)(og+0)*plane,*y1=y+(size_t)(og+1)*plane;
  float *y2=y+(size_t)(og+2)*plane,*y3=y+(size_t)(og+3)*plane;
  for(int ox=0;ox<width;){
   if(ox>=1&&ox+7<width-1){
    __m256 a0=_mm256_set1_ps(b[og]),a1=_mm256_set1_ps(b[og+1]);
    __m256 a2=_mm256_set1_ps(b[og+2]),a3=_mm256_set1_ps(b[og+3]);
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){
     int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;
     const float *row=x+((size_t)ic*h+iy)*width;
     const float *w0=w+((size_t)(og+0)*ci+ic)*9+ky*3;
     const float *w1=w+((size_t)(og+1)*ci+ic)*9+ky*3;
     const float *w2=w+((size_t)(og+2)*ci+ic)*9+ky*3;
     const float *w3=w+((size_t)(og+3)*ci+ic)*9+ky*3;
     for(int kx=0;kx<3;kx++){__m256 xv=_mm256_loadu_ps(row+ox+kx-1);a0=_mm256_fmadd_ps(xv,_mm256_set1_ps(w0[kx]),a0);a1=_mm256_fmadd_ps(xv,_mm256_set1_ps(w1[kx]),a1);a2=_mm256_fmadd_ps(xv,_mm256_set1_ps(w2[kx]),a2);a3=_mm256_fmadd_ps(xv,_mm256_set1_ps(w3[kx]),a3);}
    }
    size_t p=(size_t)oy*width+ox;_mm256_storeu_ps(y0+p,a0);_mm256_storeu_ps(y1+p,a1);_mm256_storeu_ps(y2+p,a2);_mm256_storeu_ps(y3+p,a3);ox+=8;
   }else{
    float a0=b[og],a1=b[og+1],a2=b[og+2],a3=b[og+3];
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;for(int kx=0;kx<3;kx++){int ix=ox+kx-1;if((unsigned)ix>=(unsigned)width)continue;float v=x[((size_t)ic*h+iy)*width+ix];int k=ky*3+kx;a0+=v*w[((size_t)(og+0)*ci+ic)*9+k];a1+=v*w[((size_t)(og+1)*ci+ic)*9+k];a2+=v*w[((size_t)(og+2)*ci+ic)*9+k];a3+=v*w[((size_t)(og+3)*ci+ic)*9+k];}}
    size_t p=(size_t)oy*width+ox;y0[p]=a0;y1[p]=a1;y2[p]=a2;y3[p]=a3;ox++;
   }
  }
 }
}

static void conv3_plain_direct8(const float *restrict x,float *restrict y,
                                const float *restrict w,const float *restrict b,
                                int ci,int co,int h,int width){
 size_t plane=(size_t)h*width;
 #pragma omp parallel for collapse(2) schedule(static)
 for(int og=0;og<co;og+=8)for(int oy=0;oy<h;oy++){
  float *yo[8]={y+(size_t)(og+0)*plane,y+(size_t)(og+1)*plane,
                y+(size_t)(og+2)*plane,y+(size_t)(og+3)*plane,
                y+(size_t)(og+4)*plane,y+(size_t)(og+5)*plane,
                y+(size_t)(og+6)*plane,y+(size_t)(og+7)*plane};
  for(int ox=0;ox<width;){
   if(ox>=1&&ox+7<width-1){
    __m256 a0=_mm256_set1_ps(b[og]),a1=_mm256_set1_ps(b[og+1]);
    __m256 a2=_mm256_set1_ps(b[og+2]),a3=_mm256_set1_ps(b[og+3]);
    __m256 a4=_mm256_set1_ps(b[og+4]),a5=_mm256_set1_ps(b[og+5]);
    __m256 a6=_mm256_set1_ps(b[og+6]),a7=_mm256_set1_ps(b[og+7]);
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){
     int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;
     const float *row=x+((size_t)ic*h+iy)*width;
     const float *w0=w+((size_t)(og+0)*ci+ic)*9+ky*3;
     const float *w1=w+((size_t)(og+1)*ci+ic)*9+ky*3;
     const float *w2=w+((size_t)(og+2)*ci+ic)*9+ky*3;
     const float *w3=w+((size_t)(og+3)*ci+ic)*9+ky*3;
     const float *w4=w+((size_t)(og+4)*ci+ic)*9+ky*3;
     const float *w5=w+((size_t)(og+5)*ci+ic)*9+ky*3;
     const float *w6=w+((size_t)(og+6)*ci+ic)*9+ky*3;
     const float *w7=w+((size_t)(og+7)*ci+ic)*9+ky*3;
     for(int kx=0;kx<3;kx++){__m256 xv=_mm256_loadu_ps(row+ox+kx-1);a0=_mm256_fmadd_ps(xv,_mm256_set1_ps(w0[kx]),a0);a1=_mm256_fmadd_ps(xv,_mm256_set1_ps(w1[kx]),a1);a2=_mm256_fmadd_ps(xv,_mm256_set1_ps(w2[kx]),a2);a3=_mm256_fmadd_ps(xv,_mm256_set1_ps(w3[kx]),a3);a4=_mm256_fmadd_ps(xv,_mm256_set1_ps(w4[kx]),a4);a5=_mm256_fmadd_ps(xv,_mm256_set1_ps(w5[kx]),a5);a6=_mm256_fmadd_ps(xv,_mm256_set1_ps(w6[kx]),a6);a7=_mm256_fmadd_ps(xv,_mm256_set1_ps(w7[kx]),a7);}
    }
    size_t p=(size_t)oy*width+ox;_mm256_storeu_ps(yo[0]+p,a0);_mm256_storeu_ps(yo[1]+p,a1);_mm256_storeu_ps(yo[2]+p,a2);_mm256_storeu_ps(yo[3]+p,a3);_mm256_storeu_ps(yo[4]+p,a4);_mm256_storeu_ps(yo[5]+p,a5);_mm256_storeu_ps(yo[6]+p,a6);_mm256_storeu_ps(yo[7]+p,a7);ox+=8;
   }else{
    float a[8]={b[og],b[og+1],b[og+2],b[og+3],b[og+4],b[og+5],b[og+6],b[og+7]};
    for(int ic=0;ic<ci;ic++)for(int ky=0;ky<3;ky++){int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;for(int kx=0;kx<3;kx++){int ix=ox+kx-1;if((unsigned)ix>=(unsigned)width)continue;float v=x[((size_t)ic*h+iy)*width+ix];int k=ky*3+kx;for(int q=0;q<8;q++)a[q]+=v*w[((size_t)(og+q)*ci+ic)*9+k];}}
    size_t p=(size_t)oy*width+ox;for(int q=0;q<8;q++)yo[q][p]=a[q];ox++;
   }
  }
 }
}
#endif

void pp_cpu_conv3_plain(const float*x,float*y,const float*w,const float*b,
                        int ci,int co,int h,int wi){
#ifdef PP_WITH_ACCELERATE
 if(!getenv("PP_APPLE_PLAIN_DISABLE")&&
    pp_apple_conv(x,y,w,b,ci,co,h,wi,3,1,1,0))return;
#endif
#ifdef __AVX2__
 static int direct=-1,oc2=-1,oc4=-1,oc8=-1;if(direct<0)direct=getenv("PP_CPU_PLAIN_ACCUM")==NULL;if(oc2<0)oc2=getenv("PP_CPU_PLAIN_OC1")==NULL;if(oc4<0)oc4=getenv("PP_CPU_PLAIN_OC2")==NULL;if(oc8<0)oc8=getenv("PP_CPU_PLAIN_OC4")==NULL;
 if(direct&&oc2&&oc4&&oc8&&!(co&7)){conv3_plain_direct8(x,y,w,b,ci,co,h,wi);return;}
 if(direct&&oc2&&oc4&&!(co&3)){conv3_plain_direct4(x,y,w,b,ci,co,h,wi);return;}
 if(direct&&oc2&&!(co&1)){conv3_plain_direct2(x,y,w,b,ci,co,h,wi);return;}
 if(direct){conv3_plain_direct1(x,y,w,b,ci,co,h,wi);return;}
#endif
 conv3_plain_accum(x,y,w,b,ci,co,h,wi);
}
static void conv2s2_relu(const float*x,float*y,const float*w,const float*b,int ci,int co,int hi,int wi){int h=hi/2,width=wi/2;size_t ni=(size_t)hi*wi,no=(size_t)h*width;
#ifdef PP_WITH_ACCELERATE
 if(pp_apple_conv(x,y,w,b,ci,co,hi,wi,2,2,0,1))return;
#endif
 #pragma omp parallel for schedule(static)
 for(int oc=0;oc<co;oc++){float*yo=y+(size_t)oc*no;for(size_t p=0;p<no;p++)yo[p]=b[oc];for(int ic=0;ic<ci;ic++){const float*xi=x+(size_t)ic*ni;const float*wk=w+((size_t)oc*ci+ic)*4;for(int oy=0;oy<h;oy++)for(int ky=0;ky<2;ky++)for(int kx=0;kx<2;kx++){float a=wk[ky*2+kx];
   #pragma omp simd
   for(int ox=0;ox<width;ox++)yo[(size_t)oy*width+ox]+=xi[(size_t)(oy*2+ky)*wi+ox*2+kx]*a;}}for(size_t p=0;p<no;p++)if(yo[p]<0)yo[p]=0;}
}

/* Transposed convolution [Cin,Cout,K,K], zero padding, stride=K, followed by
 * already-folded BN and ReLU. These deblocks have non-overlapping tiles. */
static void deconv_relu(const float *restrict x, float *restrict y,
                        const float *restrict w, const float *restrict b,
                        int ci, int hi, int wi, int k) {
    const int co = 128, ho = hi * k, wo = wi * k;
#ifdef PP_WITH_ACCELERATE
    if (pp_apple_deconv(x, y, w, b, ci, co, hi, wi, k, k, 1)) return;
#endif
    size_t ni = (size_t)hi * wi, no = (size_t)ho * wo;
    #pragma omp parallel for schedule(static)
    for (int oc = 0; oc < co; ++oc) {
        float *yo = y + (size_t)oc * no;
        for (size_t p = 0; p < no; ++p) yo[p] = b[oc];
        for (int ic = 0; ic < ci; ++ic) {
            const float *xi = x + (size_t)ic * ni;
            const float *wk = w + ((size_t)ic * co + oc) * k * k;
            for (int iy = 0; iy < hi; ++iy) for (int ix = 0; ix < wi; ++ix) {
                float v = xi[(size_t)iy * wi + ix];
                float *tile = yo + (size_t)(iy * k) * wo + ix * k;
                for (int ky = 0; ky < k; ++ky)
                    for (int kx = 0; kx < k; ++kx) tile[(size_t)ky * wo + kx] += v * wk[ky * k + kx];
            }
        }
        #pragma omp simd
        for (size_t p = 0; p < no; ++p) if (yo[p] < 0.0f) yo[p] = 0.0f;
    }
}

static int get2(const pp_model *m, const char *n, uint32_t a, uint32_t b, const float **p) {
    uint32_t d[2] = {a,b}; return (*p = pp_model_tensor(m,n,d,2)) != NULL;
}
static int get1(const pp_model *m, const char *n, uint32_t a, const float **p) {
    uint32_t d[1] = {a}; return (*p = pp_model_tensor(m,n,d,1)) != NULL;
}
static int get4(const pp_model *m, const char *n, uint32_t a,uint32_t b,uint32_t c,uint32_t d0,const float **p) {
    uint32_t d[4] = {a,b,c,d0}; return (*p = pp_model_tensor(m,n,d,4)) != NULL;
}

/* The first BEV convolution sees only live pillars.  Eight output channels
 * share each pillar feature while preserving the dense kernel's ic/ky/kx
 * accumulation order. */
static int sparse_first_conv(const pp_model *m, const pp_pillars *p,
                             const float *pillar, float *output,
                             char *error, size_t cap) {
    const float *weights;
    const float *bias;
    if (!get4(m, "backbone.0.0.weight", 64, 64, 3, 3, &weights) ||
        !get1(m, "backbone.0.0.bias", 64, &bias))
        return fail(error, cap, "invalid tensor backbone.0.0");
    const int height = 256;
    const int width = 256;
    const int pillars = p->pillar_count;
    const size_t plane = (size_t)height * width;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int og = 0; og < 64; og += 8) for (int oy = 0; oy < height; ++oy) {
        float *out[8] = {
            output + (size_t)(og + 0) * plane,
            output + (size_t)(og + 1) * plane,
            output + (size_t)(og + 2) * plane,
            output + (size_t)(og + 3) * plane,
            output + (size_t)(og + 4) * plane,
            output + (size_t)(og + 5) * plane,
            output + (size_t)(og + 6) * plane,
            output + (size_t)(og + 7) * plane
        };
        for (int ox = 0; ox < width; ++ox) {
            int neighbor[9];
            for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
                int iy = oy * 2 + ky - 1;
                int ix = ox * 2 + kx - 1;
                neighbor[ky * 3 + kx] =
                    (unsigned)iy < PP_GRID_Y && (unsigned)ix < PP_GRID_X ?
                    p->grid[(size_t)iy * PP_GRID_X + ix] : -1;
            }
            float a0 = bias[og + 0], a1 = bias[og + 1];
            float a2 = bias[og + 2], a3 = bias[og + 3];
            float a4 = bias[og + 4], a5 = bias[og + 5];
            float a6 = bias[og + 6], a7 = bias[og + 7];
            for (int ic = 0; ic < 64; ++ic) for (int k = 0; k < 9; ++k) {
                int pi = neighbor[k];
                if ((unsigned)pi >= (unsigned)pillars) continue;
                float value = pillar[(size_t)ic * pillars + pi];
                a0 += value * weights[((size_t)(og + 0) * 64 + ic) * 9 + k];
                a1 += value * weights[((size_t)(og + 1) * 64 + ic) * 9 + k];
                a2 += value * weights[((size_t)(og + 2) * 64 + ic) * 9 + k];
                a3 += value * weights[((size_t)(og + 3) * 64 + ic) * 9 + k];
                a4 += value * weights[((size_t)(og + 4) * 64 + ic) * 9 + k];
                a5 += value * weights[((size_t)(og + 5) * 64 + ic) * 9 + k];
                a6 += value * weights[((size_t)(og + 6) * 64 + ic) * 9 + k];
                a7 += value * weights[((size_t)(og + 7) * 64 + ic) * 9 + k];
            }
            size_t index = (size_t)oy * width + ox;
            out[0][index] = a0 > 0.0f ? a0 : 0.0f;
            out[1][index] = a1 > 0.0f ? a1 : 0.0f;
            out[2][index] = a2 > 0.0f ? a2 : 0.0f;
            out[3][index] = a3 > 0.0f ? a3 : 0.0f;
            out[4][index] = a4 > 0.0f ? a4 : 0.0f;
            out[5][index] = a5 > 0.0f ? a5 : 0.0f;
            out[6][index] = a6 > 0.0f ? a6 : 0.0f;
            out[7][index] = a7 > 0.0f ? a7 : 0.0f;
        }
    }
    return 1;
}

static int named3(const pp_model*m,const char*name,const float*x,float*y,int ci,int co,int h,int w,int stride,int relu,char*e,size_t cap){char nw[64],nb[64];const float*wt,*bias;snprintf(nw,sizeof nw,"%s.weight",name);snprintf(nb,sizeof nb,"%s.bias",name);if(!get4(m,nw,co,ci,3,3,&wt)||!get1(m,nb,co,&bias))return fail(e,cap,"invalid tensor %s",name);if(relu)pp_cpu_conv3_relu(x,y,wt,bias,ci,co,h,w,stride);else pp_cpu_conv3_plain(x,y,wt,bias,ci,co,h,w);return 1;}

int pp_infer_cpu(const pp_model*m,const pp_pillars*p,pp_raw_output*out,
                 pp_cpu_workspace*workspace,pp_profile*prof,
                 char*error,size_t cap){
 if(!m||!p||!out||!out->data||!workspace||(!workspace->data&&workspace->floats)||p->pillar_count<0||p->pillar_count>PP_MAX_PILLARS)return fail(error,cap,"invalid inference arguments");
 pp_profile local;if(!prof)prof=&local;memset(prof,0,sizeof(*prof));double begin=now_ms(),t=begin;const float*pw,*pb;
 if(!get2(m,"pfn.weight",64,11,&pw)||!get1(m,"pfn.bias",64,&pb))return fail(error,cap,"invalid PFN weights");
 int dense_first=getenv("PP_CPU_DENSE_FIRST")!=NULL;
 size_t pillarn=(size_t)64*p->pillar_count,scattern=dense_first?(size_t)64*512*512:0;
 size_t stagecap=(size_t)64*256*256,upn=(size_t)128*128*128,concatn=(size_t)384*128*128,sharedn=(size_t)64*128*128;
 size_t required=pillarn+scattern+2*stagecap+concatn+2*sharedn;
 if(workspace->floats<required){float*grown=falloc(required);if(!grown)return fail(error,cap,"workspace allocation failed");free(workspace->data);workspace->data=grown;workspace->floats=required;}
 /* Deblock channels already have concat order: write the three results into
  * consecutive slices and feed the same allocation directly to shared. */
 float*pillar=workspace->data,*scatter=pillar+pillarn,*a=scatter+scattern,*b=a+stagecap,*upall=b+stagecap,*shared=upall+concatn,*mid=shared+sharedn;
 if(dense_first)memset(scatter,0,scattern*sizeof(*scatter));
 float*up0=upall,*up1=upall+upn,*up2=upall+2*upn,*concat=upall;
 prof->workspace_bytes=workspace->floats*sizeof(*workspace->data);
 #pragma omp parallel for schedule(static)
 for(int pi=0;pi<p->pillar_count;pi++)for(int oc=0;oc<64;oc++){float best=0;for(int j=0;j<20;j++){const float*f=p->features+((size_t)pi*20+j)*11;float v=pb[oc];for(int k=0;k<11;k++)v+=pw[(size_t)oc*11+k]*f[k];if(v>best)best=v;}pillar[(size_t)oc*p->pillar_count+pi]=best;}
 prof->pfn_ms=now_ms()-t;t=now_ms();
 if(dense_first){
  #pragma omp parallel for schedule(static)
  for(int oc=0;oc<64;oc++)for(int pi=0;pi<p->pillar_count;pi++){int y=p->coords[(size_t)pi*4+2],x=p->coords[(size_t)pi*4+3];scatter[((size_t)oc*512+y)*512+x]=pillar[(size_t)oc*p->pillar_count+pi];}
 }
 prof->scatter_ms=now_ms()-t;t=now_ms();const int counts[3]={4,6,6},cos[3]={64,128,256};const float*cur=scatter;float*dst=a;int ci=64,h=512,w=512;
 for(int s=0;s<3;s++)for(int l=0;l<counts[s];l++){char n[40];snprintf(n,sizeof n,"backbone.%d.%d",s,l);int stride=l?1:2;if(s==0&&l==0&&!dense_first){if(!sparse_first_conv(m,p,pillar,dst,error,cap))goto bad;}else if(!named3(m,n,cur,dst,ci,cos[s],h,w,stride,1,error,cap))goto bad;h=(h+stride-1)/stride;w=(w+stride-1)/stride;ci=cos[s];cur=dst;dst=dst==a?b:a;if(l==counts[s]-1){char nw[48],nb[48];const float*wt,*bias;snprintf(nw,sizeof nw,"deblock.%d.weight",s);snprintf(nb,sizeof nb,"deblock.%d.bias",s);if(s==0){if(!get4(m,nw,128,64,2,2,&wt)||!get1(m,nb,128,&bias)){fail(error,cap,"invalid deblock 0");goto bad;}conv2s2_relu(cur,up0,wt,bias,64,128,256,256);}else{int k=s==1?1:2,ici=s==1?128:256;float*u=s==1?up1:up2;if(!get4(m,nw,ici,128,k,k,&wt)||!get1(m,nb,128,&bias)){fail(error,cap,"invalid deblock %d",s);goto bad;}deconv_relu(cur,u,wt,bias,ici,h,w,k);}}}
 prof->backbone_ms=now_ms()-t;t=now_ms();if(!named3(m,"shared",concat,shared,384,64,128,128,1,1,error,cap))goto bad;
 static const char*branch[6]={"cls","reg","height","size","angle","velo"};
 for(int hd=0;hd<6;hd++)for(int br=0;br<6;br++){char n[48];snprintf(n,sizeof n,"head.%d.%s.mid",hd,branch[br]);if(!named3(m,n,shared,mid,64,64,128,128,1,1,error,cap))goto bad;snprintf(n,sizeof n,"head.%d.%s.out",hd,branch[br]);int co=pp_branch_channels(hd,br);if(!named3(m,n,mid,pp_output_branch(out,hd,br),64,co,128,128,1,0,error,cap))goto bad;}
 prof->heads_ms=now_ms()-t;prof->total_ms=now_ms()-begin;
#ifdef PP_WITH_ACCELERATE
 prof->workspace_bytes+=pp_apple_cache_bytes();
#endif
 return 1;
bad:return 0;
}
