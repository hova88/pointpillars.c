#define _POSIX_C_SOURCE 200809L
#include "pp_infer.h"

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

typedef struct { int ci, co, h, w, stride; const char *id; } conv_spec;

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
int pp_head_classes(int h){static const int n[6]={1,2,2,1,2,2};return (unsigned)h<6?n[h]:0;}
int pp_branch_channels(int h,int b){static const int mul[6]={0,4,2,6,4,4};int c=pp_head_classes(h);return b==0?2*c*c:(unsigned)b<6?c*mul[b]:0;}
float *pp_output_branch(const pp_raw_output*o,int h,int b){size_t off=0,hw=(size_t)PP_OUT_H*PP_OUT_W;for(int i=0;i<h;i++)for(int j=0;j<6;j++)off+=(size_t)pp_branch_channels(i,j)*hw;for(int j=0;j<b;j++)off+=(size_t)pp_branch_channels(h,j)*hw;return o&&o->data?o->data+off:NULL;}

/* OIHW 3x3 convolution, NCHW, pad=1, fused ReLU. Parallel output channels
 * own disjoint planes. The inner x loop is contiguous and auto-vectorizes. */
static void conv3_relu(const float *restrict x, float *restrict y,
                       const float *restrict w, const float *restrict b,
                       int ci, int co, int hi, int wi, int stride) {
#ifdef __AVX2__
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

static void conv3_plain(const float*x,float*y,const float*w,const float*b,int ci,int co,int h,int wi){size_t n=(size_t)h*wi;
 #pragma omp parallel for schedule(static)
 for(int oc=0;oc<co;oc++){float*yo=y+(size_t)oc*n;for(size_t p=0;p<n;p++)yo[p]=b[oc];for(int ic=0;ic<ci;ic++){const float*xi=x+(size_t)ic*n;const float*wk=w+((size_t)oc*ci+ic)*9;for(int ky=0;ky<3;ky++)for(int oy=0;oy<h;oy++){int iy=oy+ky-1;if((unsigned)iy>=(unsigned)h)continue;for(int kx=0;kx<3;kx++){int first=kx?0:1,last=wi;if(kx==2)last--;float a=wk[ky*3+kx];
   #pragma omp simd
   for(int ox=first;ox<last;ox++)yo[(size_t)oy*wi+ox]+=xi[(size_t)iy*wi+ox+kx-1]*a;}}}}
}
static void conv2s2_relu(const float*x,float*y,const float*w,const float*b,int ci,int co,int hi,int wi){int h=hi/2,width=wi/2;size_t ni=(size_t)hi*wi,no=(size_t)h*width;
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

#if 0
static int run_conv(const pp_model *m, const conv_spec *s, const float *x, float *y,
                    char *error, size_t cap) {
    char nw[48], nb[48]; snprintf(nw,sizeof nw,"conv.%s.weight",s->id); snprintf(nb,sizeof nb,"conv.%s.bias",s->id);
    const float *w, *b;
    if (!get4(m,nw,s->co,s->ci,3,3,&w) || !get1(m,nb,s->co,&b))
        return fail(error,cap,"missing or invalid %s",s->id);
    conv3_relu(x,y,w,b,s->ci,s->co,s->h,s->w,s->stride); return 1;
}

static int run_deconv(const pp_model *m, int id, const float *x, float *y,
                      int ci,int h,int w,int k,char *error,size_t cap) {
    char nw[48],nb[48]; snprintf(nw,sizeof nw,"deconv.%d.weight",id); snprintf(nb,sizeof nb,"deconv.%d.bias",id);
    const float *wt,*bias;
    if (!get4(m,nw,ci,128,k,k,&wt)||!get1(m,nb,128,&bias)) return fail(error,cap,"invalid deconv %d",id);
    deconv_relu(x,y,wt,bias,ci,h,w,k); return 1;
}

static void head3(const float *a,const float *b,const float *c,float *y,
                  const float *w,const float *bias,int co) {
    size_t hw=(size_t)PP_OUT_H*PP_OUT_W;
    #pragma omp parallel for schedule(static)
    for(int oc=0;oc<co;++oc){
        float *yo=y+(size_t)oc*hw;
        for(size_t pb=0;pb<hw;pb+=1024){
          size_t pe=pb+1024<hw?pb+1024:hw;
          #pragma omp simd
          for(size_t p=pb;p<pe;++p)yo[p]=bias[oc];
          for(int ic=0;ic<384;++ic){
            const float *xi=ic<128?a+(size_t)ic*hw:ic<256?b+(size_t)(ic-128)*hw:c+(size_t)(ic-256)*hw;
            float q=w[(size_t)oc*384+ic];
            #pragma omp simd
            for(size_t p=pb;p<pe;++p) yo[p]+=xi[p]*q;
          }
        }
    }
}

int pp_infer_cpu(const pp_model *m,const pp_pillars *p,pp_raw_output *out,
                 pp_profile *prof,char *error,size_t cap) {
    if(!m||!p||!out||!out->cls||!out->box||!out->dir) return fail(error,cap,"invalid inference arguments");
    pp_profile local_profile;
    if (!prof) prof = &local_profile;
    memset(prof,0,sizeof(*prof)); double t=now_ms();
    const float *pw,*pb;
    if(!get2(m,"pfn.weight",64,10,&pw)||!get1(m,"pfn.bias",64,&pb)) return fail(error,cap,"invalid PFN weights");
    float *pillar=falloc((size_t)64*p->pillar_count);
    float *scatter=(float*)calloc((size_t)64*PP_GRID_Y*PP_GRID_X,sizeof(float));
    size_t maxstage=(size_t)64*PP_OUT_H*PP_OUT_W;
    float *stage0=falloc(maxstage),*stage1=falloc(maxstage);
    size_t upn=(size_t)128*PP_OUT_H*PP_OUT_W;
    float *up0=falloc(upn),*up1=falloc(upn),*up2=falloc(upn);
    if(!pillar||!scatter||!stage0||!stage1||!up0||!up1||!up2){
        free(pillar);free(scatter);free(stage0);free(stage1);free(up0);free(up1);free(up2);
        return fail(error,cap,"workspace allocation failed");
    }
    prof->workspace_bytes=((size_t)64*p->pillar_count+(size_t)64*PP_GRID_Y*PP_GRID_X+2*maxstage+3*upn)*sizeof(float);
    #pragma omp parallel for schedule(static)
    for(int pi=0;pi<p->pillar_count;++pi) for(int oc=0;oc<64;++oc){
        float best=0.0f;
        for(int j=0;j<PP_MAX_POINTS;++j){
            const float *f=p->features+((size_t)pi*PP_MAX_POINTS+j)*10;
            float v=pb[oc]; for(int k=0;k<10;++k)v+=pw[(size_t)oc*10+k]*f[k];
            if(v>best)best=v;
        }
        pillar[(size_t)oc*p->pillar_count+pi]=best;
    }
    prof->pfn_ms=now_ms()-t;t=now_ms();
    #pragma omp parallel for schedule(static)
    for(int oc=0;oc<64;++oc) for(int pi=0;pi<p->pillar_count;++pi){
        int y=p->coords[(size_t)pi*4+2],x=p->coords[(size_t)pi*4+3];
        scatter[((size_t)oc*PP_GRID_Y+y)*PP_GRID_X+x]=pillar[(size_t)oc*p->pillar_count+pi];
    }
    prof->scatter_ms=now_ms()-t;t=now_ms();
    const conv_spec specs[]={
      {64,64,496,432,2,"702"},{64,64,248,216,1,"705"},{64,64,248,216,1,"708"},{64,64,248,216,1,"711"},
      {64,128,248,216,2,"714"},{128,128,124,108,1,"717"},{128,128,124,108,1,"720"},{128,128,124,108,1,"723"},{128,128,124,108,1,"726"},{128,128,124,108,1,"729"},
      {128,256,124,108,2,"732"},{256,256,62,54,1,"735"},{256,256,62,54,1,"738"},{256,256,62,54,1,"741"},{256,256,62,54,1,"744"},{256,256,62,54,1,"747"}};
    const float *cur=scatter;float *dst=stage0;
    for(int i=0;i<16;++i){
        if(!run_conv(m,&specs[i],cur,dst,error,cap))goto bad;
        cur=dst;dst=(dst==stage0)?stage1:stage0;
        if(i==3&&!run_deconv(m,0,cur,up0,64,248,216,1,error,cap))goto bad;
        if(i==9&&!run_deconv(m,1,cur,up1,128,124,108,2,error,cap))goto bad;
        if(i==15&&!run_deconv(m,2,cur,up2,256,62,54,4,error,cap))goto bad;
    }
    prof->backbone_ms=now_ms()-t;t=now_ms();
    struct H{const char*n;int co;float*y;} hs[]={{"head.cls",18,out->cls},{"head.box",42,out->box},{"head.dir",12,out->dir}};
    for(int i=0;i<3;++i){char nw[48],nb[48];const float *w,*b;snprintf(nw,sizeof nw,"%s.weight",hs[i].n);snprintf(nb,sizeof nb,"%s.bias",hs[i].n);
      if(!get4(m,nw,hs[i].co,384,1,1,&w)||!get1(m,nb,hs[i].co,&b)){fail(error,cap,"invalid %s",hs[i].n);goto bad;}
      head3(up0,up1,up2,hs[i].y,w,b,hs[i].co);
    }
    prof->heads_ms=now_ms()-t;
    prof->total_ms=prof->pfn_ms+prof->scatter_ms+prof->backbone_ms+prof->heads_ms;
    free(pillar);free(scatter);free(stage0);free(stage1);free(up0);free(up1);free(up2);return 1;
bad:
    free(pillar);free(scatter);free(stage0);free(stage1);free(up0);free(up1);free(up2);return 0;
}
#endif

static int named3(const pp_model*m,const char*name,const float*x,float*y,int ci,int co,int h,int w,int stride,int relu,char*e,size_t cap){char nw[64],nb[64];const float*wt,*bias;snprintf(nw,sizeof nw,"%s.weight",name);snprintf(nb,sizeof nb,"%s.bias",name);if(!get4(m,nw,co,ci,3,3,&wt)||!get1(m,nb,co,&bias))return fail(e,cap,"invalid tensor %s",name);if(relu)conv3_relu(x,y,wt,bias,ci,co,h,w,stride);else conv3_plain(x,y,wt,bias,ci,co,h,w);return 1;}

int pp_infer_cpu(const pp_model*m,const pp_pillars*p,pp_raw_output*out,pp_profile*prof,char*error,size_t cap){
 if(!m||!p||!out||!out->data)return fail(error,cap,"invalid inference arguments");
 pp_profile local;if(!prof)prof=&local;memset(prof,0,sizeof(*prof));double begin=now_ms(),t=begin;const float*pw,*pb;
 if(!get2(m,"pfn.weight",64,11,&pw)||!get1(m,"pfn.bias",64,&pb))return fail(error,cap,"invalid PFN weights");
 size_t stagecap=(size_t)64*256*256,upn=(size_t)128*128*128,concatn=(size_t)384*128*128,sharedn=(size_t)64*128*128;
 float*pillar=falloc((size_t)64*p->pillar_count),*scatter=(float*)calloc((size_t)64*512*512,4),*a=falloc(stagecap),*b=falloc(stagecap),*up0=falloc(upn),*up1=falloc(upn),*up2=falloc(upn),*concat=falloc(concatn),*shared=falloc(sharedn),*mid=falloc(sharedn);
 if(!pillar||!scatter||!a||!b||!up0||!up1||!up2||!concat||!shared||!mid){fail(error,cap,"workspace allocation failed");goto bad;}
 prof->workspace_bytes=((size_t)64*p->pillar_count+(size_t)64*512*512+2*stagecap+3*upn+concatn+2*sharedn)*4;
 #pragma omp parallel for schedule(static)
 for(int pi=0;pi<p->pillar_count;pi++)for(int oc=0;oc<64;oc++){float best=0;for(int j=0;j<20;j++){const float*f=p->features+((size_t)pi*20+j)*11;float v=pb[oc];for(int k=0;k<11;k++)v+=pw[(size_t)oc*11+k]*f[k];if(v>best)best=v;}pillar[(size_t)oc*p->pillar_count+pi]=best;}
 prof->pfn_ms=now_ms()-t;t=now_ms();
 #pragma omp parallel for schedule(static)
 for(int oc=0;oc<64;oc++)for(int pi=0;pi<p->pillar_count;pi++){int y=p->coords[(size_t)pi*4+2],x=p->coords[(size_t)pi*4+3];scatter[((size_t)oc*512+y)*512+x]=pillar[(size_t)oc*p->pillar_count+pi];}
 prof->scatter_ms=now_ms()-t;t=now_ms();const int counts[3]={4,6,6},cos[3]={64,128,256};const float*cur=scatter;float*dst=a;int ci=64,h=512,w=512;
 for(int s=0;s<3;s++)for(int l=0;l<counts[s];l++){char n[40];snprintf(n,sizeof n,"backbone.%d.%d",s,l);int stride=l?1:2;if(!named3(m,n,cur,dst,ci,cos[s],h,w,stride,1,error,cap))goto bad;h=(h+stride-1)/stride;w=(w+stride-1)/stride;ci=cos[s];cur=dst;dst=dst==a?b:a;if(l==counts[s]-1){char nw[48],nb[48];const float*wt,*bias;snprintf(nw,sizeof nw,"deblock.%d.weight",s);snprintf(nb,sizeof nb,"deblock.%d.bias",s);if(s==0){if(!get4(m,nw,128,64,2,2,&wt)||!get1(m,nb,128,&bias)){fail(error,cap,"invalid deblock 0");goto bad;}conv2s2_relu(cur,up0,wt,bias,64,128,256,256);}else{int k=s==1?1:2,ici=s==1?128:256;float*u=s==1?up1:up2;if(!get4(m,nw,ici,128,k,k,&wt)||!get1(m,nb,128,&bias)){fail(error,cap,"invalid deblock %d",s);goto bad;}deconv_relu(cur,u,wt,bias,ici,h,w,k);}}}
 prof->backbone_ms=now_ms()-t;t=now_ms();memcpy(concat,up0,upn*4);memcpy(concat+upn,up1,upn*4);memcpy(concat+2*upn,up2,upn*4);if(!named3(m,"shared",concat,shared,384,64,128,128,1,1,error,cap))goto bad;
 static const char*branch[6]={"cls","reg","height","size","angle","velo"};
 for(int hd=0;hd<6;hd++)for(int br=0;br<6;br++){char n[48];snprintf(n,sizeof n,"head.%d.%s.mid",hd,branch[br]);if(!named3(m,n,shared,mid,64,64,128,128,1,1,error,cap))goto bad;snprintf(n,sizeof n,"head.%d.%s.out",hd,branch[br]);int co=pp_branch_channels(hd,br);if(!named3(m,n,mid,pp_output_branch(out,hd,br),64,co,128,128,1,0,error,cap))goto bad;}
 prof->heads_ms=now_ms()-t;prof->total_ms=now_ms()-begin;free(pillar);free(scatter);free(a);free(b);free(up0);free(up1);free(up2);free(concat);free(shared);free(mid);return 1;
bad:free(pillar);free(scatter);free(a);free(b);free(up0);free(up1);free(up2);free(concat);free(shared);free(mid);return 0;
}
