#define _POSIX_C_SOURCE 200809L
#include "pointpillars.h"
#include "pp_infer.h"
#include "pp_model.h"
#include "pp_decode.h"
#include "pp_tui.h"
#ifdef PP_WITH_CUDA
#include "pp_cuda.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include <pthread.h>

static int write_outputs(const char *path, const pp_raw_output *o) {
    FILE *f=fopen(path,"wb"); if(!f)return 0;
    const char magic[8]={'P','P','O','U','T',0,0,2};
    int ok=fwrite(magic,1,8,f)==8&&fwrite(o->data,sizeof(float),o->floats,f)==o->floats;
    ok &= fclose(f)==0; return ok;
}
static int write_detections(const char *path,const pp_detections*d){
 FILE*f=fopen(path,"w");if(!f)return 0;fputs("[\n",f);
 for(size_t i=0;i<d->count;i++){const pp_box*b=d->boxes+i;fprintf(f,"  {\"class_id\":%d,\"score\":%.9g,\"x\":%.9g,\"y\":%.9g,\"z\":%.9g,\"dx\":%.9g,\"dy\":%.9g,\"dz\":%.9g,\"yaw\":%.9g,\"vx\":%.9g,\"vy\":%.9g}%s\n",b->class_id,b->score,b->x,b->y,b->z,b->dx,b->dy,b->dz,b->yaw,b->vx,b->vy,i+1<d->count?",":"");}
 fputs("]\n",f);return fclose(f)==0;
}

static void usage(const char *p) {
    fprintf(stderr,
            "usage: %s MODE MODEL.ppw INPUT [OUTPUT] [options]\n"
            "modes: inspect, infer, infer-cuda, bench, bench-cuda, "
            "bench-detect-cuda, batch, batch-cuda, tui, tui-cuda\n",
            p);
}
static int compare_names(const void*a,const void*b){return strcmp(*(char*const*)a,*(char*const*)b);}
static double monotonic_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}
static const char *tui_backend_name(int use_cuda) {
    if (!use_cuda) {
#ifdef PP_WITH_ACCELERATE
        return getenv("PP_APPLE_DISABLE") ? "CPU scalar/SIMD" : "Apple Accelerate/BNNS";
#elif defined(PP_WITH_GGML)
        return getenv("PP_GGML_DISABLE") ? "CPU AVX2" : "CPU + GGML";
#else
        return "CPU AVX2";
#endif
    }
#ifdef PP_WITH_CUDNN
    if (!getenv("PP_CUDNN_DISABLE"))
        return getenv("PP_CUDNN_TF32") ? "cuDNN TF32" : "cuDNN FP32";
#endif
    return getenv("PP_CUDA_PRECISE") ? "CUDA FP32" : "CUDA WMMA";
}

typedef struct {pp_pillars pillars;pp_voxel_stats stats;size_t index;double prep_ms;int state;char error[256];} prep_slot;
typedef struct {char **names;size_t count,next;int stop,started,depth;pthread_t thread;pthread_mutex_t lock;pthread_cond_t changed;prep_slot slot[2];} prep_pipe;

static void *prepare_frames(void *arg){prep_pipe*p=arg;for(;;){pthread_mutex_lock(&p->lock);if(p->stop||p->next==p->count){pthread_mutex_unlock(&p->lock);return NULL;}size_t i=p->next++;prep_slot*s=&p->slot[i%(size_t)p->depth];while(s->state&&!p->stop)pthread_cond_wait(&p->changed,&p->lock);if(p->stop){pthread_mutex_unlock(&p->lock);return NULL;}s->state=1;s->index=i;pthread_mutex_unlock(&p->lock);
 float*points=NULL;size_t n=0;double begin=monotonic_ms();int ok=pp_load_points(p->names[i],5,&points,&n,s->error,sizeof s->error)&&pp_voxelize(points,n,5,&s->pillars,&s->stats);free(points);s->prep_ms=monotonic_ms()-begin;
 pthread_mutex_lock(&p->lock);s->state=ok?2:3;pthread_cond_broadcast(&p->changed);pthread_mutex_unlock(&p->lock);if(!ok)return NULL;}}
static int prep_pipe_start(prep_pipe*p,char**names,size_t count,int depth){memset(p,0,sizeof(*p));p->names=names;p->count=count;p->depth=depth;if(pthread_mutex_init(&p->lock,NULL))return 0;if(pthread_cond_init(&p->changed,NULL)){pthread_mutex_destroy(&p->lock);return 0;}for(int i=0;i<depth;i++)if(!pp_pillars_alloc(&p->slot[i].pillars))goto fail;if(pthread_create(&p->thread,NULL,prepare_frames,p))goto fail;p->started=1;return 1;
fail:pp_pillars_free(&p->slot[0].pillars);pp_pillars_free(&p->slot[1].pillars);pthread_cond_destroy(&p->changed);pthread_mutex_destroy(&p->lock);return 0;}
static prep_slot *prep_pipe_get(prep_pipe*p,size_t i){prep_slot*s=&p->slot[i%(size_t)p->depth];pthread_mutex_lock(&p->lock);while(s->state<2)pthread_cond_wait(&p->changed,&p->lock);pthread_mutex_unlock(&p->lock);return s;}
static void prep_pipe_release(prep_pipe*p,prep_slot*s){pthread_mutex_lock(&p->lock);s->state=0;pthread_cond_broadcast(&p->changed);pthread_mutex_unlock(&p->lock);}
static void prep_pipe_end(prep_pipe*p){if(p->started){pthread_mutex_lock(&p->lock);p->stop=1;pthread_cond_broadcast(&p->changed);pthread_mutex_unlock(&p->lock);pthread_join(p->thread,NULL);}pp_pillars_free(&p->slot[0].pillars);pp_pillars_free(&p->slot[1].pillars);pthread_cond_destroy(&p->changed);pthread_mutex_destroy(&p->lock);}

int main(int argc,char **argv){
    if(argc<3){usage(argv[0]);return 2;}
    pp_model m;char error[256]={0};
    if(!pp_model_open(&m,argv[2],error,sizeof error)){fprintf(stderr,"error: %s\n",error);return 1;}
    if(!strcmp(argv[1],"inspect")){
        printf("PointPillars model: %u tensors, %.2f MiB\n",m.count,m.mapping_bytes/1048576.0);
        for(uint32_t i=0;i<m.count;++i){const pp_weight_record*r=m.records+i;printf("%-32s [",r->name);for(uint32_t j=0;j<r->rank;++j)printf("%s%u",j?",":"",r->dim[j]);printf("]\n");}
        pp_model_close(&m);return 0;
    }
    int is_tui=!strcmp(argv[1],"tui")||!strcmp(argv[1],"tui-cuda");
    int is_batch=!strcmp(argv[1],"batch")||!strcmp(argv[1],"batch-cuda");
    int compact_bench=!strcmp(argv[1],"bench-detect-cuda");
    int is_bench=!strcmp(argv[1],"bench")||!strcmp(argv[1],"bench-cuda")||compact_bench;
    int use_cuda=!strcmp(argv[1],"infer-cuda")||!strcmp(argv[1],"tui-cuda")||!strcmp(argv[1],"bench-cuda")||compact_bench||!strcmp(argv[1],"batch-cuda");
    int compact_cuda=use_cuda&&!getenv("PP_CUDA_RAW_DECODE");
    if((strcmp(argv[1],"infer")&&!use_cuda&&!is_tui&&!is_bench&&!is_batch)||argc<4||(is_batch&&argc<5)){usage(argv[0]);pp_model_close(&m);return 2;}
#ifndef PP_WITH_CUDA
    if(use_cuda){fprintf(stderr,"error: this binary was built without CUDA\n");pp_model_close(&m);return 1;}
#endif
    if(is_tui||is_batch){int directory_ok=1;
      if(is_batch&&mkdir(argv[4],0755)&&errno!=EEXIST){perror(argv[4]);pp_model_close(&m);return 1;}
      DIR*d=opendir(argv[3]);if(!d){perror(argv[3]);pp_model_close(&m);return 1;}size_t capn=256,nf=0;char**names=malloc(capn*sizeof(*names));struct dirent*de;if(!names){closedir(d);pp_model_close(&m);return 1;}
      while((de=readdir(d))){
        size_t ln=strlen(de->d_name);if(ln>=4&&!strcmp(de->d_name+ln-4,".bin")){if(nf==capn){capn*=2;char**grown=realloc(names,capn*sizeof(*names));if(!grown){closedir(d);for(size_t i=0;i<nf;i++)free(names[i]);free(names);pp_model_close(&m);return 1;}names=grown;}size_t z=strlen(argv[3])+ln+2;names[nf]=malloc(z);if(!names[nf]){closedir(d);for(size_t i=0;i<nf;i++)free(names[i]);free(names);pp_model_close(&m);return 1;}snprintf(names[nf++],z,"%s/%s",argv[3],de->d_name);}
      }
      closedir(d);
      qsort(names,nf,sizeof(*names),compare_names);
      if(!nf){fprintf(stderr,"error: no .bin point files in %s\n",argv[3]);free(names);pp_model_close(&m);return 1;}
      pp_raw_output ro={0};if(!compact_cuda&&!pp_output_alloc(&ro)){pp_model_close(&m);return 1;}
      if(is_tui){
       pp_tui_state ui;if(!pp_tui_begin(&ui)){fprintf(stderr,"error: TUI requires an interactive terminal\n");pp_output_free(&ro);for(size_t i=0;i<nf;i++)free(names[i]);free(names);pp_model_close(&m);return 1;}
       pp_pillars vp={0};pp_detections dd={0};
       if(!pp_pillars_alloc(&vp)||!pp_detections_alloc(&dd,1000)){pp_tui_end();pp_pillars_free(&vp);pp_detections_free(&dd);pp_output_free(&ro);for(size_t i=0;i<nf;i++)free(names[i]);free(names);pp_model_close(&m);return 1;}
       size_t fi=0;int running=nf>0;
       while(running){float*qp=NULL;size_t qn=0;pp_voxel_stats vs;pp_profile pr;
        if(!pp_load_points(names[fi],5,&qp,&qn,error,sizeof error)||!pp_voxelize(qp,qn,5,&vp,&vs)){directory_ok=0;running=0;free(qp);break;}
        struct timespec ta,tb;clock_gettime(CLOCK_MONOTONIC,&ta);
#ifdef PP_WITH_CUDA
        int good=use_cuda?(compact_cuda?pp_infer_cuda_detect(&m,&vp,&dd,.1f,.2f,&pr,error,sizeof error):pp_infer_cuda(&m,&vp,&ro,&pr,error,sizeof error)):pp_infer_cpu(&m,&vp,&ro,&pr,error,sizeof error);
#else
        int good=pp_infer_cpu(&m,&vp,&ro,&pr,error,sizeof error);
#endif
        clock_gettime(CLOCK_MONOTONIC,&tb);double elapsed=(tb.tv_sec-ta.tv_sec)*1e3+(tb.tv_nsec-ta.tv_nsec)/1e6;if(good&&!compact_cuda)good=pp_decode(&ro,.1f,.2f,&dd);
        if(!good){directory_ok=0;running=0;free(qp);break;}
        pp_tui_update_tracks(&ui,&dd,fi);
        const char *backend=tui_backend_name(use_cuda);
        pp_tui_render(qp,qn,5,&dd,fi,nf,elapsed,backend,&ui);
        int action;
        for(;;){action=pp_tui_poll(&ui,ui.paused?-1:100);if(action==PP_TUI_REDRAW){pp_tui_render(qp,qn,5,&dd,fi,nf,elapsed,backend,&ui);continue;}if(action==PP_TUI_NONE&&!ui.paused)action=PP_TUI_NEXT;if(action!=PP_TUI_NONE)break;}
        free(qp);
        if(action==PP_TUI_QUIT)running=0;else if(action==PP_TUI_PREV)fi=fi?fi-1:nf-1;else fi=(fi+1)%nf;
       }
       pp_tui_end();pp_pillars_free(&vp);pp_detections_free(&dd);if(!running&&error[0])fprintf(stderr,"error: %s\n",error);
      } else {prep_pipe pipe;pp_detections dd={0};int depth=getenv("PP_NO_PREFETCH")?1:2,pipeline=prep_pipe_start(&pipe,names,nf,depth);
       if(!pipeline||!pp_detections_alloc(&dd,1000))directory_ok=0;
       else for(size_t fi=0;fi<nf;fi++){pp_profile pr;prep_slot*s=prep_pipe_get(&pipe,fi);if(s->state==3){fprintf(stderr,"error: %s\n",s->error);directory_ok=0;break;}
        struct timespec ta,tb;clock_gettime(CLOCK_MONOTONIC,&ta);
#ifdef PP_WITH_CUDA
        int good=use_cuda?(compact_cuda?pp_infer_cuda_detect(&m,&s->pillars,&dd,.1f,.2f,&pr,error,sizeof error):pp_infer_cuda(&m,&s->pillars,&ro,&pr,error,sizeof error)):pp_infer_cpu(&m,&s->pillars,&ro,&pr,error,sizeof error);
#else
        int good=pp_infer_cpu(&m,&s->pillars,&ro,&pr,error,sizeof error);
#endif
        clock_gettime(CLOCK_MONOTONIC,&tb);double elapsed=(tb.tv_sec-ta.tv_sec)*1e3+(tb.tv_nsec-ta.tv_nsec)/1e6,decode0=monotonic_ms();if(good&&!compact_cuda)good=pp_decode(&ro,.1f,.2f,&dd);double decode_ms=compact_cuda?0:monotonic_ms()-decode0;if(!good){fprintf(stderr,"%s\n",error);directory_ok=0;prep_pipe_release(&pipe,s);break;}const char*base=strrchr(names[fi],'/');base=base?base+1:names[fi];char path[2048];snprintf(path,sizeof path,"%s/%s.json",argv[4],base);good=write_detections(path,&dd);fprintf(stderr,"[%zu/%zu] %s: %zu boxes infer %.2f ms prep %.2f ms decode %.2f ms d2h %.1f KiB\n",fi+1,nf,base,dd.count,elapsed,s->prep_ms,decode_ms,pr.device_to_host_bytes/1024.0);prep_pipe_release(&pipe,s);if(!good){directory_ok=0;break;}}
       pp_detections_free(&dd);if(pipeline)prep_pipe_end(&pipe);}
      pp_output_free(&ro);for(size_t i=0;i<nf;i++)free(names[i]);free(names);pp_model_close(&m);return directory_ok?0:1;
    }
    size_t stride=argc>5?(size_t)strtoul(argv[5],NULL,10):5,count=0;float *points=NULL;
    if(!pp_load_points(argv[3],stride,&points,&count,error,sizeof error)){fprintf(stderr,"error: %s\n",error);pp_model_close(&m);return 1;}
    pp_pillars p;pp_voxel_stats vs;pp_raw_output out={0};pp_profile pr;pp_detections det={0};
    if(!pp_pillars_alloc(&p)||(!compact_bench&&!pp_output_alloc(&out))||!pp_voxelize(points,count,stride,&p,&vs)||
       (compact_bench&&!pp_detections_alloc(&det,1000))){
      fprintf(stderr,"error: allocation/voxelization failed\n");free(points);pp_model_close(&m);return 1;}
    fprintf(stderr,"points=%zu accepted=%llu pillars=%d clipped=%llu dropped=%llu\n",count,
      (unsigned long long)vs.accepted_points,p.pillar_count,(unsigned long long)vs.clipped_points,(unsigned long long)vs.dropped_pillars);
    int ok;
    if(is_bench){int reps=argc>4?atoi(argv[4]):5;if(reps<1)reps=1;double sum=0;
      for(int k=0;k<reps;k++){
#ifdef PP_WITH_CUDA
        ok=compact_bench?pp_infer_cuda_detect(&m,&p,&det,.1f,.2f,&pr,error,sizeof error):
           (use_cuda?pp_infer_cuda(&m,&p,&out,&pr,error,sizeof error):pp_infer_cpu(&m,&p,&out,&pr,error,sizeof error));
#else
        ok=pp_infer_cpu(&m,&p,&out,&pr,error,sizeof error);
#endif
        if(!ok){fprintf(stderr,"error: %s\n",error);break;}fprintf(stderr,"run %d total %.3f ms (pfn %.3f, scatter %.3f, backbone %.3f, heads %.3f, workspace %zu, d2h %zu)\n",k,pr.total_ms,pr.pfn_ms,pr.scatter_ms,pr.backbone_ms,pr.heads_ms,pr.workspace_bytes,pr.device_to_host_bytes);if(k)sum+=pr.total_ms;
      }
      if(ok&&reps>1)fprintf(stderr,"warm mean %.2f ms over %d runs\n",sum/(reps-1),reps-1);
      pp_detections_free(&det);pp_output_free(&out);pp_pillars_free(&p);free(points);pp_model_close(&m);return ok?0:1;
    }
#ifdef PP_WITH_CUDA
    ok=use_cuda?pp_infer_cuda(&m,&p,&out,&pr,error,sizeof error):pp_infer_cpu(&m,&p,&out,&pr,error,sizeof error);
#else
    ok=pp_infer_cpu(&m,&p,&out,&pr,error,sizeof error);
#endif
    if(ok&&pp_detections_alloc(&det,1000)){ok=pp_decode(&out,.1f,.2f,&det);fprintf(stderr,"detections=%zu\n",det.count);
      for(size_t i=0;i<det.count&&i<10;++i){pp_box*b=det.boxes+i;fprintf(stderr,"  %zu class=%d score=%.3f xyz=(%.2f %.2f %.2f) size=(%.2f %.2f %.2f) yaw=%.2f\n",i,b->class_id,b->score,b->x,b->y,b->z,b->dx,b->dy,b->dz,b->yaw);}if(ok&&argc>6)ok=write_detections(argv[6],&det);pp_detections_free(&det);}
    if(ok&&argc>4)ok=write_outputs(argv[4],&out);
    if(ok)fprintf(stderr,"pfn %.2f ms | scatter %.2f ms | backbone %.2f ms | heads %.2f ms | total %.2f ms | workspace %.1f MiB\n",
      pr.pfn_ms,pr.scatter_ms,pr.backbone_ms,pr.heads_ms,pr.total_ms,pr.workspace_bytes/1048576.0);
    else fprintf(stderr,"error: %s\n",error);
    pp_output_free(&out);pp_pillars_free(&p);free(points);pp_model_close(&m);return ok?0:1;
}
