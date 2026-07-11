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
    fprintf(stderr,"usage: %s inspect MODEL.ppw\n       %s infer MODEL.ppw POINTS.bin [OUTPUT.ppout] [stride=5]\n",p,p);
}
static int compare_names(const void*a,const void*b){return strcmp(*(char*const*)a,*(char*const*)b);}

int main(int argc,char **argv){
    if(argc<3){usage(argv[0]);return 2;}
    pp_model m;char error[256];
    if(!pp_model_open(&m,argv[2],error,sizeof error)){fprintf(stderr,"error: %s\n",error);return 1;}
    if(!strcmp(argv[1],"inspect")){
        printf("PointPillars model: %u tensors, %.2f MiB\n",m.count,m.mapping_bytes/1048576.0);
        for(uint32_t i=0;i<m.count;++i){const pp_weight_record*r=m.records+i;printf("%-32s [",r->name);for(uint32_t j=0;j<r->rank;++j)printf("%s%u",j?",":"",r->dim[j]);printf("]\n");}
        pp_model_close(&m);return 0;
    }
    int is_tui=!strcmp(argv[1],"tui")||!strcmp(argv[1],"tui-cuda");
    int is_batch=!strcmp(argv[1],"batch")||!strcmp(argv[1],"batch-cuda");
    int is_bench=!strcmp(argv[1],"bench")||!strcmp(argv[1],"bench-cuda");
    int use_cuda=!strcmp(argv[1],"infer-cuda")||!strcmp(argv[1],"tui-cuda")||!strcmp(argv[1],"bench-cuda")||!strcmp(argv[1],"batch-cuda");
    if((strcmp(argv[1],"infer")&&!use_cuda&&!is_tui&&!is_bench&&!is_batch)||argc<4||(is_batch&&argc<5)){usage(argv[0]);pp_model_close(&m);return 2;}
#ifndef PP_WITH_CUDA
    if(use_cuda){fprintf(stderr,"error: this binary was built without CUDA\n");pp_model_close(&m);return 1;}
#endif
    if(is_tui||is_batch){
      if(is_batch&&mkdir(argv[4],0755)&&errno!=EEXIST){perror(argv[4]);pp_model_close(&m);return 1;}
      DIR*d=opendir(argv[3]);if(!d){perror(argv[3]);pp_model_close(&m);return 1;}size_t capn=256,nf=0;char**names=malloc(capn*sizeof(*names));struct dirent*de;
      while((de=readdir(d))){
        if(strstr(de->d_name,".bin")){if(nf==capn){capn*=2;names=realloc(names,capn*sizeof(*names));}size_t z=strlen(argv[3])+strlen(de->d_name)+2;names[nf]=malloc(z);snprintf(names[nf++],z,"%s/%s",argv[3],de->d_name);}
      }
      closedir(d);
      qsort(names,nf,sizeof(*names),compare_names);
      pp_raw_output ro;if(!pp_output_alloc(&ro)){pp_model_close(&m);return 1;}for(size_t fi=0;fi<nf;fi++){float*qp=NULL;size_t qn=0;pp_pillars vp;pp_voxel_stats vs;pp_profile pr;pp_detections dd;
        if(!pp_load_points(names[fi],5,&qp,&qn,error,sizeof error)||!pp_pillars_alloc(&vp)||!pp_voxelize(qp,qn,5,&vp,&vs)||!pp_detections_alloc(&dd,1000))break;
        struct timespec ta,tb;clock_gettime(CLOCK_MONOTONIC,&ta);
#ifdef PP_WITH_CUDA
        int good=use_cuda?pp_infer_cuda(&m,&vp,&ro,&pr,error,sizeof error):pp_infer_cpu(&m,&vp,&ro,&pr,error,sizeof error);
#else
        int good=pp_infer_cpu(&m,&vp,&ro,&pr,error,sizeof error);
#endif
        clock_gettime(CLOCK_MONOTONIC,&tb);double elapsed=(tb.tv_sec-ta.tv_sec)*1e3+(tb.tv_nsec-ta.tv_nsec)/1e6;if(good)good=pp_decode(&ro,.1f,.2f,&dd);if(!good){fprintf(stderr,"%s\n",error);break;}if(is_tui)pp_tui_render(qp,qn,5,&dd,fi,elapsed,use_cuda?"CUDA":"CPU");else{const char*base=strrchr(names[fi],'/');base=base?base+1:names[fi];char path[2048];snprintf(path,sizeof path,"%s/%s.json",argv[4],base);good=write_detections(path,&dd);fprintf(stderr,"[%zu/%zu] %s: %zu boxes %.2f ms\n",fi+1,nf,base,dd.count,elapsed);}pp_detections_free(&dd);pp_pillars_free(&vp);free(qp);if(!good)break;}
      if(is_tui)printf("\033[?25h\033[0m\n");
      pp_output_free(&ro);for(size_t i=0;i<nf;i++)free(names[i]);free(names);pp_model_close(&m);return 0;
    }
    size_t stride=argc>5?(size_t)strtoul(argv[5],NULL,10):5,count=0;float *points=NULL;
    if(!pp_load_points(argv[3],stride,&points,&count,error,sizeof error)){fprintf(stderr,"error: %s\n",error);pp_model_close(&m);return 1;}
    pp_pillars p;pp_voxel_stats vs;pp_raw_output out;pp_profile pr;pp_detections det;
    if(!pp_pillars_alloc(&p)||!pp_output_alloc(&out)||!pp_voxelize(points,count,stride,&p,&vs)){
      fprintf(stderr,"error: allocation/voxelization failed\n");free(points);pp_model_close(&m);return 1;}
    fprintf(stderr,"points=%zu accepted=%llu pillars=%d clipped=%llu dropped=%llu\n",count,
      (unsigned long long)vs.accepted_points,p.pillar_count,(unsigned long long)vs.clipped_points,(unsigned long long)vs.dropped_pillars);
    int ok;
    if(is_bench){int reps=argc>4?atoi(argv[4]):5;if(reps<1)reps=1;double sum=0;
      for(int k=0;k<reps;k++){
#ifdef PP_WITH_CUDA
        ok=use_cuda?pp_infer_cuda(&m,&p,&out,&pr,error,sizeof error):pp_infer_cpu(&m,&p,&out,&pr,error,sizeof error);
#else
        ok=pp_infer_cpu(&m,&p,&out,&pr,error,sizeof error);
#endif
        if(!ok){fprintf(stderr,"error: %s\n",error);break;}fprintf(stderr,"run %d total %.2f ms (pfn %.2f, backbone %.2f, heads %.2f)\n",k,pr.total_ms,pr.pfn_ms,pr.backbone_ms,pr.heads_ms);if(k)sum+=pr.total_ms;
      }
      if(ok&&reps>1)fprintf(stderr,"warm mean %.2f ms over %d runs\n",sum/(reps-1),reps-1);
      pp_output_free(&out);pp_pillars_free(&p);free(points);pp_model_close(&m);return ok?0:1;
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
