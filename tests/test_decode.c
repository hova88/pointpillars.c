#include "pp_decode.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(void){
  pp_raw_output r;pp_detections d;
  if(!pp_output_alloc(&r)||!pp_detections_alloc(&d,500))return 1;
  for(size_t i=0;i<r.floats;i++)r.data[i]=0.0f;
  for(int h=0;h<6;h++){float*x=pp_output_branch(&r,h,0);size_t n=(size_t)pp_branch_channels(h,0)*PP_OUT_H*PP_OUT_W;for(size_t i=0;i<n;i++)x[i]=-100.0f;}
  if(!pp_decode(&r,.1f,.01f,&d)||d.count!=0){fprintf(stderr,"empty decode count=%zu\n",d.count);return 2;}
  pp_output_branch(&r,0,0)[0]=10.0f;
  if(!pp_decode(&r,.1f,.01f,&d)||d.count!=1||d.boxes[0].class_id!=0)return 3;
  pp_box reference=d.boxes[0];pp_compact_candidate c={0};c.logit=10;c.class_id=0;c.head=0;c.anchor=0;c.position=0;
  if(!pp_decode_compact(&c,1,.1f,.01f,&d)||d.count!=1||memcmp(&reference,&d.boxes[0],sizeof(reference)))return 4;
  pp_detections_free(&d);pp_output_free(&r);puts("decoder: ok");return 0;
}
