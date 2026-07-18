#ifndef PP_INFER_H
#define PP_INFER_H
#include "pointpillars.h"
#include "pp_model.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

enum { PP_OUT_H=128,PP_OUT_W=128,PP_HEADS=6,PP_BRANCHES=6 };
enum { PP_BRANCH_CLS,PP_BRANCH_REG,PP_BRANCH_HEIGHT,PP_BRANCH_SIZE,PP_BRANCH_ANGLE,PP_BRANCH_VELO };
typedef struct {float *data;size_t floats;} pp_raw_output;
typedef struct {float *data;size_t floats;} pp_cpu_workspace;
typedef struct {double pfn_ms,scatter_ms,backbone_ms,heads_ms,total_ms;size_t workspace_bytes,device_to_host_bytes;} pp_profile;
int pp_output_alloc(pp_raw_output*out);void pp_output_free(pp_raw_output*out);
void pp_cpu_workspace_free(pp_cpu_workspace*workspace);
int pp_head_classes(int head);int pp_branch_channels(int head,int branch);
float *pp_output_branch(const pp_raw_output*out,int head,int branch);
int pp_infer_cpu(const pp_model*model,const pp_pillars*pillars,
                 pp_raw_output*out,pp_cpu_workspace*workspace,
                 pp_profile*profile,char*error,size_t error_size);
#ifdef __cplusplus
}
#endif
#endif
