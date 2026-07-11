#!/usr/bin/env python3
"""Export the native OpenPCDet nuScenes multihead checkpoint for the C runtime."""
from __future__ import annotations
import argparse,pathlib,struct,zlib
import numpy as np
import torch,yaml

MAGIC=b"PPWGT\0\0\0";VERSION=2;ALIGN=64;NAME_BYTES=48
HEADER=struct.Struct("<8sIIIIQQI20x");ENTRY=struct.Struct(f"<{NAME_BYTES}sI4I4xQQI20x")
def align(n):return (n+ALIGN-1)&-ALIGN
def fold(w,gamma,beta,mean,var,axis,eps=1e-3):
 s=gamma/torch.sqrt(var+eps);shape=[1]*w.ndim;shape[axis]=len(s)
 return w*s.reshape(shape),beta-mean*s
def arr(x):return x.detach().cpu().contiguous().numpy().astype('<f4',copy=False)

def export(state,cfg):
 out=[]
 def add(n,x):out.append((n,arr(x) if torch.is_tensor(x) else np.asarray(x,dtype='<f4')))
 def folded(name,w,bn,axis=0,eps=1e-3):
  fw,fb=fold(state[w],state[bn+'.weight'],state[bn+'.bias'],state[bn+'.running_mean'],state[bn+'.running_var'],axis,eps);add(name+'.weight',fw);add(name+'.bias',fb)
 folded('pfn','vfe.pfn_layers.0.linear.weight','vfe.pfn_layers.0.norm')
 counts=(4,6,6)
 for stage,n in enumerate(counts):
  for layer in range(n):
   ci=1+layer*3;folded(f'backbone.{stage}.{layer}',f'backbone_2d.blocks.{stage}.{ci}.weight',f'backbone_2d.blocks.{stage}.{ci+1}')
 folded('deblock.0','backbone_2d.deblocks.0.0.weight','backbone_2d.deblocks.0.1',0)
 folded('deblock.1','backbone_2d.deblocks.1.0.weight','backbone_2d.deblocks.1.1',1)
 folded('deblock.2','backbone_2d.deblocks.2.0.weight','backbone_2d.deblocks.2.1',1)
 folded('shared','dense_head.shared_conv.0.weight','dense_head.shared_conv.1')
 branches=('cls','reg','height','size','angle','velo')
 for h in range(6):
  for branch in branches:
   base=f'dense_head.rpn_heads.{h}.'+('conv_cls' if branch=='cls' else f'conv_box.conv_{branch}')
   folded(f'head.{h}.{branch}.mid',base+'.0.weight',base+'.1',eps=1e-5)
   add(f'head.{h}.{branch}.out.weight',state[base+'.3.weight']);add(f'head.{h}.{branch}.out.bias',state[base+'.3.bias'])
 model=cfg['MODEL'];data=cfg['DATA_CONFIG'];dense=model['DENSE_HEAD'];anchors=dense['ANCHOR_GENERATOR_CONFIG']
 add('meta.range',data['POINT_CLOUD_RANGE']);add('meta.voxel',data['DATA_PROCESSOR'][2]['VOXEL_SIZE'])
 add('meta.anchor',[[*a['anchor_sizes'][0],a['anchor_bottom_heights'][0]] for a in anchors])
 labels=[]
 for h in range(6):labels.extend(state[f'dense_head.rpn_heads.{h}.head_label_indices'].float().tolist())
 add('meta.labels',labels)
 return out

def write(path,arrays):
 records=[];blobs=[];cursor=0
 for name,a in arrays:
  a=np.ascontiguousarray(a,dtype='<f4');blob=a.tobytes();cursor=align(cursor);dims=list(a.shape)+[0]*(4-a.ndim)
  records.append(ENTRY.pack(name.encode(),a.ndim,*dims,cursor,len(blob),zlib.crc32(blob)));blobs.append((cursor,blob));cursor+=len(blob)
 table=b''.join(records);start=align(HEADER.size+len(table));data_bytes=align(cursor)
 with path.open('wb') as f:
  f.write(HEADER.pack(MAGIC,VERSION,len(arrays),ENTRY.size,ALIGN,start,data_bytes,zlib.crc32(table)));f.write(table);f.write(b'\0'*(start-f.tell()))
  for off,blob in blobs:f.write(b'\0'*(start+off-f.tell()));f.write(blob)
  f.write(b'\0'*(start+data_bytes-f.tell()))
 print(f'wrote {path}: {len(arrays)} tensors, {path.stat().st_size/2**20:.2f} MiB')

def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=pathlib.Path);p.add_argument('config',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args()
 cfg=yaml.safe_load(a.config.read_text());assert cfg['CLASS_NAMES']==['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
 assert cfg['MODEL']['DENSE_HEAD']['NAME']=='AnchorHeadMulti' and cfg['MODEL']['DENSE_HEAD']['SEPARATE_MULTIHEAD']
 obj=torch.load(a.checkpoint,map_location='cpu',weights_only=False);state=obj['model_state'];write(a.output,export(state,cfg))
if __name__=='__main__':main()
