#!/usr/bin/env python3
"""Compare C raw outputs with a direct PyTorch execution of the checkpoint."""
from __future__ import annotations
import argparse,pathlib
import numpy as np,torch
import torch.nn.functional as F
def voxelize(path):
 pts=np.fromfile(path,np.float32).reshape(-1,5);grid=np.full((512,512),-1,np.int32);vox=np.zeros((30000,20,5),np.float32);co=np.zeros((30000,4),np.int32);cnt=np.zeros(30000,np.int32);nv=0
 for q in pts:
  x,y,z=q[:3]
  if not(-51.2<=x<51.2 and -51.2<=y<51.2 and -5<=z<3):continue
  ix=int((x+51.2)/.2);iy=int((y+51.2)/.2);pi=grid[iy,ix]
  if pi<0:
   if nv==30000:continue
   pi=nv;nv+=1;grid[iy,ix]=pi;co[pi]=(0,0,iy,ix)
  j=cnt[pi]
  if j<20:vox[pi,j]=q;cnt[pi]+=1
 vox=vox[:nv];co=co[:nv];cnt=cnt[:nv];mean=vox[:,:,:3].sum(1)/cnt[:,None];feat=np.zeros((nv,20,11),np.float32);feat[:,:,:5]=vox;feat[:,:,5:8]=vox[:,:,:3]-mean[:,None]
 feat[:,:,8]=vox[:,:,0]-(co[:,3,None]*.2-51.1);feat[:,:,9]=vox[:,:,1]-(co[:,2,None]*.2-51.1);feat[:,:,10]=vox[:,:,2]-(-1.0);feat*=np.arange(20)[None,:,None]<cnt[:,None,None]
 return feat,co
def bn(x,s,prefix,eps=1e-3):return F.batch_norm(x,s[prefix+'.running_mean'],s[prefix+'.running_var'],s[prefix+'.weight'],s[prefix+'.bias'],False,0.01,eps)
def cbr(x,s,w,bnname,stride=1,padding=1,transpose=False,eps=1e-3):
 x=F.conv_transpose2d(x,s[w],stride=stride) if transpose else F.conv2d(x,s[w],stride=stride,padding=padding);return F.relu(bn(x,s,bnname,eps))
def main():
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=pathlib.Path);p.add_argument('points',type=pathlib.Path);p.add_argument('c_output',type=pathlib.Path);p.add_argument('--device',default='cuda');a=p.parse_args();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;dev=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu');s=torch.load(a.checkpoint,map_location=dev,weights_only=False)['model_state'];feat,co=voxelize(a.points);x=torch.from_numpy(feat).to(dev);w=s['vfe.pfn_layers.0.linear.weight'];x=F.linear(x,w);x=bn(x.permute(0,2,1),s,'vfe.pfn_layers.0.norm').permute(0,2,1).relu().amax(1)
 canvas=torch.zeros((1,64,512,512),device=dev);idx=torch.from_numpy(co).to(dev);canvas[0,:,idx[:,2],idx[:,3]]=x.T;ups=[]
 for stage,count in enumerate((4,6,6)):
  for layer in range(count):
   ci=1+layer*3;canvas=cbr(canvas,s,f'backbone_2d.blocks.{stage}.{ci}.weight',f'backbone_2d.blocks.{stage}.{ci+1}',2 if layer==0 else 1,1)
  if stage==0:ups.append(cbr(canvas,s,'backbone_2d.deblocks.0.0.weight','backbone_2d.deblocks.0.1',2,0))
  elif stage==1:ups.append(cbr(canvas,s,'backbone_2d.deblocks.1.0.weight','backbone_2d.deblocks.1.1',1,0,True))
  else:ups.append(cbr(canvas,s,'backbone_2d.deblocks.2.0.weight','backbone_2d.deblocks.2.1',2,0,True))
 x=cbr(torch.cat(ups,1),s,'dense_head.shared_conv.0.weight','dense_head.shared_conv.1');outs=[]
 for h in range(6):
  for branch in ('cls','reg','height','size','angle','velo'):
   base=f'dense_head.rpn_heads.{h}.'+('conv_cls' if branch=='cls' else f'conv_box.conv_{branch}');y=cbr(x,s,base+'.0.weight',base+'.1',eps=1e-5);outs.append(F.conv2d(y,s[base+'.3.weight'],s[base+'.3.bias'],padding=1).cpu().numpy()[0])
 ref=np.concatenate([z.reshape(-1) for z in outs]);raw=a.c_output.read_bytes();assert raw[:8]==b'PPOUT\0\0\2';got=np.frombuffer(raw,np.float32,offset=8);assert got.shape==ref.shape;d=np.abs(got-ref);print('device',dev,'pillars',len(feat),'floats',len(ref),'max_abs',float(d.max()),'mean_abs',float(d.mean()),'allclose',np.allclose(got,ref,rtol=2e-4,atol=2e-3))
if __name__=='__main__':main()
