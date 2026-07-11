#!/usr/bin/env python3
"""Build OpenPCDet-compatible 10-sweep nuScenes point frames."""
from __future__ import annotations
import argparse,json,pathlib
import numpy as np
def load(root,n):
 with (root/'v1.0-mini'/f'{n}.json').open() as f:return json.load(f)
def qc(q):return np.array([q[0],-q[1],-q[2],-q[3]])
def rot(q,v):
 q=np.asarray(q,np.float64);v=np.asarray(v);u=q[1:];s=q[0];return 2*np.sum(v*u,axis=-1,keepdims=True)*u+(s*s-np.dot(u,u))*v+2*s*np.cross(u,v)
def transform(points,sd,ref,cal,ego):
 cs,ep=cal[sd['calibrated_sensor_token']],ego[sd['ego_pose_token']];rc,re=cal[ref['calibrated_sensor_token']],ego[ref['ego_pose_token']]
 p=rot(cs['rotation'],points)+cs['translation'];p=rot(ep['rotation'],p)+ep['translation'];p=rot(qc(re['rotation']),p-np.asarray(re['translation']));return np.asarray(rot(qc(rc['rotation']),p-np.asarray(rc['translation'])),np.float32)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('/data/nuscenes'));p.add_argument('--output',type=pathlib.Path,default=pathlib.Path('data/nuscenes-10sweep'));p.add_argument('--sweeps',type=int,default=10);p.add_argument('--limit',type=int);a=p.parse_args()
 sd_list=load(a.root,'sample_data');sd={x['token']:x for x in sd_list};cal={x['token']:x for x in load(a.root,'calibrated_sensor')};ego={x['token']:x for x in load(a.root,'ego_pose')};sensor={x['token']:x for x in load(a.root,'sensor')}
 refs=[x for x in sd_list if x.get('is_key_frame') and sensor[cal[x['calibrated_sensor_token']]['sensor_token']]['channel']=='LIDAR_TOP'];refs.sort(key=lambda x:x['timestamp']);
 if a.limit:refs=refs[:a.limit]
 a.output.mkdir(parents=True,exist_ok=True);manifest=[]
 for i,ref in enumerate(refs):
  parts=[];cur=ref
  for _ in range(a.sweeps):
   pts=np.fromfile(a.root/cur['filename'],np.float32).reshape(-1,5);keep=~((np.abs(pts[:,0])<1)&(np.abs(pts[:,1])<1));pts=pts[keep];xyz=transform(pts[:,:3],cur,ref,cal,ego) if cur is not ref else pts[:,:3].copy();inten=pts[:,3:4]/np.float32(255);lag=np.full((len(xyz),1),(ref['timestamp']-cur['timestamp'])/1e6,np.float32);parts.append(np.c_[xyz,inten,lag]);
   if not cur['prev']:break
   cur=sd[cur['prev']]
  out=np.ascontiguousarray(np.concatenate(parts),dtype=np.float32);np.random.default_rng(int(ref['sample_token'][:16],16)).shuffle(out);name=f'{ref["timestamp"]}_{ref["sample_token"]}.bin';out.tofile(a.output/name);manifest.append({'sample_token':ref['sample_token'],'sample_data_token':ref['token'],'timestamp':ref['timestamp'],'file':name,'points':len(out),'sweeps':len(parts)});print(f'[{i+1}/{len(refs)}] {name}: {len(out)} points')
 (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
