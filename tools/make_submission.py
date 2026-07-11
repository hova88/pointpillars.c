#!/usr/bin/env python3
"""Convert C detector JSON files to the official nuScenes result schema."""
import argparse,json,math,pathlib
import numpy as np
from nuscenes.utils.splits import create_splits_scenes
NAMES=['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
def load(root,n):
 with (root/'v1.0-mini'/f'{n}.json').open() as f:return json.load(f)
def qm(a,b):
 w,x,y,z=a;W,X,Y,Z=b;return [w*W-x*X-y*Y-z*Z,w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W]
def rot(q,v):q=np.asarray(q);v=np.asarray(v);u=q[1:];s=q[0];return 2*np.sum(v*u,axis=-1,keepdims=True)*u+(s*s-np.dot(u,u))*v+2*s*np.cross(u,v)
def attr(name,speed):
 if name in ('car','truck','construction_vehicle','bus','trailer'):return 'vehicle.moving' if speed>.2 else 'vehicle.stopped'
 if name=='pedestrian':return 'pedestrian.moving' if speed>.2 else 'pedestrian.standing'
 if name in ('motorcycle','bicycle'):return 'cycle.with_rider' if speed>.2 else 'cycle.without_rider'
 return ''
def main():
 p=argparse.ArgumentParser();p.add_argument('detections',type=pathlib.Path);p.add_argument('manifest',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);p.add_argument('--root',type=pathlib.Path,default=pathlib.Path('/data/nuscenes'));p.add_argument('--split',default='mini_val');a=p.parse_args();sd={x['token']:x for x in load(a.root,'sample_data')};cal={x['token']:x for x in load(a.root,'calibrated_sensor')};ego={x['token']:x for x in load(a.root,'ego_pose')};samples={x['token']:x for x in load(a.root,'sample')};scenes={x['token']:x for x in load(a.root,'scene')};allowed=set(create_splits_scenes()[a.split]);result={}
 for item in json.loads(a.manifest.read_text()):
  d=sd[item['sample_data_token']]
  if scenes[samples[d['sample_token']]['scene_token']]['name'] not in allowed:continue
  cs=cal[d['calibrated_sensor_token']];ep=ego[d['ego_pose_token']];boxes=json.loads((a.detections/(item['file']+'.json')).read_text());out=[]
  for b in boxes:
   center=rot(ep['rotation'],rot(cs['rotation'],[b['x'],b['y'],b['z']])+cs['translation'])+ep['translation'];qy=[math.cos(b['yaw']/2),0,0,math.sin(b['yaw']/2)];q=qm(ep['rotation'],qm(cs['rotation'],qy));vel=rot(ep['rotation'],rot(cs['rotation'],[b['vx'],b['vy'],0]));name=NAMES[b['class_id']];out.append({'sample_token':item['sample_token'],'translation':[float(x) for x in center],'size':[b['dy'],b['dx'],b['dz']],'rotation':[float(x) for x in q],'velocity':[float(vel[0]),float(vel[1])],'detection_name':name,'detection_score':b['score'],'attribute_name':attr(name,float(np.linalg.norm(vel[:2])))})
  result[item['sample_token']]=out
 obj={'meta':{'use_camera':False,'use_lidar':True,'use_radar':False,'use_map':False,'use_external':False},'results':result};a.output.write_text(json.dumps(obj)+'\n');print('wrote',a.output,len(result),'samples',sum(map(len,result.values())),'boxes')
if __name__=='__main__':main()
