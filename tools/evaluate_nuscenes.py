#!/usr/bin/env python3
import argparse,pathlib
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
p=argparse.ArgumentParser();p.add_argument('result',type=pathlib.Path);p.add_argument('--root',default='/data/nuscenes');p.add_argument('--output',default='evaluation/nuscenes-mini');a=p.parse_args();n=NuScenes(version='v1.0-mini',dataroot=a.root,verbose=False);NuScenesEval(n,config_factory('detection_cvpr_2019'),str(a.result),eval_set='mini_val',output_dir=a.output,verbose=True).main(render_curves=False)
