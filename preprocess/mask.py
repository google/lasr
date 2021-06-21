# Copyright 2021 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cv2
import glob
import numpy as np
import pdb
import os

import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")


import sys
seqname=sys.argv[1]
detbase=sys.argv[2]
datadir='../database/DAVIS/JPEGImages/Full-Resolution/%s-tmp/'%seqname
odir='../database/DAVIS/'
imgdir= '%s/JPEGImages/Full-Resolution/%s'%(odir,seqname)
maskdir='%s/Annotations/Full-Resolution/%s'%(odir,seqname)
import shutil
if os.path.exists(imgdir): shutil.rmtree(imgdir)
if os.path.exists(maskdir): shutil.rmtree(maskdir)
os.mkdir(imgdir)
os.mkdir(maskdir)

import sys
sys.path.insert(0,'%s/projects/PointRend/'%detbase)
import point_rend
cfg = get_cfg()
point_rend.add_pointrend_config(cfg)
cfg.merge_from_file('%s/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml'%(detbase))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.3
cfg.MODEL.WEIGHTS ='https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl'

predictor = DefaultPredictor(cfg)

counter=0
for i,path in enumerate(sorted(glob.glob('%s/*'%datadir))):
    print(path)
    img = cv2.imread(path)
    shape = img.shape[:2]
    mask = np.zeros(shape)

    imgt = img
    segs = predictor(imgt)['instances'].to('cpu')

    for it,ins_cls in enumerate(segs.pred_classes):
        print(ins_cls)
        #if ins_cls ==15: # cat
        if ins_cls==0 or (ins_cls >= 14 and ins_cls <= 23):
            mask += np.asarray(segs.pred_masks[it])

    if (mask.sum())<1000: continue

    mask = mask.astype(bool).astype(int)*128
    mask = np.concatenate([mask[:,:,np.newaxis],mask[:,:,np.newaxis],mask[:,:,np.newaxis]],-1)
    mask[:,:,:2] = 0

    cv2.imwrite('%s/%05d.jpg'%(imgdir,counter), img)
    cv2.imwrite('%s/%05d.png'%(maskdir,counter), mask)

    ## vis
    #v = Visualizer(img, coco_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
    #vis = v.draw_instance_predictions(segs)
    #mask_result = np.concatenate([vis.get_image(), mask],1)
    #cv2.imwrite('%s/%05d.jpg'%(imgdir,counter), mask_result)


    counter+=1
