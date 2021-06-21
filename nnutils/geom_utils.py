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

"""
Utils related to geometry like projection,,
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import soft_renderer as sr
import pdb

def pinhole_cam(verts,pp,fl):
    n_hypo = verts.shape[0] // pp.shape[0]
    pp = pp.clone()[:,None].repeat(1,n_hypo,1).view(-1,2)
    fl = fl.clone()[:,None].view(-1,1)
    verts = verts.clone()
    verts[:,:,1] = pp[:,1:2]+verts[:, :, 1].clone()*fl/ verts[:,:,2].clone()
    verts[:,:,0] = pp[:,0:1]+verts[:, :, 0].clone()*fl/ verts[:,:,2].clone()
    return verts

def orthographic_cam(verts,pp,fl):
    n_hypo = verts.shape[0] // pp.shape[0]
    pp = pp.clone()[:,None].repeat(1,n_hypo,1).view(-1,2)
    fl = fl.clone()[:,None].view(-1,1)
    verts = verts.clone()
    verts[:,:,1] = pp[:,1:2]+verts[:, :, 1].clone()*fl
    verts[:,:,0] = pp[:,0:1]+verts[:, :, 0].clone()*fl
    return verts
            
def obj_to_cam(verts, Rmat, Tmat,nmesh,n_hypo,skin,tocam=True):
    """
    transform from canonical object coordinates to camera coordinates
    """
    verts = verts.clone()
    Rmat = Rmat.clone()
    Tmat = Tmat.clone()
    verts = verts.view(-1,verts.shape[1],3)

    bodyR = Rmat[::nmesh].clone()
    bodyT = Tmat[::nmesh].clone()
    if nmesh>1:
        vs = []
        for k in range(nmesh-1):
            partR = Rmat[k+1::nmesh].clone()
            partT = Tmat[k+1::nmesh].clone()
            vs.append( (verts.matmul(partR) + partT)[:,np.newaxis] )
        vs = torch.cat(vs,1) # N, K, Nv, 3
        vs = (vs * skin).sum(1)
    else:
        vs = verts
    
    if tocam:
        vs =  vs.clone().matmul(bodyR) + bodyT
    else:
        vs = vs.clone()
    return vs

def render_flow_soft_3(renderer_soft, verts, verts_target, faces):
    """
    Render optical flow from two frame 3D vertex locations 
    """
    offset = torch.Tensor( renderer_soft.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
    
    verts_pos_px = renderer_soft.render_mesh(sr.Mesh(verts_pre, faces, 
                                            textures=verts_target[:,:,:3],texture_type='vertex')).clone()
    fgmask = verts_pos_px[:,-1]
    verts_pos_px = verts_pos_px.permute(0,2,3,1)
    
    bgmask = (verts_pos_px[:,:,:,2]<1e-9)
    verts_pos_px[bgmask]=10

    verts_pos0_px = torch.Tensor(np.meshgrid(range(bgmask.shape[2]), range(bgmask.shape[1]))).cuda()
    verts_pos0_px[0] = verts_pos0_px[0]*2 / (bgmask.shape[2] - 1) - 1
    verts_pos0_px[1] = verts_pos0_px[1]*2 / (bgmask.shape[1] - 1) - 1
    verts_pos0_px = verts_pos0_px.permute(1,2,0)[None] 

    flow_fw = (verts_pos_px[:,:,:,:2] - verts_pos0_px)
    flow_fw[bgmask] = flow_fw[bgmask].detach()
    return flow_fw, bgmask, fgmask

def label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])
