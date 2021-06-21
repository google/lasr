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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import os.path as osp
from absl import flags, app
import time
import sys
sys.path.insert(0,'third_party')

import torch
from scipy.ndimage import binary_erosion
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import cv2

from ext_utils import image as image_utils
from ext_utils.util_flow import readPFM

# -------------- Dataset ------------- #
# ------------------------------------ #
class BaseDataset(Dataset):
    ''' 
    img, mask, flow data loader
    '''

    def __init__(self, opts, filter_key=None):
        self.opts = opts
        self.img_size = opts.img_size
    
    def __len__(self):
        return self.num_imgs

    def __getitem__(self, index):
        im0idx = self.baselist[index]
        im1idx = im0idx + self.dframe if self.directlist[index]==1 else im0idx-self.dframe
        img_path = self.imglist[im0idx]
        img = cv2.imread(img_path)[:,:,::-1] / 255.0

        img_path = self.imglist[im1idx]
        imgn = cv2.imread(img_path)[:,:,::-1] / 255.0
        # Some are grayscale:
        shape = img.shape
        if len(shape) == 2:
            img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
            imgn = np.repeat(np.expand_dims(imgn, 2), 3, axis=2)

        mask = cv2.imread(self.masklist[im0idx],0)
        if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
            mask = cv2.resize(mask, img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
            mask = binary_erosion(mask,iterations=2)
        mask = np.expand_dims(mask, 2)

        maskn = cv2.imread(self.masklist[im1idx],0)
        if maskn.shape[0]!=imgn.shape[0] or maskn.shape[1]!=imgn.shape[1]:
            maskn = cv2.resize(maskn, imgn.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)
            maskn = binary_erosion(maskn,iterations=1)
        maskn = np.expand_dims(maskn, 2)
        
        # complement color
        color = 1-img[mask[:,:,0].astype(bool)].mean(0)[None,None,:]
        colorn = 1-imgn[maskn[:,:,0].astype(bool)].mean(0)[None,None,:]
        img =   img*(mask>0).astype(float)    + color  *(1-(mask>0).astype(float))
        imgn =   imgn*(maskn>0).astype(float) + colorn *(1-(maskn>0).astype(float))

        # flow
        if self.directlist[index]==1:
            flowpath = self.flowfwlist[im0idx]
            flowpathn =self.flowbwlist[im0idx+self.dframe]
        else:
            flowpath = self.flowbwlist[im0idx]
            flowpathn =self.flowfwlist[im0idx-self.dframe]
        flow = readPFM(flowpath)[0]
        flown =readPFM(flowpathn)[0]
        occ = readPFM(flowpath.replace('flo-', 'occ-'))[0]
        occn =readPFM(flowpathn.replace('flo-', 'occ-'))[0]
        #print('time: %f'%(time.time()-ss))
       
        # crop box
        indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
        indicesn = np.where(maskn>0); xidn = indicesn[1]; yidn = indicesn[0]
        center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
        centern = ( (xidn.max()+xidn.min())//2, (yidn.max()+yidn.min())//2)
        length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
        lengthn = ( (xidn.max()-xidn.min())//2, (yidn.max()-yidn.min())//2)
        maxlength = int(1.2*max(length))
        maxlengthn = int(1.2*max(lengthn))
        length = (maxlength,maxlength)
        lengthn = (maxlengthn,maxlengthn)

        x0,y0=np.meshgrid(range(2*length[0]),range(2*length[0]))
        x0=(x0+(center[0]-length[0])).astype(np.float32)
        y0=(y0+(center[1]-length[0])).astype(np.float32)
        img = cv2.remap(img,x0,y0,interpolation=cv2.INTER_LINEAR,borderValue=color[0,0])
        mask = cv2.remap(mask.astype(int),x0,y0,interpolation=cv2.INTER_NEAREST)
        flow = cv2.remap(flow,x0,y0,interpolation=cv2.INTER_LINEAR)
        occ = cv2.remap(occ,x0,y0,interpolation=cv2.INTER_LINEAR)
        

        x0n,y0n=np.meshgrid(range(2*lengthn[0]),range(2*lengthn[0]))
        x0n=(x0n+(centern[0]-lengthn[0])).astype(np.float32)
        y0n=(y0n+(centern[1]-lengthn[0])).astype(np.float32)
        imgn = cv2.remap(imgn,x0n,y0n,interpolation=cv2.INTER_LINEAR,borderValue=colorn[0,0])
        maskn = cv2.remap(maskn.astype(int),x0n,y0n,interpolation=cv2.INTER_NEAREST)
        flown = cv2.remap(flown,x0n,y0n,interpolation=cv2.INTER_LINEAR)
        occn = cv2.remap(occn,x0n,y0n,interpolation=cv2.INTER_LINEAR)

        orisize = img.shape[:2]
        orisizen = imgn.shape[:2]

        maxw=self.opts.img_size;maxh=self.opts.img_size
        img = cv2.resize(img ,  (maxw,maxh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (maxw,maxh), interpolation=cv2.INTER_NEAREST)
        
        imgn = cv2.resize(imgn ,  (maxw,maxh), interpolation=cv2.INTER_LINEAR)
        maskn = cv2.resize(maskn, (maxw,maxh), interpolation=cv2.INTER_NEAREST)

        flow = cv2.resize(flow ,  (maxw,maxh), interpolation=cv2.INTER_LINEAR)
        flown = cv2.resize(flown ,  (maxw,maxh), interpolation=cv2.INTER_LINEAR)
        occ = cv2.resize(occ ,  (maxw,maxh), interpolation=cv2.INTER_LINEAR)
        occn = cv2.resize(occn ,  (maxw,maxh), interpolation=cv2.INTER_LINEAR)

        alp = orisize[0]/maxw
        alpn = orisizen[0]/maxw
        betax,betay=np.meshgrid(range(maxw),range(maxh))

        flow[:,:,0] += (center[0]-length[0]) - (centern[0]-lengthn[0]) + betax*(alp-alpn)
        flow[:,:,1] += (center[1]-length[1]) - (centern[1]-lengthn[1]) + betay*(alp-alpn)
        flow /= alpn
    
        flow[:,:,0] = 2 * (flow[:,:,0]/maxw)
        flow[:,:,1] = 2 * (flow[:,:,1]/maxh)
        flow[:,:,2] = np.logical_and(flow[:,:,2]!=0, occ<10)  # as the valid pixels

        flown[:,:,0] += (centern[0]-lengthn[0]) - (center[0]-length[0]) + betax*(alpn-alp)
        flown[:,:,1] += (centern[1]-lengthn[1]) - (center[1]-length[1]) + betay*(alpn-alp)
        flown /= alp
    
        flown[:,:,0] = 2 * (flown[:,:,0]/maxw)
        flown[:,:,1] = 2 * (flown[:,:,1]/maxh)
        flown[:,:,2] = np.logical_and(flown[:,:,2]!=0, occn<10)  # as the valid pixels

        # Finally transpose the image to 3xHxW
        img = np.transpose(img, (2, 0, 1))
        mask = (mask>0).astype(float)
        
        imgn = np.transpose(imgn, (2, 0, 1))
        maskn = (maskn>0).astype(float)
        flow = np.transpose(flow, (2, 0, 1))
        flown = np.transpose(flown, (2, 0, 1))
            
        cam = np.zeros((7,))
        cam = np.asarray([1.,0.,0. ,1.,0.,0.,0.])
        camn = np.asarray([1.,0.,0. ,1.,0.,0.,0.])
        depth=0.; depthn=0.
        # correct cx,cy at clip space (not tx, ty)
        pps  = np.asarray([float( center[0] - length[0] ), float( center[1] - length[1]  )])
        ppsn = np.asarray([float( centern[0]- lengthn[0]), float(centern[1] - lengthn[1] )])
        if osp.exists(self.camlist[im0idx]):
            cam0=np.loadtxt(self.camlist[im0idx]).astype(np.float32)
            cam1=np.loadtxt(self.camlist[im1idx]).astype(np.float32)
            cam[:]=cam0[:-1]
            camn[:]=cam1[:-1]
            cam[0]=1./alp   # modify focal length according to rescale
            camn[0]=1./alpn
            depth = cam0[-1:]
            depthn = cam1[-1:]
        else:
            cam[0]=1./alp   # modify focal length according to rescale
            camn[0]=1./alpn

        # compute transforms
        mask = np.stack([mask,maskn])
        mask_dts = np.stack([ image_utils.compute_dt(m,iters=0) for m in mask])
        dmask_dts =  np.stack([image_utils.compute_dt(m, iters=10) for m in mask])
        mask_contour =  np.stack([image_utils.sample_contour(np.asarray(m)) for m in mask])

        try:dataid = self.dataid
        except: dataid=0

        # remove background
        elem = {
            'img': img,
            'mask': mask,
            'mask_dts': mask_dts,
            'dmask_dts': dmask_dts,
            'mask_contour': mask_contour,
            'cam': cam,
            'inds': index,

            'imgn': imgn,
            'camn': camn,
            'indsn': index,
            'flow': flow,
            'flown': flown,

            'pps': np.stack([pps,ppsn]),

            'depth':depth,
            'depthn':depthn,
            'is_canonical':  self.can_frame == im0idx,
            'is_canonicaln': self.can_frame == im1idx,
            'dataid': dataid,        

            'id0': im0idx,
            'id1': im1idx,

            'occ': occ,
            'occn': occn,  # out-of-range score; 0: border

            'shape': np.asarray(shape)[:2][::-1].copy(),
            }
        return elem
