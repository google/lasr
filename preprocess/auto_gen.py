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


from __future__ import print_function
import sys
sys.path.insert(0,'third_party')

import cv2
import pdb
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time
from ext_utils.io import mkdir_p
from ext_utils.util_flow import write_pfm
import glob
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='VCN+expansion')
parser.add_argument('--datapath', default='/ssd/kitti_scene/training/',
                    help='dataset path')
parser.add_argument('--loadmodel', default=None,
                    help='model path')
parser.add_argument('--testres', type=float, default=1,
                    help='resolution')
parser.add_argument('--maxdisp', type=int ,default=256,
                    help='maxium disparity. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float ,default=1,
                    help='controls the shape of search grid. Only affect the coarse cost volume size')
parser.add_argument('--flow_threshold', type=float ,default=0.05,
                    help='flow threshold that controls frame skipping')
args = parser.parse_args()


mean_L = [[0.33,0.33,0.33]]
mean_R = [[0.33,0.33,0.33]]

# construct model, VCN-expansion
from ext_nnutils.VCNplus import VCN
from ext_nnutils.VCNplus import WarpModule, flow_reg
model = VCN([1, 256, 256], md=[int(4*(args.maxdisp/256)),4,4,4,4], fac=args.fac)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('dry run')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

seqname = args.datapath.strip().split('/')[-2]

mkdir_p('./%s/JPEGImages' % (seqname))
mkdir_p('./%s/FlowFW'     % (seqname))
mkdir_p('./%s/FlowBW'     % (seqname))
mkdir_p('./%s/Annotations'% (seqname))

test_left_img = sorted(glob.glob('%s/*'%(args.datapath)))
silhouettes = sorted(glob.glob('%s/*'%(args.datapath.replace('JPEGImages', 'Annotations'))))

def flow_inference(imgL_o, imgR_o):
    # for gray input images
    if len(imgL_o.shape) == 2:
        imgL_o = np.tile(imgL_o[:,:,np.newaxis],(1,1,3))
        imgR_o = np.tile(imgR_o[:,:,np.newaxis],(1,1,3))

    # resize
    maxh = imgL_o.shape[0]*args.testres
    maxw = imgL_o.shape[1]*args.testres
    max_h = int(maxh // 64 * 64)
    max_w = int(maxw // 64 * 64)
    if max_h < maxh: max_h += 64
    if max_w < maxw: max_w += 64

    input_size = imgL_o.shape
    imgL = cv2.resize(imgL_o,(max_w, max_h))
    imgR = cv2.resize(imgR_o,(max_w, max_h))
    imgL_noaug = torch.Tensor(imgL/255.)[np.newaxis].float().cuda()

    # flip channel, subtract mean
    imgL = imgL[:,:,::-1].copy() / 255. - np.asarray(mean_L).mean(0)[np.newaxis,np.newaxis,:]
    imgR = imgR[:,:,::-1].copy() / 255. - np.asarray(mean_R).mean(0)[np.newaxis,np.newaxis,:]
    imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
    imgR = np.transpose(imgR, [2,0,1])[np.newaxis]

    # modify module according to inputs
    for i in range(len(model.module.reg_modules)):
        model.module.reg_modules[i] = flow_reg([1,max_w//(2**(6-i)), max_h//(2**(6-i))], 
                        ent=getattr(model.module, 'flow_reg%d'%2**(6-i)).ent,\
                        maxdisp=getattr(model.module, 'flow_reg%d'%2**(6-i)).md,\
                        fac=getattr(model.module, 'flow_reg%d'%2**(6-i)).fac).cuda()
    for i in range(len(model.module.warp_modules)):
        model.module.warp_modules[i] = WarpModule([1,max_w//(2**(6-i)), max_h//(2**(6-i))]).cuda()

    # get intrinsics
    intr_list = [torch.Tensor(inxx).cuda() for inxx in [[1],[1],[1],[1],[1],[0],[0],[1],[0],[0]]]
    fl_next = 1
    intr_list.append(torch.Tensor([input_size[1] / max_w]).cuda())
    intr_list.append(torch.Tensor([input_size[0] / max_h]).cuda())
    intr_list.append(torch.Tensor([fl_next]).cuda())
    
    disc_aux = [None,None,None,intr_list,imgL_noaug,None]
    
    # forward
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())
    with torch.no_grad():
        imgLR = torch.cat([imgL,imgR],0)
        model.eval()
        torch.cuda.synchronize()
        start_time = time.time()
        rts = model(imgLR, disc_aux)
        torch.cuda.synchronize()
        ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
        flow, occ = rts

    # upsampling
    occ = cv2.resize(occ.data.cpu().numpy(),  (input_size[1],input_size[0]),interpolation=cv2.INTER_LINEAR)
    flow = torch.squeeze(flow).data.cpu().numpy()
    flow = np.concatenate( [cv2.resize(flow[0],(input_size[1],input_size[0]))[:,:,np.newaxis],
                            cv2.resize(flow[1],(input_size[1],input_size[0]))[:,:,np.newaxis]],-1)
    flow[:,:,0] *= imgL_o.shape[1] / max_w
    flow[:,:,1] *= imgL_o.shape[0] / max_h
    flow = np.concatenate( (flow, np.ones([flow.shape[0],flow.shape[1],1])),-1)
    return flow, occ

def main():
    model.eval()
    inx=0;jnx=1
    ix =0
    while True:
        print('%s/%s'%(test_left_img[inx],test_left_img[jnx]))
        imgL_o = cv2.imread(test_left_img[inx])[:,:,::-1]
        imgR_o = cv2.imread(test_left_img[jnx])[:,:,::-1]
        mask =cv2.imread(silhouettes[inx],0)>0
        maskR =cv2.imread(silhouettes[jnx],0)>0
            
        indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
        length = [ (xid.max()-xid.min())//2, (yid.max()-yid.min())//2]

        flowfw, occfw = flow_inference(imgL_o, imgR_o)
        flowfw_normed = np.concatenate( [flowfw[:,:,:1]/length[0], flowfw[:,:,1:2]/length[1]],-1 )
        medflow = np.median(np.linalg.norm(flowfw_normed[mask],2,-1))
        medocc = np.median(occfw[mask])
        print('%.3f, %.2f'%(medflow, medocc))
       
        if medflow > args.flow_threshold:
            flowbw, occbw = flow_inference(imgR_o, imgL_o)
            # save predictions
            write_pfm('%s/FlowFW/flo-%05d.pfm'% (seqname,ix  ),flowfw[::-1].astype(np.float32))
            write_pfm('%s/FlowFW/occ-%05d.pfm'% (seqname,ix  ),occfw[::-1].astype(np.float32))
            write_pfm('%s/FlowBW/flo-%05d.pfm'% (seqname,ix+1),flowbw[::-1].astype(np.float32))
            write_pfm('%s/FlowBW/occ-%05d.pfm'% (seqname,ix+1),occbw[::-1].astype(np.float32))
            cv2.imwrite('%s/JPEGImages/%05d.jpg'% (seqname,ix), imgL_o[:,:,::-1])
            cv2.imwrite('%s/JPEGImages/%05d.jpg'% (seqname,ix+1), imgR_o[:,:,::-1])
            cv2.imwrite('%s/Annotations/%05d.png'% (seqname,ix), mask.astype(np.uint8))
            cv2.imwrite('%s/Annotations/%05d.png'% (seqname,ix+1), maskR.astype(np.uint8))
            inx=jnx 
            ix+=1
        jnx+=1


        torch.cuda.empty_cache()
                
            

if __name__ == '__main__':
    main()

