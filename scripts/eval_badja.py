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

import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
sys.path.insert(0,'third_party')
sys.path.insert(0,'./')

import torch
import torch.nn as nn
from torch.autograd import Variable
from ext_utils.badja_data import BADJAData
from ext_utils.joint_catalog import SMALJointInfo
import ext_utils.flowlib as flowlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import pdb
import soft_renderer as sr
import argparse
import trimesh
from nnutils.geom_utils import obj_to_cam, pinhole_cam, orthographic_cam, render_flow_soft_3

parser = argparse.ArgumentParser(description='BADJA')
parser.add_argument('--testdir', default='',
                    help='path to test dir')
parser.add_argument('--seqname', default='camel',
                    help='sequence to test')
parser.add_argument('--type', default='mesh',
                    help='load mesh data or flow or zero')
parser.add_argument('--cam_type', default='perspective',
                    help='camera model, orthographic or perspective')
parser.add_argument('--vis', dest='vis', action='store_true',
                    help='whether to draw visualization')
args = parser.parse_args()

renderer_softflf = sr.SoftRenderer(image_size=256,dist_func='hard' ,aggr_func_alpha='hard',
               camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
               light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)

def process_flow(model, imgL_o,imgR_o, mean_L, mean_R):
    testres=1

    # for gray input images
    if len(imgL_o.shape) == 2:
        imgL_o = np.tile(imgL_o[:,:,np.newaxis],(1,1,3))
        imgR_o = np.tile(imgR_o[:,:,np.newaxis],(1,1,3))

    # resize
    maxh = imgL_o.shape[0]*testres
    maxw = imgL_o.shape[1]*testres
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
    from models.VCN_exp import WarpModule, flow_reg
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
        flow, logmid, occ, biseg, objseg = rts

    # upsampling
    flow = torch.squeeze(flow).data.cpu().numpy()
    flow = np.concatenate( [cv2.resize(flow[0],(input_size[1],input_size[0]))[:,:,np.newaxis],
                            cv2.resize(flow[1],(input_size[1],input_size[0]))[:,:,np.newaxis]],-1)
    flow[:,:,0] *= imgL_o.shape[1] / max_w
    flow[:,:,1] *= imgL_o.shape[0] / max_h
    flow = np.concatenate( (flow, np.ones([flow.shape[0],flow.shape[1],1])),-1)
    torch.cuda.empty_cache()

    flow = torch.Tensor(flow).cuda()[None]
    return flow

def preprocess_image(img,mask,imgsize):
    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
    mask = mask[:,:,:1]
    # crop box
    indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
    center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
    length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
    maxlength = int(1.2*max(length))
    length = (maxlength,maxlength)

    alp = 2*length[0]/float(imgsize)
    refpp = np.asarray(center)/(imgsize/2.) - 1
    return alp, refpp,center,length[0]

def draw_joints_on_image(rgb_img, joints, visibility, region_colors, marker_types,pred=None,correct=None):
    joints = joints[:, ::-1] # OpenCV works in (x, y) rather than (i, j)

    disp_img = rgb_img.copy()    
    i=0
    for joint_coord, visible, color, marker_type in zip(joints, visibility, region_colors, marker_types):
        if visible:
            joint_coord = joint_coord.astype(int)
            cv2.circle(disp_img, tuple(joint_coord),  radius=3, color=[255,0,0], thickness = 10)
            if pred is not None:
                if correct[i]:
                    color=[0,255,0]
                else:
                    color=[0,0,255]
                error = np.linalg.norm(joint_coord - pred[i,::-1],2,-1)
                cv2.circle(disp_img, tuple(joint_coord),  radius=int(error), color=color, thickness = 3)
                cv2.line(disp_img, tuple(joint_coord), tuple(pred[i,::-1]),color , thickness = 3)
        i+=1
    return disp_img

def main():
    smal_joint_info = SMALJointInfo()
    badja_data = BADJAData(args.seqname)
    data_loader = badja_data.get_loader()
    
    print(args.testdir)
    # store all the data
    all_anno = []
    all_mesh = []
    all_cam = []
    all_fr = []
    all_fl = []
    #import pdb; pdb.set_trace()
    for anno in data_loader:
        all_anno.append(anno)
        rgb_img, sil_img, joints, visible, name = anno
        seqname = name.split('/')[-2]
        fr = int(name.split('/')[-1].split('.')[-2])
        all_fr.append(fr)
        print('%s/%d'%(seqname, fr))
        
        # load mesh data or flow
        if args.type=='mesh':
            mesh = trimesh.load('%s/pred%d.ply'%(args.testdir, fr),process=False)
            all_mesh.append(mesh)
            cam = np.loadtxt('%s/cam%d.txt'%(args.testdir,fr))
            all_cam.append(cam)
      
    if args.type=='flow':
        from models.VCN_exp import VCN
        model = VCN([1, 256, 256], md=[int(4*(256/256)),4,4,4,4], fac=1)
        model = nn.DataParallel(model, device_ids=[0])
        model.cuda()
        pretrained_dict = torch.load('/data/gengshay/vcn_weights/robexp.pth',map_location='cpu') 
        mean_L=pretrained_dict['mean_L']
        mean_R=pretrained_dict['mean_R']
        model.load_state_dict(pretrained_dict['state_dict'],strict=False)
        
    # store all the results
    pck_all = [] 
    for i in range(len(all_anno)):
        for j in range(len(all_anno)):
            if i!=j:
                # evaluate every two-frame
                refimg, refsil, refkp, refvis, refname = all_anno[i]
                tarimg, tarsil, tarkp, tarvis, tarname = all_anno[j]
                print('%s vs %s'%(refname, tarname))
                
                if args.type=='mesh':
                    refmesh, tarmesh = all_mesh[i], all_mesh[j]
                    refcam, tarcam = all_cam[i], all_cam[j]
                    img_size = max(refimg.shape)
                    renderer_softflf.rasterizer.image_size = img_size
                    # render flow between mesh 1 and 2
                    
                    refface = torch.Tensor(refmesh.faces[None]).cuda()
                    verts = torch.Tensor(np.concatenate([refmesh.vertices[None], tarmesh.vertices[None]],0)).cuda()
                    Rmat =  torch.Tensor(np.concatenate([refcam[None,:3,:3], tarcam[None,:3,:3]], 0)).cuda()
                    Tmat =  torch.Tensor(np.concatenate([refcam[None,:3,3], tarcam[None,:3,3]], 0)).cuda()
                    ppoint =  torch.Tensor(np.concatenate([refcam[None,3,2:], tarcam[None,3,2:]], 0)).cuda()
                    scale =  torch.Tensor(np.concatenate([refcam[None,3,:1], tarcam[None,3,:1]], 0)).cuda()
                    scale = scale/img_size*2
                    ppoint = ppoint/img_size * 2 -1
                    verts_fl = obj_to_cam(verts, Rmat, Tmat[:,None],nmesh=1,n_hypo=1,skin=None)
                    verts_fl = torch.cat([verts_fl,torch.ones_like(verts_fl[:, :, 0:1])], dim=-1)
                    verts_pos = verts_fl.clone()
                    
                    verts_fl = pinhole_cam(verts_fl, ppoint, scale)
                    flow_fw, bgmask_fw, fgmask_flowf = render_flow_soft_3(renderer_softflf, verts_fl[:1], verts_fl[1:], refface)
                    flow_fw[bgmask_fw]=0.
                    flow_fw = torch.cat([flow_fw, torch.zeros_like(flow_fw)[:,:,:,:1]],-1)[:,:refimg.shape[0],:refimg.shape[1]]
                elif args.type=='flow':
                    flow_fw = process_flow(model, refimg, tarimg, mean_L, mean_R)
                    flow_fw = (flow_fw)/(refimg.shape[0]/2.)
                elif args.type=='zero':
                    flow_fw = torch.zeros(refimg.shape).cuda()[None]
                refkpx = torch.Tensor(refkp.astype(float)).cuda()
                x0,y0=np.meshgrid(range(refimg.shape[1]),range(refimg.shape[0]))
                x0 = torch.Tensor(x0).cuda()
                y0 = torch.Tensor(y0).cuda()
                idx = ( (flow_fw[:,:,:,:2].norm(2,-1)<1e-6).float().view(1,-1)*1e6+ torch.pow(refkpx[:,0:1]-y0.view(1,-1),2) + torch.pow(refkpx[:,1:2]-x0.view(1,-1),2)).argmin(-1)
                samp_flow = flow_fw.view(-1,3)[idx][:,:2]
                tarkp_pred = refkpx.clone()
                tarkp_pred[:,0] = tarkp_pred[:,0] +(samp_flow[:,1])*refimg.shape[0]/2
                tarkp_pred[:,1] = tarkp_pred[:,1] +(samp_flow[:,0])*refimg.shape[1]/2
                tarkp_pred = np.asarray(tarkp_pred.cpu())

                diff = np.linalg.norm(tarkp_pred - tarkp, 2,-1)
                sqarea = np.sqrt((refsil[:,:,0]>0).sum())
                correct = diff < sqarea * 0.2
                correct = correct[np.logical_and(tarvis, refvis)]
                if args.vis:
                    rgb_vis = draw_joints_on_image(refimg, refkp, refvis, smal_joint_info.joint_colors, smal_joint_info.annotated_markers)
                    tarimg = draw_joints_on_image(tarimg, tarkp, tarvis, smal_joint_info.joint_colors, smal_joint_info.annotated_markers, pred=tarkp_pred,correct=diff < sqarea * 0.2)
                    cv2.addWeighted(rgb_vis, 0.5, flowlib.flow_to_image(np.asarray(flow_fw[0].clamp(-1,1).detach().cpu())), 0.5,0.0,rgb_vis)
                    cv2.imwrite('%s/%05d-%05d-flo.png'%(args.testdir,all_fr[i],all_fr[j]),rgb_vis[:,:,::-1]) 
                    cv2.imwrite('%s/%05d-%05d.png'%(args.testdir,all_fr[i],all_fr[j]),tarimg[:,:,::-1])
                    

                pck_all.append(correct)
    print('PCK %.02f'%(100*np.concatenate(pck_all).astype(float).mean()))

if __name__ == '__main__':
    main()
