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

from absl import app
from absl import flags
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')

import numpy as np
import configparser
import time
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import trimesh
import pytorch3d
import pytorch3d.loss

import pdb
from ext_utils import mesh
from ext_utils.quatlib import q_rnd_m, q_scale_m
from ext_utils.util_rot import compute_geodesic_distance_from_two_matrices
from ext_utils import geometry as geom_utils
from ext_nnutils import net_blocks as nb
from ext_nnutils.mesh_net import MeshNet
import kornia
import configparser
import soft_renderer as sr
from nnutils.geom_utils import pinhole_cam, obj_to_cam, render_flow_soft_3
from nnutils.geom_utils import label_colormap
citylabs = label_colormap()

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_boolean('noise', True, 'Add random noise to pose')
flags.DEFINE_boolean('symmetric', True, 'Use symmetric mesh or not')
flags.DEFINE_boolean('symmetric_loss', True, 'Use symmetric loss or not')
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')

flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_boolean('symmetric_texture', True, 'if true texture is symmetric!')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')
flags.DEFINE_integer('symidx', 0, 'symmetry index: 0-x 1-y 2-z')
flags.DEFINE_integer('n_bones', 1, 'num of meshes')
flags.DEFINE_string('n_faces', '1280','number of faces for remeshing')
flags.DEFINE_integer('n_hypo', 1, 'num of hypothesis cameras')

flags.DEFINE_boolean('only_mean_sym', False, 'If true, only the meanshape is symmetric')
flags.DEFINE_string('dataname', 'fashion', 'name of the test data')
flags.DEFINE_string('opt_tex', 'no', 'optimize texture')
flags.DEFINE_float('rscale', 1.0, 'scale random variance')
flags.DEFINE_float('l1tex_wt', 1.0, 'weight of l1 texture')
flags.DEFINE_float('sigval', 1e-4, 'blur radius of soft renderer')

def render_flow_soft_2(renderer_soft, verts, faces, verts_pos0, verts_pos1, pp0, pp1, proj_cam0,proj_cam1):
    # flow (no splat): 1) get mask; 2) render 3D coords for 1st/2nd frame 
    n_hypo = verts.shape[0] // faces.shape[0]
    faces = faces[:,None].repeat(1,n_hypo,1,1).view(-1,faces.shape[1],3)
    verts_pos0 = verts_pos0.clone()
    verts_pos1 = verts_pos1.clone()
    offset = torch.Tensor( renderer_soft.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]

    nb = verts.shape[0]
    verts_pos_px = renderer_soft.render_mesh(sr.Mesh(torch.cat([verts_pre         ,verts_pre],0),
                                                     torch.cat([faces             ,faces],0), 
                                            textures=torch.cat([verts_pos0[:,:,:3],verts_pos1[:,:,:3]],0),texture_type='vertex')).clone()
    fgmask = verts_pos_px[:nb,-1]
    verts_pos_px = verts_pos_px[:,:3]
    verts_pos0_px = verts_pos_px[:nb].permute(0,2,3,1)
    verts_pos1_px = verts_pos_px[nb:].permute(0,2,3,1)

    bgmask = (verts_pos0_px[:,:,:,2]<1e-9) | (verts_pos1_px[:,:,:,2]<1e-9)
    verts_pos0_px[bgmask]=10
    verts_pos1_px[bgmask]=10

    # projet 3D verts with different intrinsics
    verts_pos0_px[:,:,:,1] = pp0[:,1:2,np.newaxis]+verts_pos0_px[:,:,:,1].clone()*proj_cam0[:,:1,np.newaxis] / verts_pos0_px[:,:,:,2].clone()
    verts_pos0_px[:,:,:,0] = pp0[:,0:1,np.newaxis]+verts_pos0_px[:,:,:,0].clone()*proj_cam0[:,:1,np.newaxis] / verts_pos0_px[:,:,:,2].clone()
    verts_pos1_px[:,:,:,1] = pp1[:,1:2,np.newaxis]+verts_pos1_px[:,:,:,1].clone()*proj_cam1[:,:1,np.newaxis] / verts_pos1_px[:,:,:,2].clone()
    verts_pos1_px[:,:,:,0] = pp1[:,0:1,np.newaxis]+verts_pos1_px[:,:,:,0].clone()*proj_cam1[:,:1,np.newaxis] / verts_pos1_px[:,:,:,2].clone()
    flow_fw = (verts_pos1_px - verts_pos0_px.detach())[:,:,:,:2]
    flow_fw[bgmask] = flow_fw[bgmask].detach()
    return flow_fw, bgmask, fgmask

def reg_decay(curr_steps, max_steps, min_wt,max_wt):
    """
    max weight to min weight
    """
    if curr_steps>max_steps:current = min_wt
    else:
        current = np.exp(curr_steps/float(max_steps)*(np.log(min_wt)-np.log(max_wt))) * max_wt 
    return current

class LASR(MeshNet):
    def __init__(self, input_shape, opts, nz_feat=100):
        super(LASR, self).__init__(input_shape, opts, nz_feat)
        self.rest_rs = torch.Tensor([[0,0,0,1]]).repeat(opts.n_bones-1,1).cuda()
        self.transg =  torch.Tensor([[0,0,0]]).cuda().repeat(opts.n_bones-1,1)  # not including the body-to-world transform
        self.ctl_rs =  torch.Tensor([[0,0,0,1]]).repeat(opts.n_hypo*(opts.n_bones-1),1).cuda()
        self.rest_ts = torch.zeros(opts.n_hypo*(opts.n_bones-1),3).cuda()
        self.ctl_ts =  torch.zeros(opts.n_hypo*(opts.n_bones-1),3).cuda()
        self.log_ctl = torch.Tensor([[0,0,0]]).cuda().repeat(opts.n_hypo*(opts.n_bones-1),1)  # control point varuance

        if self.opts.n_bones>1:
            self.ctl_rs  = nn.Parameter(self.ctl_rs) 
            self.rest_ts = nn.Parameter(self.rest_ts)
            self.ctl_ts  = nn.Parameter(self.ctl_ts) 
            self.log_ctl = nn.Parameter(self.log_ctl)

        # For renderering.
        self.renderer_soft = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_softflf = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4, gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        self.renderer_softflb = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4, gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_softtex = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,gamma_val=1e-2,
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        
        self.renderer_softpart = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,gamma_val=1e-4, 
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        

    def forward(self, batch_input):
        if self.training:
            local_batch_size = batch_input['input_imgs  '].shape[0]//2
            for k,v in batch_input.items():
                batch_input[k] = v.view(local_batch_size,2,-1).permute(1,0,2).reshape(v.shape)
            self.input_imgs   = batch_input['input_imgs  ']
            self.imgs         = batch_input['imgs        ']
            self.masks        = batch_input['masks       ']
            self.cams         = batch_input['cams        ']
            self.depth_gt     = batch_input['depth_gt    ']
            self.flow         = batch_input['flow        ']
            self.dts_barrier  = batch_input['dts_barrier ']
            self.ddts_barrier = batch_input['ddts_barrier']
            self.mask_contour = batch_input['mask_contour']
            self.pp           = batch_input['pp          ']
            self.occ          = batch_input['occ         ']
            self.oriimg_shape = batch_input['oriimg_shape']
            self.frameid      = batch_input['frameid']
            self.dataid      = batch_input['dataid']
            self.is_canonical = batch_input['is_canonical']
        else:
            local_batch_size = 1
            self.input_imgs = batch_input
        img = self.input_imgs

#        torch.cuda.synchronize()
#        start_time = time.time()

        opts = self.opts
        pred_v, tex,faces= self.get_mean_shape(local_batch_size)
        if self.training:
            tex = tex.view(2*local_batch_size,-1,opts.n_hypo,tex.shape[-2],3)
            texnew = torch.zeros_like(tex[:,0])
            for i in range(local_batch_size*2):
                texnew[i] = tex[i,self.dataid[i]]
            tex = texnew.reshape(-1,tex.shape[-2],3)
        faces = faces.cuda()
        
        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        #if opts.n_hypo==1:
        self.apply(set_bn_eval)

        img_feat = self.encoder.forward(img)
        scale, trans, quat, depth, ppoint = self.code_predictor.forward(img_feat)
        if not self.training:
           return scale,trans,quat, depth, ppoint
        
        # transform the CNN-predicted focal length 
        # (wrt original image without cropping) to the cropped image
        scale = self.cams[:,:1]*scale # here assumes intrinsic may change
        # transform depth as well, to ensure the rendered silhouette 
        # occupies enough pixels of the cropped image at initialization 
        depth[:,:1] = self.cams[:,:1]* depth[:,:1]; depth = depth.view(-1,1) 
       
        # transform the CNN-predicted principal points 
        # (wrt original image, before cropping) to the cropped image
        ppb1 = self.cams[:local_batch_size,:1]*self.pp[:local_batch_size]/(opts.img_size/2.)
        ppb2 = self.cams[local_batch_size:,:1]*self.pp[local_batch_size:]/(opts.img_size/2.)
        # represent pp of croped frame 2 as transformed pp of cropped frame 1
        # to reduce ambiguity caused by inconsistent pp over time
        ppa1 = ppoint[:local_batch_size] + ppb1 + 1
        ppa2 = ppa1 * (self.cams[local_batch_size:,:1] / self.cams[:local_batch_size,:1]) 
        ppoint[local_batch_size:]= ppa2 - ppb2 -1
        
        quat = quat.view(-1,9)
        if opts.noise and self.epoch>0 and self.iters<100 and self.iters>1:
            # add noise
            decay_factor = 0.2*(1e-4)**(self.iters/100)
            decay_factor_r = decay_factor * np.ones(quat.shape[0])
            ### smaller noise for bones
            decay_factor_r = decay_factor_r.reshape((-1,opts.n_bones))
            decay_factor_r[:,1:] *= 1
            decay_factor_r[:,0] *=  1
            decay_factor_r = decay_factor_r.flatten()
            noise = q_scale_m(q_rnd_m(b=quat.shape[0]), decay_factor_r)  # wxyz
            noise = torch.Tensor(noise).cuda()
            noise = torch.cat([noise[:,1:], noise[:,:1]],-1)
            quat = quat.view(-1,3,3).matmul(kornia.quaternion_to_rotation_matrix(noise)).view(-1,9)

            decay_factor_s = decay_factor
            scale = scale * (decay_factor_s*torch.normal(torch.zeros(scale.shape).cuda(),opts.rscale)).exp()
            
        depth = depth.view(local_batch_size*2,1,opts.n_bones,1).repeat(1,opts.n_hypo,1,1).view(-1,1)
        trans = trans.view(local_batch_size*2,1,opts.n_bones,2).repeat(1,opts.n_hypo,1,1).view(-1,2)

        if opts.use_gtpose:
            # w/ gt cam
            quat_pred = quat.clone()
            scale_pred = scale.clone()
            trans_pred = trans.clone()
            ppoint_pred = ppoint.clone()
            depth_pred = depth.clone()

            scale = 10*self.cams[:,:1]
            trans = self.cams[:,1:3]
            quat = kornia.quaternion_to_rotation_matrix( torch.cat( (self.cams[:,4:],self.cams[:,3:4] ) ,-1) ).view(-1,9)
            depth    = self.depth_gt[:]
            halforisize = 0.5*opts.img_size / self.cams[:,:1]  # half size before scaling 
            ppoint = (0.5*self.oriimg_shape - self.pp[:]) / halforisize - 1
    
        ## rendering
        # obj-cam rigid transform;  proj_cam: [focal, tx,ty,qw,qx,qy,qz]; 
        # 1st/2nd frame stored as 1st:[0:local_batch_size], 2nd: [local_batch_size:-1]
        # transforms [body-to-cam, part1-to-body, ...]
        Rmat = quat.view(-1,3,3).permute(0,2,1)
        Tmat = torch.cat([trans, depth],1)
        if opts.n_bones>1:
            # skin computation
            # GMM
            dis_norm = (self.ctl_ts.view(opts.n_hypo,-1,1,3) - pred_v.view(2*local_batch_size,opts.n_hypo,-1,3)[0,:,None].detach()) # p-v, H,J,1,3 - H,1,N,3
            dis_norm = dis_norm.matmul(kornia.quaternion_to_rotation_matrix(self.ctl_rs).view(opts.n_hypo,-1,3,3)) # h,j,n,3
            dis_norm = self.log_ctl.exp().view(opts.n_hypo,-1,1,3) * dis_norm.pow(2) # (p-v)^TS(p-v) 
            skin = (-10 * dis_norm.sum(3)).softmax(1)[:,:,:,None] # h,j,n,1
            # color palette
            colormap = torch.Tensor(citylabs[:skin.shape[1]]).cuda() # 5x3
            skin_colors = (skin[self.optim_idx] * colormap[:,None]).sum(0)/256.
            skin = skin.repeat(local_batch_size*2,1,1,1)
          
            # rest_ts: joint center
            # ctl_ts: control points
            rest_ts = self.rest_ts[:,None,:,None].repeat(local_batch_size*2,1,1,1).view(-1,opts.n_bones-1,3,1) # bh,m,3,1
            ctl_ts = self.ctl_ts[:,None,:,None].repeat(local_batch_size*2,1,1,1).view(-1,opts.n_bones-1,3,1) # bh,m,3,1
            
            Rmat = Rmat.view(-1,opts.n_bones,3,3)
            Tmat = Tmat.view(-1,opts.n_bones,3,1)
            Tmat[:,1:] = -Rmat[:,1:].matmul(rest_ts)+Tmat[:,1:]+rest_ts
            Rmat[:,1:] = Rmat[:,1:].permute(0,1,3,2)
            Rmat = Rmat.view(-1,3,3)
            Tmat = Tmat.view(-1,3)

            self.joints_proj = obj_to_cam(rest_ts[:,:,:,0], Rmat.detach(), Tmat[:,np.newaxis].detach(), opts.n_bones, opts.n_hypo,torch.eye(opts.n_bones-1)[None,:,:,None].cuda())
            self.joints_proj = pinhole_cam(self.joints_proj, ppoint.detach(), scale.detach())
            self.ctl_proj =    obj_to_cam(ctl_ts[:,:,:,0],  Rmat.detach(), Tmat[:,np.newaxis].detach(), opts.n_bones, opts.n_hypo, torch.eye(opts.n_bones-1)[None,:,:,None].cuda())
            self.ctl_proj = pinhole_cam(self.ctl_proj, ppoint.detach(), scale.detach())
        else:skin=None                

        self.deform_v = obj_to_cam(pred_v, Rmat.view(-1,3,3), Tmat[:,np.newaxis,:],opts.n_bones, opts.n_hypo,skin,tocam=False)

#        torch.cuda.synchronize()
#        print('before rend time:%.2f'%(time.time()-start_time))

        
        # 1) flow rendering 
        verts_fl = obj_to_cam(pred_v, Rmat, Tmat[:,np.newaxis,:],opts.n_bones, opts.n_hypo,skin)
        verts_fl = torch.cat([verts_fl,torch.ones_like(verts_fl[:, :, 0:1])], dim=-1)
        verts_pos0 = verts_fl.view(2*local_batch_size,opts.n_hypo,-1,4)[:local_batch_size].clone().view(local_batch_size*opts.n_hypo,-1,4)
        verts_pos1 = verts_fl.view(2*local_batch_size,opts.n_hypo,-1,4)[local_batch_size:].clone().view(local_batch_size*opts.n_hypo,-1,4)
        verts_fl = pinhole_cam(verts_fl, ppoint, scale)

        dmax=verts_fl[:,:,-2].max()
        dmin=verts_fl[:,:,-2].min()
        self.renderer_softflf.rasterizer.near=dmin-(dmax-dmin)/2
        self.renderer_softflf.rasterizer.far= dmax+(dmax-dmin)/2
        self.renderer_softflb.rasterizer.near=dmin-(dmax-dmin)/2
        self.renderer_softflb.rasterizer.far= dmax+(dmax-dmin)/2
        self.renderer_softtex.rasterizer.near=dmin-(dmax-dmin)/2
        self.renderer_softtex.rasterizer.far= dmax+(dmax-dmin)/2
        if opts.sigval!=1e-4:
            self.renderer_soft.rasterizer.sigma_val=   opts.sigval 
            self.renderer_softflf.rasterizer.sigma_val=opts.sigval 
            self.renderer_softflb.rasterizer.sigma_val=opts.sigval 
            self.renderer_softtex.rasterizer.sigma_val=opts.sigval 

        self.flow_fw, self.bgmask_fw, self.fgmask_flowf = render_flow_soft_2(self.renderer_softflf, verts_fl.view(2*local_batch_size,opts.n_hypo,-1,4)[:local_batch_size].view(-1,verts_fl.shape[1],4), 
                                              faces[:local_batch_size], 
                                              verts_pos0, verts_pos1, 
                                       ppoint[:,None][:local_batch_size].repeat(1,opts.n_hypo,1).view(-1,2),
                                       ppoint[:,None][local_batch_size:].repeat(1,opts.n_hypo,1).view(-1,2), 
                                        scale[:,None][:local_batch_size].view(-1,1),
                                        scale[:,None][local_batch_size:].view(-1,1))
        self.flow_bw, self.bgmask_bw, self.fgmask_flowb = render_flow_soft_2(self.renderer_softflb, verts_fl.view(2*local_batch_size,opts.n_hypo,-1,4)[local_batch_size:].view(-1,verts_fl.shape[1],4), 
                                              faces[local_batch_size:], 
                                              verts_pos1, verts_pos0, 
                                       ppoint[:,None][local_batch_size:].repeat(1,opts.n_hypo,1).view(-1,2),
                                       ppoint[:,None][:local_batch_size].repeat(1,opts.n_hypo,1).view(-1,2), 
                                        scale[:,None][local_batch_size:].view(-1,1),
                                        scale[:,None][:local_batch_size].view(-1,1))

        self.bgmask =  torch.cat([self.bgmask_fw, self.bgmask_bw],0) 
        self.fgmask_flow =  torch.cat([self.fgmask_flowf, self.fgmask_flowb],0) 
        self.flow_rd = torch.cat([self.flow_fw, self.flow_bw    ],0) 

#        torch.cuda.synchronize()
#        print('before rend + flow time:%.2f'%(time.time()-start_time))
              
        # 2) silhouette
        Rmat_mask = Rmat.clone().view(-1,opts.n_bones,3,3)
        Rmat_mask = torch.cat([Rmat_mask[:,:1].detach(), Rmat_mask[:,1:]],1).view(-1,3,3)
        verts_mask = obj_to_cam(pred_v, Rmat_mask, Tmat[:,np.newaxis,:],opts.n_bones, opts.n_hypo,skin)
        verts_mask = torch.cat([verts_mask,torch.ones_like(verts_mask[:, :, 0:1])], dim=-1)
        verts_mask = pinhole_cam(verts_mask, ppoint, scale)


        if opts.opt_tex=='yes':
            # 3) texture rendering
            Rmat_tex = Rmat.clone().view(2*local_batch_size,opts.n_hypo,opts.n_bones,3,3).view(-1,3,3)
            verts_tex = obj_to_cam(pred_v, Rmat_tex, Tmat[:,np.newaxis,:],opts.n_bones, opts.n_hypo,skin)
            verts_tex = torch.cat([verts_tex,torch.ones_like(verts_tex[:, :, 0:1])], dim=-1)
            verts_tex = pinhole_cam(verts_tex, ppoint, scale)
            offset = torch.Tensor( self.renderer_softtex.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
            verts_pre = verts_tex[:,:,:3]+offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
            self.renderer_softtex.rasterizer.background_color = [1,1,1]
            self.texture_render = self.renderer_softtex.render_mesh(sr.Mesh(verts_pre, faces[:,None].repeat(1,opts.n_hypo,1,1).view(-1,faces.shape[1],3), textures=tex,  texture_type=self.texture_type)).clone()
            self.mask_pred = self.texture_render[:,-1]
            self.fgmask_tex = self.texture_render[:,-1]
            self.texture_render = self.texture_render[:,:3]
            img_obs = self.imgs[:]*(self.masks[:]>0).float()[:,None]
            img_rnd = self.texture_render*(self.fgmask_tex)[:,None]
            img_white = 1-(self.masks[:]>0).float()[:,None] + img_obs

#        torch.cuda.synchronize()
#        print('before rend + flow + sil + tex time:%.2f'%(time.time()-start_time))              

        if opts.n_bones>1 and self.iters==0:
            # part rendering
            self.part_render = self.renderer_softpart.render_mesh(sr.Mesh(verts_pre.view(2*local_batch_size,opts.n_hypo,-1,3)[:1,self.optim_idx].detach(), faces[:1], textures=skin_colors[None], texture_type='vertex'))[:,:3].detach()
        
        # losses
        # 1) mask loss
        mask_pred = self.mask_pred.view(2*local_batch_size,-1,opts.img_size, opts.img_size)
        self.mask_loss_sub = (mask_pred - self.masks[:,None]).pow(2)

        #self.mask_loss_sub = 0
        #for i in range (5): # 256,128,64,32,16
        #    diff_img = (F.interpolate(mask_pred         , scale_factor=(0.5)**i,mode='area')
        #              - F.interpolate(self.masks[:,None], scale_factor=(0.5)**i,mode='area')).pow(2)
        #    self.mask_loss_sub += F.interpolate(diff_img, mask_pred.shape[2:4])
        #self.mask_loss_sub *= 0.2
        
        tmplist = torch.zeros(2*local_batch_size, opts.n_hypo).cuda()
        for i in range(2*local_batch_size):
            for j in range(opts.n_hypo):
                tmplist[i,j] = self.mask_loss_sub[i,j][self.occ[i]!=0].mean()
        self.mask_loss_sub = 0.5 * tmplist
        self.mask_loss = self.mask_loss_sub.mean()  # get rid of invalid pixels (out of border)
        self.total_loss = self.mask_loss.clone()
    
        # 2) flow loss
        flow_rd = self.flow_rd.view(2*local_batch_size,-1,opts.img_size, opts.img_size,2)
        mask = (~self.bgmask).view(2*local_batch_size,-1,opts.img_size, opts.img_size) & ((self.occ!=0)[:,None] &  (self.masks[:]>0) [:,None]).repeat(1,opts.n_hypo,1,1)
        self.flow_rd_map = torch.norm((flow_rd-self.flow[:,None,:2].permute(0,1,3,4,2)),2,-1)
        #self.flow_rd_map = 0
        #for i in range (5): # 256,128,64,32,16
        #    diff_img = torch.norm((F.interpolate(flow_rd,                                 scale_factor=((0.5)**i,(0.5)**i,1),mode='area')-
        #                           F.interpolate(self.flow[:,None,:2].permute(0,1,3,4,2), scale_factor=((0.5)**i,(0.5)**i,1),mode='area')),2,-1)
        #    self.flow_rd_map += F.interpolate(diff_img, flow_rd.shape[2:4])
        #self.flow_rd_map *= 0.2
        self.vis_mask = mask.clone()
        weights_flow = (-self.occ).sigmoid()[:,None].repeat(1,opts.n_hypo,1,1)
        for i in range(weights_flow.shape[0]):
            weights_flow[i] = weights_flow[i] / weights_flow[i][mask[i]].mean()
        self.flow_rd_map = self.flow_rd_map * weights_flow
    
        tmplist = torch.zeros(2*local_batch_size, opts.n_hypo).cuda()
        for i in range(2*local_batch_size):
            for j in range(opts.n_hypo):
                tmplist[i,j] = self.flow_rd_map[i,j][mask[i,j]].mean()
                if mask[i,j].sum()==0: tmplist[i,j]=0
        self.flow_rd_loss_sub = 0.5*tmplist
    
        self.flow_rd_loss = self.flow_rd_loss_sub.mean()
        self.total_loss += self.flow_rd_loss
    
        # 3) texture loss
        if opts.opt_tex=='yes':
            imgobs_rep = img_obs[:,None].repeat(1,opts.n_hypo,1,1,1).view(-1,3,opts.img_size,opts.img_size)
            imgwhite_rep = img_white[:,None].repeat(1,opts.n_hypo,1,1,1).view(-1,3,opts.img_size,opts.img_size)
            obspair = torch.cat([imgobs_rep, imgwhite_rep],0) 
            rndpair = torch.cat([img_rnd, self.texture_render],0) 
    
            tmplist = torch.zeros(2*local_batch_size, opts.n_hypo).cuda()
            for i in range(2*local_batch_size):
                for j in range(opts.n_hypo):
                    tmplist[i,j] +=  (img_obs[i] - img_rnd.view(2*local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j]).abs().mean(0)[self.occ[i]!=0].mean()  
                    tmplist[i,j] +=  (img_white[i] - self.texture_render.view(2*local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j]).abs().mean(0)[self.occ[i]!=0].mean()  
                    #tmplist[i,j] = 0
                    #for k in range(5):
                    #    diff_img = (F.interpolate(img_obs[i]                                                             ,scale_factor=(0.5)**i,mode='area')- 
                    #                F.interpolate(img_rnd.view(2*local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j],scale_factor=(0.5)**i,mode='area')
                    #                ).abs()
                    #    tmplist[i,j] += F.interpolate(diff_img[None], img_obs.shape[2:4])[0].mean(0)[self.occ[i]!=0].mean()
                    #    diff_img = (F.interpolate(img_white[i]                                                             ,scale_factor=(0.5)**i,mode='area')- 
                    #                F.interpolate(self.texture_render.view(2*local_batch_size,-1,3,opts.img_size, opts.img_size)[i,j],scale_factor=(0.5)**i,mode='area')
                    #                ).abs()
                    #    tmplist[i,j] += F.interpolate(diff_img[None], img_obs.shape[2:4])[0].mean(0)[self.occ[i]!=0].mean()
                    #tmplist[i,j] *= 0.2
                    tmplist[i,j] *= 2*opts.l1tex_wt
            percept_loss = self.ptex_loss.forward_pair(2*obspair-1, 2*rndpair-1)
            tmplist +=  0.005*percept_loss.view(2,-1).sum(0).view(local_batch_size*2,opts.n_hypo)
            self.texture_loss_sub = 0.25*tmplist
            self.texture_loss = self.texture_loss_sub.mean()
    
            self.total_loss += self.texture_loss
    
        # 4) shape smoothness/symmetry
        factor=int(opts.n_faces)/1280
        if opts.n_hypo>1:
            factor = 1 # possibly related to symmetry loss?
        else: 
            factor = reg_decay(self.epoch, opts.num_epochs, 0.05, 0.5)
        self.triangle_loss_sub = factor*0.005*self.triangle_loss_fn_sr(pred_v)*(4**opts.subdivide)/64.
        self.triangle_loss_sub +=factor*5e-4*self.flatten_loss(pred_v)*(2**opts.subdivide/8.0)
        self.triangle_loss_sub = self.triangle_loss_sub.view(2*local_batch_size,opts.n_hypo)
        self.triangle_loss = self.triangle_loss_sub.mean()
        self.total_loss += self.triangle_loss
        
        if (not opts.symmetric) and opts.symmetric_loss:
            # symmetry
            pointa = pred_v.view(2*local_batch_size, opts.n_hypo,-1,3)[0]
            pointb = torch.Tensor([[[-1,1,1]]]).cuda()*pointa
            mesha = pytorch3d.structures.meshes.Meshes(verts=pointa, faces=faces[:1].repeat(opts.n_hypo,1,1).detach())
            meshb = pytorch3d.structures.meshes.Meshes(verts=pointb, faces=faces[:1].repeat(opts.n_hypo,1,1).detach())
            pointa = pytorch3d.structures.Pointclouds(pointa)
            pointb = pytorch3d.structures.Pointclouds(pointb)
            fac=1
            self.total_loss += fac*pytorch3d.loss.point_mesh_face_distance(mesha, pointb)
            self.total_loss += fac*pytorch3d.loss.point_mesh_face_distance(meshb, pointa)
    
            if opts.opt_tex=='yes':
                # color
                pointa = pred_v[:1].detach()
                pointb = torch.Tensor([[[-1,1,1]]]).cuda()*pointa
                _,_,idx1,_ = self.chamLoss(pointa,pointb)
                self.total_loss += (self.tex[0][idx1[0].long()].detach() - self.tex[0]).abs().mean()*1e-3
    
        # 5) shape deformation loss
        if opts.n_bones>1:
            # bones
            self.bone_rot_l1 =  compute_geodesic_distance_from_two_matrices(
                        quat.view(-1,opts.n_hypo,opts.n_bones,9)[:,:,1:].reshape(-1,3,3), 
             torch.eye(3).cuda().repeat(2*local_batch_size*opts.n_hypo*(opts.n_bones-1),1,1)).mean() # small rotation
            self.bone_trans_l1 = torch.cat([trans,depth],1).view(-1,opts.n_hypo,opts.n_bones,3)[:,:,1:].abs().mean()
            if opts.n_hypo>1:
                factor=1
            else: 
                factor = reg_decay(self.epoch, opts.num_epochs, 0.05, 0.5)
            self.lmotion_loss_sub = factor*(self.deform_v - pred_v).norm(2,-1).mean(-1).view(2*local_batch_size,opts.n_hypo)
            self.lmotion_loss = self.lmotion_loss_sub.mean()
            self.total_loss += self.lmotion_loss
    
            # skins
            self.arap_loss = self.arap_loss_fn(self.deform_v[:local_batch_size*opts.n_hypo], self.deform_v[local_batch_size*opts.n_hypo:]).mean()*(4**opts.subdivide)/64.
            self.total_loss += self.arap_loss

        # 6) bone symmetry 
        if opts.n_bones>1 and opts.symmetric_loss:
           pointa = self.ctl_ts.view(opts.n_hypo, -1,3)
           pointb = torch.Tensor([[[-1,1,1]]]).cuda()*pointa
           self.total_loss += 0.1*pytorch3d.loss.chamfer_distance(pointa, pointb)[0]

        # 7) camera loss
        if opts.use_gtpose:
            self.cam_loss = compute_geodesic_distance_from_two_matrices(quat.view(-1,3,3), quat_pred.view(-1,3,3)).mean()
            self.cam_loss += (scale_pred - scale).abs().mean()
            self.cam_loss += (trans_pred - trans).abs().mean()
            self.cam_loss += (depth_pred - depth).abs().mean()
            self.cam_loss += (ppoint_pred - ppoint).abs().mean()
            self.cam_loss = 0.2 * self.cam_loss
        else:
            self.rotg_sm_sub = compute_geodesic_distance_from_two_matrices(quat.view(-1,opts.n_hypo,opts.n_bones,9)[:local_batch_size,:].view(-1,3,3),
                                                                            quat.view(-1,opts.n_hypo,opts.n_bones,9)[local_batch_size:,:].view(-1,3,3)).view(-1,opts.n_hypo,opts.n_bones)
            self.cam_loss =  0.001*self.rotg_sm_sub.mean()
            if opts.n_bones>1:
                self.cam_loss += 0.01*(trans.view(-1,opts.n_hypo,opts.n_bones,2)[:local_batch_size,:,1:] - 
                              trans.view(-1,opts.n_hypo,opts.n_bones,2)[local_batch_size:,:,1:]).abs().mean()
                self.cam_loss += 0.01*(depth.view(-1,opts.n_hypo,opts.n_bones,1)[:local_batch_size,:,1:] - 
                              depth.view(-1,opts.n_hypo,opts.n_bones,1)[local_batch_size:,:,1:]).abs().mean()
        self.total_loss += self.cam_loss

        # 8) aux losses
        # pull far away from the camera center
        self.total_loss += 0.02*F.relu(2-Tmat.view(-1, 1, opts.n_bones, 3)[:,:,:1,-1]).mean()
        if opts.n_bones>1:
            bone_loc_loss = 0.1* F.grid_sample(self.ddts_barrier.repeat(1,opts.n_hypo,1,1).view(-1,1,opts.img_size,opts.img_size), self.joints_proj[:,:,:2].view(-1,opts.n_bones-1,1,2),padding_mode='border').mean()
            ctl_loc_loss =  0.1* F.grid_sample(self.ddts_barrier.repeat(1,opts.n_hypo,1,1).view(-1,1,opts.img_size,opts.img_size), self.ctl_proj[:,:,:2].view(-1,opts.n_bones-1,1,2)   ,padding_mode='border').mean()
            self.total_loss += 100*(bone_loc_loss+ctl_loc_loss)
        

        aux_output={}
        aux_output['flow_rd_map'] = self.flow_rd_map
        aux_output['flow_rd'] = self.flow_rd
        aux_output['vis_mask'] = self.vis_mask
        aux_output['mask_pred'] = self.mask_pred
        aux_output['total_loss'] = self.total_loss
        aux_output['mask_loss'] = self.mask_loss
        aux_output['texture_loss'] = self.texture_loss
        aux_output['flow_rd_loss'] = self.flow_rd_loss
        aux_output['triangle_loss'] = self.triangle_loss
        if opts.n_bones>1:
            aux_output['lmotion_loss'] = self.lmotion_loss
        aux_output['current_nscore'] = self.texture_loss_sub.mean(0) + self.flow_rd_loss_sub.mean(0) + self.mask_loss_sub.mean(0)
        if opts.n_hypo > 1:
            for ihp in range(opts.n_hypo):
                aux_output['mask_hypo_%d'%(ihp)] = self.mask_loss_sub[:,ihp].mean()
                aux_output['flow_hypo_%d'%(ihp)] = self.flow_rd_loss_sub[:,ihp].mean()
                aux_output['tex_hypo_%d'%(ihp)] = self.texture_loss_sub[:,ihp].mean()
        try:
            aux_output['part_render'] = self.part_render
            aux_output['texture_render'] = self.texture_render
            aux_output['ctl_proj'] = self.ctl_proj
        except:pass
        return self.total_loss, aux_output
