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
Generic Training Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import time
import pdb
import numpy as np
from absl import flags

import cv2
import soft_renderer as sr
import subprocess
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributed as dist
import trimesh
from kmeans_pytorch import kmeans
import chamfer3D.dist_chamfer_3D

from ext_utils.flowlib import flow_to_image
from ext_nnutils.train_utils import Trainer
from nnutils.geom_utils import label_colormap
from ext_nnutils import loss_utils as ext_loss_utils
from nnutils import loss_utils
from nnutils import mesh_net
from nnutils import geom_utils
from dataloader import vid as vid_data


#-------------- flags -------------#
#----------------------------------#
## Flags for training
flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
flags.DEFINE_float('learning_rate', 0.0001, 'learning rate')
flags.DEFINE_integer('batch_size', 8, 'Size of minibatches')

## Flags for logging and snapshotting
flags.DEFINE_string('checkpoint_dir', './logdir',
                    'Root directory for output files')
flags.DEFINE_string('model_path', '',
                    'load model path')
flags.DEFINE_integer('save_epoch_freq', 1, 'save model every k epochs')


citylabs = label_colormap()

def add_image(log,tag,img,step,scale=True):
    timg = img[0]
    if scale:
        timg = (timg-timg.min())/(timg.max()-timg.min())

    if len(timg.shape)==2:
        formats='HW'
    elif timg.shape[0]==3:
        formats='CHW'
    else:
        formats='HWC'
    log.add_image(tag,timg,step,dataformats=formats)


class LASRTrainer(Trainer):
    def define_model(self):
        opts = self.opts
        self.symmetric = opts.symmetric

        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.LASR(
            img_size, opts, nz_feat=opts.nz_feat)
        
        if opts.model_path!='':
            self.load_network(self.model, model_path = opts.model_path)

        # ddp
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = self.model.to(device)

        self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[opts.local_rank],
                output_device=opts.local_rank,
                find_unused_parameters=True,
        )
        self.define_criterion_ddp()
        return

    def define_criterion_ddp(self):
        self.model.module.mask_loss_fn = torch.nn.MSELoss()
        mean_v, tex, faces = self.model.module.get_mean_shape(1)
        self.model.module.triangle_loss_fn_sr = ext_loss_utils.LaplacianLoss(mean_v[0].cpu(), faces[0].cpu()).cuda()
        self.model.module.arap_loss_fn = loss_utils.ARAPLoss(mean_v[0].cpu(), faces[0].cpu()).cuda()
        self.model.module.flatten_loss = ext_loss_utils.FlattenLoss(faces[0].cpu()).cuda()
        from PerceptualSimilarity.models import dist_model
        self.model.module.ptex_loss = dist_model.DistModel()
        self.model.module.ptex_loss.initialize(model='net', net='alex', use_gpu=False)
        self.model.module.ptex_loss.cuda(self.opts.local_rank)
        self.model.module.chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

    def set_input(self, batch):
        opts = self.opts
        self.model.module.is_canonical = torch.cat([batch['is_canonical'][:opts.batch_size], batch['is_canonicaln'][:opts.batch_size]],0)
        self.model.module.frameid = torch.cat([    batch['id0'][:opts.batch_size], batch['id1'][:opts.batch_size]],0)

        # Image with annotations.
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        input_imgn_tensor = batch['imgn'].type(torch.FloatTensor)
        for b in range(input_imgn_tensor.size(0)):
            input_imgn_tensor[b] = self.resnet_transform(input_imgn_tensor[b])
        input_img_tensor = torch.cat([input_img_tensor, input_imgn_tensor],0)
        self.model.module.input_imgs = input_img_tensor.cuda()

        img_tensor = batch['img'].type(torch.FloatTensor)
        imgn_tensor = batch['imgn'].type(torch.FloatTensor)
        img_tensor = torch.cat([            img_tensor, imgn_tensor      ],0)
        self.model.module.imgs = img_tensor.cuda()

        shape = img_tensor.shape[2:]

        cam_tensor = batch['cam'].type(torch.FloatTensor)
        camn_tensor = batch['camn'].type(torch.FloatTensor)
        depth_tensor = batch['depth'].type(torch.FloatTensor)
        depthn_tensor = batch['depthn'].type(torch.FloatTensor)
        cam_tensor = torch.cat([            cam_tensor, camn_tensor      ],0)
        depth_tensor = torch.cat([        depth_tensor,depthn_tensor     ],0)
        self.model.module.cams = cam_tensor.cuda()
        self.model.module.depth_gt = depth_tensor.cuda()

        flow_tensor = batch['flow'].type(torch.FloatTensor)
        flown_tensor = batch['flown'].type(torch.FloatTensor)
        self.model.module.flow = torch.cat([flow_tensor, flown_tensor],0).cuda()
        self.model.module.occ = torch.cat([  batch['occ'].type(torch.FloatTensor),
                                batch['occn'].type(torch.FloatTensor),],0).cuda()
        self.model.module.oriimg_shape = batch['shape'][:1,:2].repeat(opts.batch_size*2,1).cuda()
                            
        batch_input = {}
        batch_input['input_imgs  '] = self.model.module.input_imgs  
        batch_input['imgs        '] = self.model.module.imgs        
        batch_input['masks       '] = batch['mask'].type(torch.FloatTensor).cuda().permute(1,0,2,3).reshape(opts.batch_size*2,shape[0],shape[1])
        batch_input['cams        '] = self.model.module.cams        
        batch_input['depth_gt    '] = self.model.module.depth_gt    
        batch_input['flow        '] = self.model.module.flow        
        batch_input['dts_barrier '] = batch['mask_dts'].type(torch.FloatTensor).cuda().permute(1,0,2,3).reshape(opts.batch_size*2,1,shape[0],shape[1])
        batch_input['ddts_barrier'] = batch['dmask_dts'].type(torch.FloatTensor).cuda().permute(1,0,2,3).reshape(opts.batch_size*2,1,shape[0],shape[1])
        batch_input['mask_contour'] = batch['mask_contour'].type(torch.FloatTensor).cuda().permute(1,0,2,3).reshape(opts.batch_size*2,1,1000,2)
        batch_input['pp          '] = batch['pps'].type(torch.FloatTensor).cuda().permute(1,0,2).reshape(opts.batch_size*2,-1) # bs, 2, x 
        batch_input['occ         '] = self.model.module.occ              
        batch_input['oriimg_shape'] = self.model.module.oriimg_shape     
        batch_input['is_canonical'] = self.model.module.is_canonical 
        batch_input['frameid']      = self.model.module.frameid
        batch_input['dataid']      = torch.cat([    batch['dataid'][:opts.batch_size], batch['dataid'][:opts.batch_size]],0)
        for k,v in batch_input.items():
            batch_input[k] = v.view(2,self.opts.batch_size,-1).permute(1,0,2).reshape(v.shape)
        return batch_input 

    def init_training(self):
        opts = self.opts
        self.init_dataset()    
        self.define_model()
        new_params=[]
        for name,p in self.model.module.named_parameters():
            if name == 'mean_v': print('found mean v'); continue
            if name == 'rotg': print('found rotg'); continue
            if name == 'rots': print('found rots'); continue
            if name == 'focal': print('found fl');continue
            if name == 'pps': print('found pps');continue
            if name == 'tex': print('found tex');continue
            if name == 'texhr': print('found texhr');continue
            if name == 'body_score': print('found body_score');continue
            if name == 'skin': print('found skin'); continue
            if name == 'rest_rs': print('found rest rotation'); continue
            if name == 'ctl_rs': print('found ctl rotation'); continue
            if name == 'rest_ts': print('found rest translation'); continue
            if name == 'ctl_ts': print('found ctl points'); continue
            if name == 'transg': print('found global translation'); continue
            if name == 'log_ctl': print('found log ctl'); continue
            new_params.append(p)
        self.optimizer = torch.optim.AdamW(
            [{'params': new_params},
             {'params': self.model.module.mean_v, 'lr':50*opts.learning_rate},
             {'params': self.model.module.tex, 'lr':50*opts.learning_rate},
             {'params': self.model.module.ctl_rs, 'lr':50*opts.learning_rate},
             {'params': self.model.module.rest_ts, 'lr':50*opts.learning_rate},
             {'params': self.model.module.ctl_ts, 'lr':50*opts.learning_rate},
             {'params': self.model.module.log_ctl, 'lr':50*opts.learning_rate}
            ],
            lr=opts.learning_rate,betas=(0.9, 0.999),weight_decay=1e-4)
        lr_meanv = 50*opts.learning_rate
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
        [opts.learning_rate,
        lr_meanv,
        50*opts.learning_rate, # tex
        50*opts.learning_rate, # ctl rs 
        50*opts.learning_rate, # rest ts 
        50*opts.learning_rate, # ctl ts 
        50*opts.learning_rate, # log ctl
        ],
        200*len(self.dataloader), pct_start=0.01, cycle_momentum=False, anneal_strategy='linear',final_div_factor=1./25)

    def train(self):
        opts = self.opts
        if opts.local_rank==0:
            log = SummaryWriter('%s/%s'%(opts.checkpoint_dir,opts.name), comment=opts.name)
        total_steps = 0
        dataset_size = len(self.dataloader)
        torch.manual_seed(8)  # do it again
        torch.cuda.manual_seed(1)
        self.epoch_nscore = torch.zeros(self.opts.n_hypo).cuda()
        if opts.local_rank==0:        self.save('0')

        for epoch in range(0, opts.num_epochs):
            self.model.module.epoch = epoch
            epoch_iter = 0

            # reinit bones
            if self.opts.local_rank==0 and epoch==0 and self.opts.n_bones>1:
                for hypo_idx in range(self.opts.n_hypo):
                    
                    cluster_ids_x, cluster_centers = kmeans(
                    X=self.model.module.symmetrize(self.model.module.mean_v[hypo_idx]), num_clusters=self.opts.n_bones-1, distance='euclidean', device=torch.device('cuda:%d'%(opts.local_rank)))
                    self.model.module.rest_ts.data[hypo_idx*(self.opts.n_bones-1):(hypo_idx+1)*(self.opts.n_bones-1)] = cluster_centers.cuda()
                    self.model.module.ctl_ts .data[hypo_idx*(self.opts.n_bones-1):(hypo_idx+1)*(self.opts.n_bones-1)] = cluster_centers.cuda()
                    self.model.module.ctl_rs.data = torch.Tensor([[0,0,0,1]]).repeat(opts.n_hypo*(opts.n_bones-1),1).cuda()
                    self.model.module.log_ctl.data= torch.Tensor([[1,1,1]]).cuda().repeat(opts.n_hypo*(opts.n_bones-1),1)  # control point varuance
            dist.barrier()
            dist.broadcast(self.model.module.ctl_ts, 0)
            dist.broadcast(self.model.module.rest_ts, 0)
            dist.broadcast(self.model.module.ctl_rs, 0)
            dist.broadcast(self.model.module.log_ctl, 0)
            print('new bone locations')
 
            # modify dataset
            optim_cam = (-self.epoch_nscore).argmax()
            self.model.module.optim_idx = optim_cam
            if self.opts.local_rank==0:
                print('scores:')
                print(self.epoch_nscore)
                print('selecting %d'%optim_cam)
            self.epoch_nscore[:] = 0
            for i, batch in enumerate(self.dataloader):
                self.model.module.iters=i
                input_batch = self.set_input(batch)

#               torch.cuda.synchronize()
#               start_time = time.time()

                self.model.module.total_steps = total_steps
                self.optimizer.zero_grad()
                total_loss,aux_output = self.model(input_batch)
                total_loss.mean().backward()

#                torch.cuda.synchronize()
#                print('forward back time:%.2f'%(time.time()-start_time))

                cam_grad = []
                for name,p in self.model.module.named_parameters():
                    if 'mean_v' == name and p.grad is not None:
                        torch.nn.utils.clip_grad_norm_(p, 1.)
                        self.grad_meanv_norm = p.grad.view(-1).norm(2,-1)
                    elif p.grad is not None and ('code_predictor' in name or 'encoder' in name):
                        cam_grad.append(p)
                    if (not p.grad is None) and (torch.isnan(p.grad).sum()>0):
                        self.optimizer.zero_grad()
                self.grad_cam_norm = torch.nn.utils.clip_grad_norm_(cam_grad, 10.)

                if opts.local_rank==0 and torch.isnan(self.model.module.total_loss):
                    pdb.set_trace()
                self.optimizer.step()
                self.scheduler.step()

                total_steps += 1
                epoch_iter += 1

                if opts.local_rank==0:
                    if i==0:
                        gu = np.asarray(self.model.module.flow[0,0,:,:].detach().cpu()) 
                        gv = np.asarray(self.model.module.flow[0,1,:,:].detach().cpu())
                        mask = aux_output['vis_mask'][:,optim_cam]
                        mask = np.asarray(mask[0].float().cpu())
                        gu[~mask.astype(bool)] = 0.;  gv[~mask.astype(bool)] = 0.
                        add_image(log,'train/flowobs', flow_to_image(np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis],mask[:,:,np.newaxis]),-1))[np.newaxis],epoch)
                        gu = np.asarray(aux_output['flow_rd'].view(2*opts.batch_size,opts.n_hypo,opts.img_size,opts.img_size,2)[0,optim_cam,:,:,0].detach().cpu())
                        gv = np.asarray(aux_output['flow_rd'].view(2*opts.batch_size,opts.n_hypo,opts.img_size,opts.img_size,2)[0,optim_cam,:,:,1].detach().cpu())
                        gu[~mask.astype(bool)] = 0.;  gv[~mask.astype(bool)] = 0.
                        add_image(log,'train/flowrd', flow_to_image(np.concatenate((gu[:,:,np.newaxis],gv[:,:,np.newaxis],mask[:,:,np.newaxis]),-1))[np.newaxis],epoch)
                        error = np.asarray(aux_output['flow_rd_map'][:1,optim_cam].detach().cpu()); error=error*mask
                        add_image(log,'train/flow_error', opts.img_size*error,epoch)

                        add_image(log,'train/mask', 255*np.asarray(aux_output['mask_pred'][optim_cam:optim_cam+1].detach().cpu()),epoch)
                        if opts.n_bones>1:
                            add_image(log,'train/part', np.asarray(255*aux_output['part_render'][:1].detach().cpu().permute(0,2,3,1), dtype=np.uint8),epoch)
                        add_image(log,'train/maskgt', 255*np.asarray(self.model.module.masks[:1].detach().cpu()),epoch)
                        img1_j = np.asarray(255*self.model.module.imgs[:1].permute(0,2,3,1).detach().cpu()).astype(np.uint8)
                        add_image(log,'train/img1', img1_j,epoch      ,scale=False)
                        add_image(log,'train/img2', np.asarray(255*self.model.module.imgs[opts.batch_size:opts.batch_size+1].permute(0,2,3,1).detach().cpu()).astype(np.uint8),epoch, scale=False)

                        if 'texture_render' in aux_output.keys():
                            texture_j = np.asarray(aux_output['texture_render'][optim_cam:optim_cam+1].permute(0,2,3,1).detach().cpu()*255,dtype=np.uint8)
                            if opts.n_bones>1:
                                for k in range(aux_output['ctl_proj'].shape[1]):
                                    texture_j[0] = cv2.circle(texture_j[0].copy(),tuple(128+128*np.asarray(aux_output['ctl_proj'][optim_cam].detach().cpu())[k,:2]),3,citylabs[k].tolist(),3)
                            add_image(log,'train/texture', texture_j,epoch,scale=False)
                    log.add_scalar('train/total_loss',  aux_output['total_loss'].mean()  , total_steps)
                    if 'mask_loss' in aux_output.keys():
                        log.add_scalar('train/mask_loss' ,  aux_output['mask_loss'].mean()   , total_steps)
                    log.add_scalar('train/flow_rd_loss',aux_output['flow_rd_loss'].mean(), total_steps)
                    log.add_scalar('train/texture_loss',aux_output['texture_loss'].mean(), total_steps)
                    if opts.n_hypo > 1:
                        for ihp in range(opts.n_hypo):
                            log.add_scalar('train/mask_hypo_%d'%ihp,aux_output['mask_hypo_%d'%ihp].mean()   , total_steps)
                            log.add_scalar('train/flow_hypo_%d'%ihp,aux_output['flow_hypo_%d'%ihp].mean(), total_steps)
                            log.add_scalar('train/tex_hypo_%d'%ihp,aux_output['tex_hypo_%d'%ihp].mean(), total_steps)
                    log.add_scalar('train/triangle_loss',aux_output['triangle_loss'], total_steps)
                    if 'lmotion_loss' in aux_output.keys():
                        log.add_scalar('train/lmotion_loss', aux_output['lmotion_loss'], total_steps)
                    if hasattr(self, 'grad_meanv_norm'): log.add_scalar('train/grad_meanv_norm',self.grad_meanv_norm, total_steps)
                    if hasattr(self, 'grad_cam_norm'):log.add_scalar('train/grad_cam_norm',self.grad_cam_norm, total_steps)
                    if i>100:
                        self.epoch_nscore += aux_output['current_nscore']
                        
                    if hasattr(self.model.module, 'sampled_img_obs_vis'):
                        if i%10==0:
                            add_image(log,'train/sampled_img_obs_vis', np.asarray(255*self.model.module.sampled_img_obs_vis[0:1, optim_cam].detach().cpu()).astype(np.uint8),epoch, scale=False)
                            add_image(log,'train/sampled_img_rdc_vis', np.asarray(255*self.model.module.sampled_img_rdc_vis[0:1, optim_cam].detach().cpu()).astype(np.uint8),epoch, scale=False)
                            add_image(log,'train/sampled_img_rdf_vis', np.asarray(255*self.model.module.sampled_img_rdf_vis[0:1, optim_cam].detach().cpu()).astype(np.uint8),epoch, scale=False)
                        log.add_scalar('train/coarse_loss',self.model.module.coarse_loss, total_steps)
                        log.add_scalar('train/sil_coarse_loss',self.model.module.sil_coarse_loss, total_steps)
                        log.add_scalar('train/fine_loss',self.model.module.fine_loss, total_steps)

            if (epoch+1) % opts.save_epoch_freq == 0:
                print('saving the model at the end of epoch {:d}, iters {:d}'.format(epoch, total_steps))
                self.save('latest')
                self.save(epoch+1)
    
    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, local_rank=None):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = network.state_dict()

        if 'latest' not in save_path:
            save_dict = {k:v for k,v in save_dict.items() if 'uncertainty_predictor' not in k}
        save_dict['faces'] = self.model.module.faces.cpu()
        optim_cam = (-self.epoch_nscore).argmax()
        save_dict['full_shape'] = [self.model.module.symmetrize(   i.cuda()).cpu() for i in self.model.module.mean_v][optim_cam]
        save_dict['full_tex'] =   [self.model.module.symmetrize_color(i.cuda()).cpu()  for i in  self.model.module.tex][optim_cam]
        save_dict['epoch_nscore'] = self.epoch_nscore
        torch.save(save_dict, save_path)
        #if local_rank is not None and torch.cuda.is_available():
        #    network.cuda(device=local_rank)
        return

    # helper loading function that can be used by subclasses
    def load_network(self, network, model_path=None):
        save_path = model_path
        pretrained_dict = torch.load(save_path,map_location='cpu')
        
        states = pretrained_dict
        score_cams = -states['epoch_nscore']
        if self.opts.n_hypo<len(score_cams):  # select hypothesis
            optim_cam = score_cams.argmax()
            print('selecting hypothesis #%d'%optim_cam)
            nfeat=states['code_predictor.quat_predictor.pred_layer.weight'].shape[-1]
            quat_weights = states['code_predictor.quat_predictor.pred_layer.weight'].view(len(score_cams),-1,nfeat)
            quat_bias = states['code_predictor.quat_predictor.pred_layer.bias'].view(len(score_cams),-1)
            states['code_predictor.quat_predictor.pred_layer.weight'] = quat_weights[optim_cam]
            states['code_predictor.quat_predictor.pred_layer.bias'] = quat_bias[optim_cam]
            
            scale_weights = states['code_predictor.scale_predictor.pred_layer.weight'].view(len(score_cams),-1,nfeat)
            scale_bias =    states['code_predictor.scale_predictor.pred_layer.bias'].view(len(score_cams),-1)
            states['code_predictor.scale_predictor.pred_layer.weight'] = scale_weights[optim_cam]
            states['code_predictor.scale_predictor.pred_layer.bias'] =   scale_bias[optim_cam]
            
            states['mean_v'] = states['mean_v'][optim_cam:optim_cam+1]
            if 'tex' in states.keys():
                states['tex'] = states['tex'][optim_cam:optim_cam+1]
            else:
                states['tex'] = self.model.tex
            states['score_cams'] = score_cams[optim_cam:optim_cam+1]
            if states['mean_v'].shape[1] < states['faces'].max():
                states['mean_v'] = states['full_shape'][None]
                states['tex'] = states['full_tex'][None]
            
            if 'ctl_rs' in states.keys():
                states['ctl_rs'] =   states['ctl_rs'].view(score_cams.shape[0],-1,4)[optim_cam]
                states['rest_ts'] = states['rest_ts'].view(score_cams.shape[0],-1,3)[optim_cam]
                states['ctl_ts'] =   states['ctl_ts'].view(score_cams.shape[0],-1,3)[optim_cam]
                states['log_ctl'] = states['log_ctl'].view(score_cams.shape[0],-1,3)[optim_cam]

        pretrained_dict = states

        if (not self.opts.symmetric) and (int(self.opts.n_faces)!=states['faces'].shape[0]):
            sr.Mesh(states['mean_v'], states['faces']).save_obj('tmp/input-%d.obj'%(self.opts.local_rank))
            import subprocess
            print(subprocess.check_output(['./Manifold/build/manifold', 'tmp/input-%d.obj'%(self.opts.local_rank), 'tmp/output-%d.obj'%(self.opts.local_rank), '10000']))
            print(subprocess.check_output(['./Manifold/build/simplify', '-i', 'tmp/output-%d.obj'%(self.opts.local_rank), '-o', 'tmp/simple-%d.obj'%(self.opts.local_rank), '-m', '-f', self.opts.n_faces]))
            # load remeshed 
            loadmesh = sr.Mesh.from_obj('tmp/simple-%d.obj'%(self.opts.local_rank))
            mean_shape = loadmesh.vertices[0]
            self.model.faces  = loadmesh.faces[0]
            tex = torch.zeros(1,mean_shape.shape[0],3)
        else: 
            if self.opts.symmetric:
                mean_shape = self.model.symmetrize(states['mean_v'].cuda()[0])
                self.model.faces = states['faces'].cuda()
                if 'tex' in states.keys():                tex = states['tex']
            elif self.opts.load_mesh=='':
                mean_shape = states['mean_v'].cuda()[0]
                self.model.faces = states['faces'].cuda()
                if 'tex' in states.keys(): tex = states['tex']
                if self.model.faces.max()>mean_shape.shape[0]: 
                    mean_shape = states['full_shape']
                    tex = states['full_tex'][None]
            else:
                mean_shape = self.model.mean_v.data[0]
                tex = self.model.tex.data

        if self.opts.symmetric:
            self.model.mean_v.data = states['mean_v']
        else:
            self.model.mean_v.data = mean_shape.cpu()[None]
        
        try: self.model.tex.data    = tex; del states['tex']; 
        except:pass
        del states['mean_v']; 
        self.triangle_loss_fn_sr = ext_loss_utils.LaplacianLoss(mean_shape.cpu(), self.model.faces.cpu()).cuda()
        self.arap_loss_fn = loss_utils.ARAPLoss(            mean_shape.cpu(), self.model.faces.cpu()).cuda()
        if states['code_predictor.depth_predictor.pred_layer.bias'].shape[0] != self.opts.n_bones:  # from rigid body to deformable
            nfeat = states['code_predictor.quat_predictor.pred_layer.weight'].shape[-1]
            quat_weights = torch.cat( [states['code_predictor.quat_predictor.pred_layer.weight'].view(-1,4,nfeat)[:1], self.model.code_predictor.quat_predictor.pred_layer.weight.view(self.opts.n_bones,4,-1)[1:]],0).view(self.opts.n_bones*4,-1)
            quat_bias =    torch.cat( [states['code_predictor.quat_predictor.pred_layer.bias'].view(-1,4)[:1],         self.model.code_predictor.quat_predictor.pred_layer.bias.view(self.opts.n_bones,-1)[1:]],0).view(-1)
            states['code_predictor.quat_predictor.pred_layer.weight'] = quat_weights
            states['code_predictor.quat_predictor.pred_layer.bias'] = quat_bias
            
            tmp_weights = torch.cat( [states['code_predictor.trans_predictor.pred_layer.weight'].view(-1,2,nfeat)[:1], self.model.code_predictor.trans_predictor.pred_layer.weight.view(self.opts.n_bones,2,-1)[1:]],0).view(self.opts.n_bones*2,-1)
            tmp_bias =    torch.cat( [states['code_predictor.trans_predictor.pred_layer.bias'].view(-1,2)[:1],         self.model.code_predictor.trans_predictor.pred_layer.bias.view(self.opts.n_bones,-1)[1:]],0).view(-1)
            states['code_predictor.trans_predictor.pred_layer.weight'] = tmp_weights
            states['code_predictor.trans_predictor.pred_layer.bias'] =   tmp_bias
            
            tmp_weights = torch.cat( [states['code_predictor.depth_predictor.pred_layer.weight'].view(-1,1,nfeat)[:1], self.model.code_predictor.depth_predictor.pred_layer.weight.view(self.opts.n_bones,1,-1)[1:]],0).view(self.opts.n_bones*1,-1)
            tmp_bias =    torch.cat( [states['code_predictor.depth_predictor.pred_layer.bias'].view(-1,1)[:1],         self.model.code_predictor.depth_predictor.pred_layer.bias.view(self.opts.n_bones,-1)[1:]],0).view(-1)
            states['code_predictor.depth_predictor.pred_layer.weight'] = tmp_weights
            states['code_predictor.depth_predictor.pred_layer.bias'] =   tmp_bias

            # initialize skin based on mean shape 
            np.random.seed(18)
            if self.opts.n_bones>2:
                cluster_ids_x, cluster_centers = kmeans(
                X=mean_shape, num_clusters=self.opts.n_bones-1, distance='euclidean')
            else:
                cluster_centers = mean_shape.mean(0)[None]
            print('centers:')
            print(cluster_centers)
            states['rest_ts'] = cluster_centers.cuda()
            states['ctl_ts'] = cluster_centers.cuda()
            states['ctl_rs'] = self.model.ctl_rs
            states['log_ctl'] = self.model.log_ctl

        network.load_state_dict(pretrained_dict,strict=False)
        return
