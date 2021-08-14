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
Takes an image, returns stuff.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import scipy.misc
import torch
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import scipy.io as sio

from nnutils import mesh_net
from nnutils import geom_utils
import pdb
import kornia
import soft_renderer as sr
from nnutils.geom_utils import label_colormap
import configparser
citylabs = label_colormap()
import trimesh

from pytorch3d.renderer.mesh import TexturesAtlas, TexturesUV, TexturesVertex
from pytorch3d.structures.meshes import Meshes

# These options are off by default, but used for some ablations reported.
flags.DEFINE_boolean('dynamic', False, 'Use dynamic shape for prediction')

class MeshPredictor(object):
    def __init__(self, opts):
        self.opts = opts
        self.symmetric = opts.symmetric

        img_size = (opts.img_size, opts.img_size)
        print('Setting up model..')
        self.model = mesh_net.LASR(img_size, opts, nz_feat=opts.nz_feat)

        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.model.eval()
        self.model = self.model.cuda()

        self.renderer_softtex = sr.SoftRenderer(image_size=opts.img_size,  
                       camera_mode='look_at',perspective=False,
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
        self.renderer_softpart = sr.SoftRenderer(image_size=opts.img_size, sigma_val=1e-4,gamma_val=1e-4, 
                       camera_mode='look_at',perspective=False, aggr_func_rgb='softmax',
                       light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)

        # pytorch3d 
        from pytorch3d.renderer.mesh.shader import (
             BlendParams,
         )
        from pytorch3d.renderer import (
         PointLights, 
         RasterizationSettings, 
         MeshRenderer, 
         MeshRasterizer,  
         SoftPhongShader,
         SoftSilhouetteShader,
         )
        from pytorch3d.renderer.cameras import OrthographicCameras
        device = torch.device("cuda:0") 
        cameras = OrthographicCameras(device = device)
        lights = PointLights(
            device=device,
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((1.0, 1.0, 1.0),),
            specular_color=((1.0, 1.0, 1.0),),
        )
        self.renderer_pyr = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(image_size=opts.img_size,cull_backfaces=True)),
                shader=SoftPhongShader(device = device,cameras=cameras, lights=lights, blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)))
        )

        faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        network_dir = os.path.join(self.opts.checkpoint_dir, self.opts.name)
        save_path = os.path.join(network_dir, save_filename)
        print('loading {}..'.format(save_path))
        states = torch.load(save_path)
        score_cams = states['epoch_nscore']
        print(score_cams)
        optim_cam = (-score_cams).argmax()
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
        
        states['score_cams'] = score_cams[optim_cam:optim_cam+1]
        if 'tex' in states.keys():
            states['tex'] = states['tex'][optim_cam:optim_cam+1]
            self.model.tex.data = states['tex']
            del states['tex']

        # save all hypos
        for i,verts in enumerate(states['mean_v']):
            trimesh.Trimesh(vertices=np.asarray(self.model.symmetrize(verts).cpu()), faces=np.asarray(states['faces'])).export('tmp/%d.ply'%i)
            

        states['mean_v'] = states['mean_v'][optim_cam:optim_cam+1]
        if 'faces' in states.keys():
            network.faces = states['faces'].cuda()

        if 'ctl_rs' in states.keys():
            states['ctl_rs'] =   states['ctl_rs'].view(score_cams.shape[0],self.opts.n_bones-1,-1)[optim_cam]
            states['rest_ts'] = states['rest_ts'].view(score_cams.shape[0],self.opts.n_bones-1,-1)[optim_cam]
            states['ctl_ts'] =   states['ctl_ts'].view(score_cams.shape[0],self.opts.n_bones-1,-1)[optim_cam]
            states['log_ctl'] = states['log_ctl'].view(score_cams.shape[0],self.opts.n_bones-1,-1)[optim_cam]

        # delete unused vars
        # load mesh
        self.model.mean_v.data = states['mean_v']
        del states['mean_v']

        network.load_state_dict(states, strict=False)

        

        return

    def set_input(self, batch):
        opts = self.opts

        # original image where texture is sampled from.
        img_tensor = batch['img'].clone().type(torch.FloatTensor)

        # input_img is the input to resnet
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs = Variable(
            input_img_tensor.cuda(), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(), requires_grad=False)

    def predict(self, batch,alp,pp,frameid):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        self.frameid = frameid
        self.cams = torch.Tensor(np.asarray([[1./alp]])).cuda()
        self.pps = torch.Tensor(np.asarray([pp])).cuda()
        self.forward()
        return self.collect_outputs()

    def forward(self):
        opts = self.opts
        self.model.frameid=torch.Tensor([self.frameid]).long().cuda()
        pred_codes = self.model(self.input_imgs)
        scale, trans, quat, depth, ppoint = pred_codes
        self.uncrop_scale = scale.clone()[:,:,None].repeat(1,1,2) * 128
        self.uncrop_pp = ((ppoint + 1)*128/self.cams + self.pps)[0]
        scale = scale * self.cams[:,:1]
        depth[:,:1] = self.cams[:,:1]* depth[:,:1]; depth = depth.view(-1,1)

        quat = kornia.rotation_matrix_to_quaternion(quat.view(-1,3,3))
        quat = torch.cat([quat[:,3:],quat[:,:3]],1)
        self.cam_pred = torch.cat([scale.repeat(1,opts.n_bones).view(-1,1), trans, quat], 1)
        
        self.depth = depth
        self.ppoint = ppoint

        # Deform mean shape:
        self.pred_v, self.tex, self.faces = self.model.get_mean_shape(1)
        self.pred_v = self.pred_v[:1]
        self.tex = self.tex[:1]
        self.faces = self.faces[:1]

        if self.opts.dynamic:
            self.pred_v = self.pred_v + del_v

        print('focal: %f / depth: %f / ppx: %f / ppy: %f'%(self.cam_pred[0,0],self.depth[0,0], self.ppoint[0,0], self.ppoint[0,1]))
       
        proj_cam = self.cam_pred 
        config = configparser.RawConfigParser()
        config.read('configs/%s.config'%self.opts.dataname)
        canonical_frame = int(config.get('data', 'can_frame'))

        tmp_proj = proj_cam.clone()
        pidx=None
        self.pidx = pidx
        if pidx is not None: 
            proj_cam[:,3]=1; proj_cam[:,4:]=0; proj_cam[:,1:3]=0; self.depth[1:]=0.; scale[:] = 10; self.depth[0:1]=10; ppoint[:]=0
            proj_cam[pidx,1:]=tmp_proj[pidx,1:]
     
        from nnutils.geom_utils import pinhole_cam, obj_to_cam
       
        Rmat = kornia.quaternion_to_rotation_matrix(torch.cat((-proj_cam[:,4:],proj_cam[:,3:4]),1))
        Tmat = torch.cat([proj_cam[:,1:3],self.depth],1)
        if opts.n_bones>1:
            dis_norm = (self.model.ctl_ts[:,None] - self.pred_v[:1].detach()) # p-v, J,1,3 - 1,N,3
            dis_norm = dis_norm.matmul(kornia.quaternion_to_rotation_matrix(self.model.ctl_rs))
            dis_norm = self.model.log_ctl.exp()[:,None] * dis_norm.pow(2) # (p-v)^TS(p-v)
            skin = (-10 * dis_norm.sum(2)).softmax(0)[None,:,:,None]
            # create vis for skins
            sphere_list = []
            sphere = trimesh.creation.uv_sphere(radius=0.05,count=[16, 16])
            for i in range(opts.n_bones-1):
                sphere_verts = sphere.vertices
                sphere_verts = sphere_verts / np.asarray((0.5*self.model.log_ctl.clamp(-2,2)).exp()[i,None].cpu())
                sphere_verts = sphere_verts.dot(np.asarray(kornia.quaternion_to_rotation_matrix(self.model.ctl_rs[i]).cpu()).T)
                sphere_verts = sphere_verts+np.asarray(self.model.ctl_ts[i,None].cpu())
                sphere_list.append( trimesh.Trimesh(vertices = sphere_verts, faces=sphere.faces) )
            self.sphere = trimesh.util.concatenate(sphere_list)
            # skin
            dis_norm = (self.model.ctl_ts[:,None] - torch.Tensor(self.sphere.vertices)[None].cuda()) # p-v, J,1,3 - 1,N,3
            dis_norm = dis_norm.matmul(kornia.quaternion_to_rotation_matrix(self.model.ctl_rs))
            dis_norm = self.model.log_ctl.exp()[:,None] * dis_norm.pow(2) # (p-v)^TS(p-v)
            self.gauss_skin = (-10 * dis_norm.sum(2)).softmax(0)[None,:,:,None]

            # Gg*Gx*Gg_inv
            rest_rs = self.model.rest_rs.data; rest_rs[:,-1] =1; rest_rs[:,:3] =0  # identical rest R
            rest_rs = kornia.quaternion_to_rotation_matrix(rest_rs)
            rest_ts = self.model.rest_ts

            # part transform
            Rmat = Rmat.view(-1,opts.n_bones,3,3)
            Tmat = Tmat.view(-1,opts.n_bones,3,1)
            Tmat[:,1:] = -Rmat[:,1:].matmul(rest_ts[None,:,:,None])+Tmat[:,1:]+rest_ts[None,:,:,None]
            Rmat[:,1:] = Rmat[:,1:].permute(0,1,3,2)
            Rmat = Rmat.view(-1,3,3)
            Tmat = Tmat.view(-1,3)
        else:skin=None

        verts = obj_to_cam(self.pred_v, Rmat, Tmat[:,np.newaxis,:], opts.n_bones,opts.n_hypo,skin)
        if opts.n_bones>1:
            # joints
            # joints to skin
            self.joints_proj = obj_to_cam(self.model.ctl_ts[None].repeat(2*opts.batch_size,1,1), Rmat, Tmat[:,np.newaxis], opts.n_bones,opts.n_hypo, torch.eye(opts.n_bones-1)[None,:,:,None].cuda())
            self.joints_proj = pinhole_cam(self.joints_proj, ppoint, scale)[0]
            self.bones_3d = obj_to_cam(self.model.ctl_ts[None].repeat(2*opts.batch_size,1,1), Rmat, Tmat[:,np.newaxis], opts.n_bones,opts.n_hypo, torch.eye(opts.n_bones-1)[None,:,:,None].cuda(),tocam=False)
            self.nsphere_verts = self.sphere.vertices.shape[0] // (opts.n_bones-1)
            self.gaussian_3d = obj_to_cam(torch.Tensor(self.sphere.vertices).cuda()[None], Rmat, Tmat[:,np.newaxis], opts.n_bones, opts.n_hypo,
                                torch.eye(opts.n_bones-1)[None].repeat(self.nsphere_verts,1,1).permute(1,2,0).reshape(1,opts.n_bones-1,-1,1).cuda(),tocam=False)
        else:
            self.joints_proj = torch.zeros(0,3).cuda()
            self.bones_3d = None

        self.verts = obj_to_cam(self.pred_v, Rmat, Tmat[:,np.newaxis,:], opts.n_bones,opts.n_hypo,skin,tocam=True)
        self.Tmat = torch.zeros(3,1).cuda()
        self.Rmat = torch.eye(3).cuda()
        if opts.n_bones>1:
            self.gaussian_3d = obj_to_cam(torch.Tensor(self.sphere.vertices).cuda()[None], Rmat, Tmat[:,np.newaxis], opts.n_bones, opts.n_hypo,
                                torch.eye(opts.n_bones-1)[None].repeat(self.nsphere_verts,1,1).permute(1,2,0).reshape(1,opts.n_bones-1,-1,1).cuda(),tocam=True)
        self.ppoint, self.scale = self.ppoint, scale
        verts = torch.cat([verts,torch.ones_like(verts[:, :, 0:1])], dim=-1)
        verts[:,:,1] = self.ppoint[:,1:2]+verts[:, :, 1].clone()*scale[:,:1]/ verts[:,:,2].clone()
        verts[:,:,0] = self.ppoint[:,0:1]+verts[:, :, 0].clone()*scale[:,:1]/ verts[:,:,2].clone()
        verts[:,:,2] = ( (verts[:,:,2]-verts[:,:,2].min())/(verts[:,:,2].max()-verts[:,:,2].min())-0.5).detach()
        proj_mat = torch.eye(4).cuda()[np.newaxis].repeat(verts.shape[0],1,1)

        # texture
        proj_cam =   self.cam_pred.clone()
        proj_depth = self.depth.clone()   
        proj_pp =    self.ppoint.clone()  
        self.model.texture_type='vertex'
        self.tex = self.tex
        
        # start rendering
        Rmat_tex = Rmat.clone()
        verts_tex = obj_to_cam(self.pred_v, Rmat_tex, Tmat[:,np.newaxis,:], opts.n_bones,opts.n_hypo,skin)
        verts_tex = torch.cat([verts_tex,torch.ones_like(verts_tex[:, :, 0:1])], dim=-1)
        verts_tex = pinhole_cam(verts_tex, proj_pp, scale)
        self.renderer_softtex.rasterizer.background_color=[1,1,1]
        offset = torch.Tensor( self.renderer_softtex.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
        verts_pre = verts_tex[:,:,:3]-offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
        verts_pre[:,:,0] = -1*verts_pre[:,:,0]
        mesh = trimesh.Trimesh(vertices=np.asarray(verts_pre[0].cpu()), faces=np.asarray(self.faces[0].cpu()),vertex_colors=(self.tex[0].cpu()))
        trimesh.repair.fix_normals(mesh)
        mesh = Meshes(verts=torch.Tensor(mesh.vertices[None]).cuda(), faces=torch.Tensor(mesh.faces[None]).cuda(),textures=TexturesVertex(verts_features=torch.Tensor(mesh.visual.vertex_colors[None,:,:3]).cuda()/500.))
        self.texture_render = self.renderer_pyr(mesh)
        self.mask_pred = self.texture_render[:,:,:,-1:]
        self.texture_render = self.texture_render[:,:,:,:3].permute(0,3,1,2)

        Rmat_tex[:1] = Rmat[:1].clone().matmul(kornia.quaternion_to_rotation_matrix(torch.Tensor([[0,-0.707,0,0.707]]).cuda()))
        verts_tex = obj_to_cam(self.pred_v, Rmat_tex, Tmat[:,np.newaxis,:], opts.n_bones,opts.n_hypo,skin)
        verts_tex = torch.cat([verts_tex,torch.ones_like(verts_tex[:, :, 0:1])], dim=-1)
        verts_tex = pinhole_cam(verts_tex, proj_pp, scale)
        offset = torch.Tensor( self.renderer_softtex.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
        verts_pre = verts_tex[:,:,:3]-offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
        verts_pre[:,:,0] = -1*verts_pre[:,:,0]
        mesh = trimesh.Trimesh(vertices=np.asarray(verts_pre[0].cpu()/1.2), faces=np.asarray(self.faces[0].cpu()),vertex_colors=(self.tex[0].cpu()))
        trimesh.repair.fix_normals(mesh)
        mesh = Meshes(verts=torch.Tensor(mesh.vertices[None]).cuda(), faces=torch.Tensor(mesh.faces[None]).cuda(),textures=TexturesVertex(verts_features=torch.Tensor(mesh.visual.vertex_colors[None,:,:3]).cuda()/500.))
        self.texture_vp2 = self.renderer_pyr(mesh)[:,:,:,:3].permute(0,3,1,2)
        self.verts_vp2 = obj_to_cam(self.pred_v, Rmat_tex, Tmat[:,np.newaxis,:], opts.n_bones,opts.n_hypo,skin,tocam=True)
        
        Rmat_tex[:1] = Rmat[:1].clone().matmul(kornia.quaternion_to_rotation_matrix(torch.Tensor([[-0.707,0,0,0.707]]).cuda()))
        verts_tex = obj_to_cam(self.pred_v, Rmat_tex, Tmat[:,np.newaxis,:], opts.n_bones,opts.n_hypo,skin)
        verts_tex = torch.cat([verts_tex,torch.ones_like(verts_tex[:, :, 0:1])], dim=-1)
        verts_tex = pinhole_cam(verts_tex, proj_pp, scale)
        offset = torch.Tensor( self.renderer_softtex.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
        verts_pre = verts_tex[:,:,:3]-offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
        verts_pre[:,:,0] = -1*verts_pre[:,:,0]
        mesh = trimesh.Trimesh(vertices=np.asarray(verts_pre[0].cpu())/1.2, faces=np.asarray(self.faces[0].cpu()),vertex_colors=(self.tex[0].cpu()))
        trimesh.repair.fix_normals(mesh)
        mesh = Meshes(verts=torch.Tensor(mesh.vertices[None]).cuda(), faces=torch.Tensor(mesh.faces[None]).cuda(),textures=TexturesVertex(verts_features=torch.Tensor(mesh.visual.vertex_colors[None,:,:3]).cuda()/500.))
        self.texture_vp3 = self.renderer_pyr(mesh)[:,:,:,:3].permute(0,3,1,2)
        self.verts_vp3 = obj_to_cam(self.pred_v, Rmat_tex, Tmat[:,np.newaxis,:], opts.n_bones,opts.n_hypo,skin,tocam=True)
        # end rendering

        Rmat_tex = Rmat.clone()
        verts_tex = obj_to_cam(self.pred_v, Rmat_tex, Tmat[:,np.newaxis,:], opts.n_bones,opts.n_hypo,skin)
        verts_tex = torch.cat([verts_tex,torch.ones_like(verts_tex[:, :, 0:1])], dim=-1)
        verts_tex = pinhole_cam(verts_tex, proj_pp, scale)
        offset = torch.Tensor( self.renderer_softtex.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
        verts_pre = verts_tex[:,:,:3]-offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
        self.skin_vis=[]
        if self.opts.n_bones>1:
            for i in range(skin.shape[1]):
                self.skin_vis.append( self.renderer_softtex.render_mesh(sr.Mesh(verts_pre, self.faces, textures=skin[:1,i]*torch.Tensor([1,0,0]).cuda()[None,None],texture_type='vertex'))[:,:3].clone() )
            # color palette
            colormap = torch.Tensor(citylabs[:skin.shape[1]]).cuda() # 5x3
            skin_colors = (skin[0] * colormap[:,None]).sum(0)/256.
            self.part_render = self.renderer_softpart.render_mesh(sr.Mesh(verts_pre.detach(), self.faces, textures=skin_colors[None], texture_type='vertex'))[:,:3].detach()
        self.skin = skin        

    def collect_outputs(self):
        outputs = {
            'verts': self.verts.data,
            'verts_vp2': self.verts_vp2.data,
            'verts_vp3': self.verts_vp3.data,
            'joints': self.joints_proj.data,
            'cam_pred': self.cam_pred.data,
            'mask_pred': self.mask_pred[0].data,
            'tex': self.tex.data[0],
        }

        return outputs
