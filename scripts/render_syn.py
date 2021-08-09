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

import sys
sys.path.insert(0,'third_party')

import numpy as np
import trimesh
import torch
import cv2
import kornia
import pdb

import ext_utils.flowlib as flowlib
import ext_utils.util_flow as util_flow
from ext_utils.io import mkdir_p
import soft_renderer as sr
import argparse
parser = argparse.ArgumentParser(description='render data')
parser.add_argument('--outdir', default='syn-spot3f',
                    help='output dir')
parser.add_argument('--model', default='spot',
                    help='model to render, {spot, eagle}')
parser.add_argument('--nframes', default=3,type=int,
                    help='number of frames to render')
parser.add_argument('--alpha', default=1.,type=float,
                    help='0-1, percentage of a full cycle')
args = parser.parse_args()

## io
img_size = 512
dframe=1
bgcolor = None
xtime=1
filedir='database'
vertex_tex=False

def render_flow(renderer, verts, faces, verts_pos0, verts_pos1, pp0, pp1, proj_cam0,proj_cam1, h,w):
    offset = torch.Tensor( renderer.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]-offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
    verts_pos0_px = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=verts_pos0[:,:,:3],texture_type='vertex'))[:,:3].permute(0,2,3,1)
    verts_pos1_px = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=verts_pos1[:,:,:3],texture_type='vertex'))[:,:3].permute(0,2,3,1)
    
    bgmask = (verts_pos0_px[:,:,:,2]<1e-9) | (verts_pos1_px[:,:,:,2]<1e-9)
    verts_pos0_px[bgmask]=10
    verts_pos1_px[bgmask]=10
    # projet 3D verts with different intrinsics
    verts_pos0_px[:,:,:,1] = pp0[:,1:2,np.newaxis]+verts_pos0_px[:,:,:,1].clone()*proj_cam0[:,:1,np.newaxis] / verts_pos0_px[:,:,:,2].clone()
    verts_pos0_px[:,:,:,0] = pp0[:,0:1,np.newaxis]+verts_pos0_px[:,:,:,0].clone()*proj_cam0[:,:1,np.newaxis] / verts_pos0_px[:,:,:,2].clone()
    verts_pos1_px[:,:,:,1] = pp1[:,1:2,np.newaxis]+verts_pos1_px[:,:,:,1].clone()*proj_cam1[:,:1,np.newaxis] / verts_pos1_px[:,:,:,2].clone()
    verts_pos1_px[:,:,:,0] = pp1[:,0:1,np.newaxis]+verts_pos1_px[:,:,:,0].clone()*proj_cam1[:,:1,np.newaxis] / verts_pos1_px[:,:,:,2].clone()
    flow_fw = (verts_pos1_px - verts_pos0_px.detach())[:,:,:,:2]

    return flow_fw, bgmask


overts_list = []
for i in range(args.nframes):
    if args.model=='spot':
        mesh = sr.Mesh.from_obj('database/misc/spot/spot_triangulated.obj', load_texture=True, texture_res=5, texture_type='surface')
        overts = mesh.vertices
        overts[:,:,1]*= -1
        overts[:,:,1]+=0.1; overts /= 1.2;

    elif args.model=='eagle':
        mesh = sr.Mesh.from_obj('/data/gengshany_google_com/Downloads/eagled/Eagle-original_%06d.obj'%0, load_texture=True, texture_res=5, texture_type='surface')
        overts = mesh.vertices
        overts[:,:,1]*= -1
        overts[:,:,2]*= -1
        overts[:,:,1]+=2.5; overts /= 8;

    elif args.model=='dog':   
        mesh = sr.Mesh.from_obj('/home/gengshany_google_com/Downloads/dogd/dog_model_animation_%06d.obj'%(xtime*i), load_texture=True, texture_res=5, texture_type='surface')
        overts = mesh.vertices
        overts[:,:,1]*= -1
        overts[:,:,1]+= 0.4

    elif args.model=='horse':   
        mesh = sr.Mesh.from_obj('/home/gengshany_google_com/Downloads/horse3d/Horse_Rigged_%06d.obj'%(xtime*i+1),load_texture=True, texture_res=5, texture_type='surface')
        overts = mesh.vertices
        overts *= 0.5
        overts[:,:,2]*= -1
        overts[:,:,1]*= -1
        overts[:,:,1]+= 0.4
    
    elif args.model=='stone':   
        mesh = sr.Mesh.from_obj('/home/gengshany_google_com/Downloads/stone/stone_%06d.obj'%(i*xtime),load_texture=True, texture_res=5, texture_type='surface')
        overts = mesh.vertices
        overts[:,:,1]*= -1
        overts[:,:,1]+= 4.5
        overts /= 8.
        overts[:,:,2]*=-1.


    overts_list.append(overts)


if vertex_tex:
    if len(mesh.visual.to_color()._data.data)>0:
        colors = torch.Tensor(mesh.visual.to_color().vertex_colors[np.newaxis]/255.).cuda()
    else:
        colors = torch.ones_like(overts)
        colors = torch.cat([colors, colors[:,:,:1]],-1)
    faces = torch.Tensor(mesh.faces[np.newaxis]).cuda()
else:
    colors=mesh.textures
    faces = mesh.faces
proj_mat = torch.eye(4)[np.newaxis].cuda()

mkdir_p( '%s/DAVIS/JPEGImages/Full-Resolution/%s/'   %(filedir,args.outdir))
mkdir_p( '%s/DAVIS/Annotations/Full-Resolution/%s/'  %(filedir,args.outdir))
mkdir_p( '%s/DAVIS/FlowFW/Full-Resolution/%s/'       %(filedir,args.outdir))
mkdir_p( '%s/DAVIS/FlowBW/Full-Resolution/%s/'       %(filedir,args.outdir))
mkdir_p( '%s/DAVIS/Meshes/Full-Resolution/%s/'       %(filedir,args.outdir))
mkdir_p( '%s/DAVIS/Camera/Full-Resolution/%s/'       %(filedir,args.outdir))


cam_list = []
depth_list = []
verts_list = []
verts_pos_list = []

# soft renderer
renderer = sr.SoftRenderer(image_size=img_size, sigma_val=1e-12, 
               camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
               light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)
    
for i in range(0,args.nframes):
    overts = overts_list[i]
    # extract camera in/ex
    verts = overts.clone()
    rotx = np.random.rand()
    if i==0: rotx=0.
    roty = 3*1.57+args.alpha*6.28*i/args.nframes
    rotz = 0.
    rotmat = cv2.Rodrigues(np.asarray([rotx, roty, rotz]))[0]  # y-axis
    quat = kornia.rotation_matrix_to_quaternion(torch.Tensor(rotmat).cuda())
    proj_cam = torch.zeros(1,7).cuda()
    depth = torch.zeros(1,1).cuda()
    proj_cam[:,0]=10   # focal=10 
    proj_cam[:,1] = 0. # x translation = 0
    proj_cam[:,2] = 0. # y translation = 0
    proj_cam[:,3]=quat[3]
    proj_cam[:,4:]=quat[:3]
    depth[:,0] = 10   # z translation (depth) =10 for spot

    cam_list.append(proj_cam)
    depth_list.append(depth)

    # obj-cam transform 
    Rmat = kornia.quaternion_to_rotation_matrix(torch.cat((-proj_cam[:,4:],proj_cam[:,3:4]),1))
    Tmat = torch.cat([proj_cam[:,1:3],depth],1)
    verts = verts.matmul(Rmat) + Tmat[:,np.newaxis,:]  # obj to cam transform
    verts = torch.cat([verts,torch.ones_like(verts[:, :, 0:1])], dim=-1)
    
    verts_pos_list.append(verts.clone())  # this frame vertex (before projection)
    
    newmesh = trimesh.Trimesh(vertices=np.asarray(verts[0,:,:3].cpu()), faces=np.asarray(faces[0].cpu()))
   
    # pespective projection: x=fX/Z assuming px=py=0, normalization of Z
    verts[:,:,1] = verts[:, :, 1].clone()*proj_cam[:,:1]/ verts[:,:,2].clone()
    verts[:,:,0] = verts[:, :, 0].clone()*proj_cam[:,:1]/ verts[:,:,2].clone()
    verts[:,:,2] = ( (verts[:,:,2]-verts[:,:,2].min())/(verts[:,:,2].max()-verts[:,:,2].min())-0.5).detach()
    verts_list.append(verts.clone())
    
    # render sil
    offset = torch.Tensor( renderer.transform.transformer._eye ).cuda()[np.newaxis,np.newaxis]
    verts_pre = verts[:,:,:3]-offset; verts_pre[:,:,1] = -1*verts_pre[:,:,1]
    if vertex_tex:
        rendered = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=colors[:,:,:3],texture_type='vertex'))
    else:
        rendered = renderer.render_mesh(sr.Mesh(verts_pre,faces,textures=colors,texture_type='surface'))
    mask_pred=np.asarray(rendered[0,-1,:,:].detach().cpu())
    img_pred=np.asarray(rendered[0,:3,:,:].permute(1,2,0).detach().cpu())*255
   
    if bgcolor is None:
        bgcolor = 255-img_pred[mask_pred.astype(bool)].mean(0)

    img_pred[:,:,::-1][~mask_pred.astype(bool)]=bgcolor[None,::-1]
    cv2.imwrite('%s/DAVIS/JPEGImages/Full-Resolution/%s/%05d.jpg'     %(filedir,args.outdir,i),img_pred[:,:,::-1])
    cv2.imwrite('%s/DAVIS/Annotations/Full-Resolution/%s/%05d.png'    %(filedir,args.outdir,i),128*mask_pred)
    cammat = np.asarray(torch.cat([proj_cam[0], depth[0]],0).cpu())
    np.savetxt(    '%s/DAVIS/Camera/Full-Resolution/%s/%05d.txt'%(filedir,args.outdir,i),cammat)
    newmesh.export('%s/DAVIS/Meshes/Full-Resolution/%s/%05d.obj'%(filedir,args.outdir,i))

pp = torch.Tensor([[0,0]]).cuda()
occ_fw = -np.ones(mask_pred.shape).astype(np.float32)
occ_bw = -np.ones(mask_pred.shape).astype(np.float32)
for i in range(0,args.nframes):
    # render flow
    if (i-dframe)>=0:
        flow_fw,bgmask_fw = render_flow(renderer,verts_list[i-dframe], faces, verts_pos_list[i-dframe], verts_pos_list[i], pp, pp, cam_list[i],cam_list[i], img_size, img_size)
        flow_fw =   np.asarray( (flow_fw/2*(img_size-1)).cpu())[0]       
        bgmask_fw = np.asarray( bgmask_fw.float().cpu())               [0]
        flow_fw = np.concatenate([flow_fw, 1-bgmask_fw[:,:,np.newaxis]],-1)
    
        flow_bw,bgmask_bw = render_flow(renderer,verts_list[i],        faces, verts_pos_list[i], verts_pos_list[i-dframe], pp, pp, cam_list[i],cam_list[i], img_size, img_size)
        flow_bw =   np.asarray( (flow_bw/2*(img_size-1)).cpu())[0]       
        bgmask_bw = np.asarray( bgmask_bw.float().cpu())               [0]
        flow_bw = np.concatenate([flow_bw, 1-bgmask_bw[:,:,np.newaxis]],-1)
        
        util_flow.write_pfm( '%s/DAVIS/FlowFW/Full-Resolution/%s/flo-%05d.pfm'%(filedir,args.outdir,i-dframe),flow_fw)
        util_flow.write_pfm( '%s/DAVIS/FlowBW/Full-Resolution/%s/flo-%05d.pfm'%(filedir,args.outdir,i),flow_bw)
        util_flow.write_pfm( '%s/DAVIS/FlowFW/Full-Resolution/%s/occ-%05d.pfm'%(filedir,args.outdir,i-dframe),occ_fw)
        util_flow.write_pfm( '%s/DAVIS/FlowBW/Full-Resolution/%s/occ-%05d.pfm'%(filedir,args.outdir,i),occ_bw)
        cv2.imwrite(     '%s/DAVIS/FlowFW/Full-Resolution/%s/col-%05d.jpg'%(filedir,args.outdir,i-dframe),flowlib.flow_to_image(flow_fw)[:,:,::-1])
        cv2.imwrite(     '%s/DAVIS/FlowBW/Full-Resolution/%s/col-%05d.jpg'%(filedir,args.outdir,i),       flowlib.flow_to_image(flow_bw)[:,:,::-1])
