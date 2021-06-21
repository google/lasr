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

import sys, os
sys.path.insert(0,'third_party')
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU

import chamfer3D.dist_chamfer_3D
import subprocess
import pytorch3d.ops
import pytorch3d.loss
import imageio
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import pdb
import soft_renderer as sr
import argparse
import trimesh
from nnutils.geom_utils import obj_to_cam, pinhole_cam
import pyrender
from pyrender import IntrinsicsCamera,Mesh, Node, Scene,OffscreenRenderer
import configparser
import matplotlib.pyplot as plt
import imageio

parser = argparse.ArgumentParser(description='BADJA')
parser.add_argument('--testdir', default='',
                    help='path to test dir')
parser.add_argument('--gtdir', default='',
                    help='path to gt dir')
parser.add_argument('--method', default='lasr',
                    help='method to evaluate')
args = parser.parse_args()

gt_meshes =   [trimesh.load(i, process=False) for i in sorted( glob.glob('%s/*.obj'%(args.gtdir)) )]
if args.method=='vibe' or args.method=='pifuhd':
    pred_meshes = [i for i in sorted( glob.glob('%s/*.obj'%(args.testdir)) )]
elif args.method=='lasr':
    pred_meshes = [i for i in sorted( glob.glob('%s/pred*.ply'%(args.testdir)),key=lambda x: int(x.split('pred')[1].split('.ply')[0]) )]
elif args.method=='smplify-x':
    pred_meshes = [i for i in sorted( glob.glob('%s/*/*.obj'%(args.testdir)) )]
elif args.method=='smalify':
    pred_meshes = [i for i in sorted( glob.glob('%s/*/st10*.ply'%(args.testdir)) )]
else:exit()
assert(len(gt_meshes) == len(pred_meshes))
       


# pytorch3d 
from pytorch3d.renderer.mesh import TexturesAtlas, TexturesUV, TexturesVertex
from pytorch3d.structures.meshes import Meshes
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
renderer_softtex = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(image_size=512,cull_backfaces=True)),
        shader=SoftPhongShader(device = device,cameras=cameras, lights=lights)
)
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()

cds = []
norms=[]
frames=[]
for i in range(len(gt_meshes)):
    # remesh
    mesh1 = trimesh.load(pred_meshes[i], process=False)
    # load remeshed 
    if args.method=='lasr':
        import subprocess
        mesh1.export('tmp/input.obj')
        print(subprocess.check_output(['Manifold/build/manifold', 'tmp/input.obj', 'tmp/output.obj', '10000']))
        mesh1 = trimesh.load('tmp/output.obj')
    mesh2 = gt_meshes[i]

    trimesh.repair.fix_inversion(mesh1)
    trimesh.repair.fix_inversion(mesh2)

    X0 = torch.Tensor(mesh1.vertices[None] ).cuda()
    Y0 = torch.Tensor(mesh2.vertices[None] ).cuda()

    ## top down view
    #theta = -3*np.pi/9
    #init_pose = torch.Tensor([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]]).cuda()
    #X0[0] = X0.matmul(init_pose) 
    #Y0[0] = Y0.matmul(init_pose) 

    ## rotateview
    #theta = 9*np.pi/9
    #init_pose = torch.Tensor([[np.cos(theta),0,-np.sin(theta)],[0,1,0],[np.sin(theta),0,np.cos(theta)]]).cuda()
    #X0[0] = X0.matmul(init_pose) 


    if args.method=='lasr':
        cam = np.loadtxt('%s/cam%d.txt'%(args.testdir,i))
        Rmat =  torch.Tensor(cam[None,:3,:3]).cuda()
        X0 = X0.matmul(Rmat)
    elif args.method=='smalify':
        X0[:,:,1:] *= -1

    X0[:,:,1:] *= -1
    if 'sdog' in args.testdir or 'shorse' in args.testdir or 'spot' in args.testdir or 'sgolem' in args.testdir:
        Y0[:,:,1:] *= -1
   
    # normalize to have extent 10 
    Y0 = Y0 - Y0.mean(1,keepdims=True)
    max_dis = (Y0 - Y0.permute(1,0,2)).norm(2,-1).max()
    Y0 = 10* Y0 / max_dis
    
    X0 = X0 - X0.mean(1,keepdims=True)
    if args.method=='pifuhd' or args.method=='lasr':
        meshtmp = pytorch3d.structures.meshes.Meshes(verts=X0, faces=torch.Tensor(mesh1.faces[None]).cuda())
        Xtmp = pytorch3d.ops.sample_points_from_meshes(meshtmp, 10000) 
        max_dis = (Xtmp - Xtmp.permute(1,0,2)).norm(2,-1).max()
    else:
        max_dis = (X0 - X0.permute(1,0,2)).norm(2,-1).max()
    X0 = 10* X0 / max_dis
    
    meshx = pytorch3d.structures.meshes.Meshes(verts=X0, faces=torch.Tensor(mesh1.faces[None]).cuda())
    meshy = pytorch3d.structures.meshes.Meshes(verts=Y0, faces=torch.Tensor(mesh2.faces[None]).cuda())
    X = pytorch3d.ops.sample_points_from_meshes(meshx, 10000) 
    Y = pytorch3d.ops.sample_points_from_meshes(meshy, 10000) 

    sol1 = pytorch3d.ops.iterative_closest_point(X,Y,estimate_scale=False,max_iterations=10000)
    #sol2 = pytorch3d.ops.iterative_closest_point(sol1.Xt,Y,estimate_scale=True,max_iterations=10000)
    
    X0 = (sol1.RTs.s*X0).matmul(sol1.RTs.R)+sol1.RTs.T[:,None]
    

    # evaluation
    meshx = pytorch3d.structures.meshes.Meshes(verts=X0, faces=torch.Tensor(mesh1.faces[None]).cuda())
    meshy = pytorch3d.structures.meshes.Meshes(verts=Y0, faces=torch.Tensor(mesh2.faces[None]).cuda())
    X, nx= pytorch3d.ops.sample_points_from_meshes(meshx, 10000,return_normals=True)
    Y, ny= pytorch3d.ops.sample_points_from_meshes(meshy, 10000,return_normals=True)
    cd,norm = pytorch3d.loss.chamfer_distance(X,Y, x_normals=nx,y_normals=ny)
    raw_cd,_,_,_ = chamLoss(X,Y0)  # this returns distance squared

    # error render    
    cm = plt.get_cmap('plasma')
    color_cd = torch.Tensor(cm(2*np.asarray(raw_cd.cpu()[0]))).cuda()[:,:3][None]
    verts = Y0/(1.05*Y0.abs().max()); verts[:,:,0] *= -1; verts[:,:,-1] *= -1; verts[:,:,-1] -= (verts[:,:,-1].min()-1)
    mesh = Meshes(verts=verts, faces=torch.Tensor(mesh2.faces[None]).cuda(),textures=TexturesVertex(verts_features=color_cd))
    errimg = renderer_softtex(mesh)[0,:,:,:3]
    
    # shape render
    color_shape = torch.zeros_like(color_cd); color_shape += 0.5
    mesh = Meshes(verts=verts, faces=torch.Tensor(mesh2.faces[None]).cuda(),textures=TexturesVertex(verts_features=color_shape))
    imgy = renderer_softtex(mesh)[0,:,:,:3]
    
    # shape render
    color_shape = torch.zeros_like(X0); color_shape += 0.5
    verts = X0/(1.05*Y0.abs().max()); verts[:,:,0] *= -1; verts[:,:,-1] *= -1; verts[:,:,-1] -= (verts[:,:,-1].min()-1)
    mesh = Meshes(verts=verts, faces=torch.Tensor(mesh1.faces[None]).cuda(),textures=TexturesVertex(verts_features=color_shape))
    imgx = renderer_softtex(mesh)[0,:,:,:3]

    img = np.clip(255*np.asarray(torch.cat([imgy, imgx,errimg],1).cpu()),0,255).astype(np.uint8)
    #cv2.imwrite('%s/cd-%06d.png'%(args.testdir,i),img[:,:,::-1])
    cv2.imwrite('%s/gt-%06d.png'%(args.testdir,i),img[:,:512,::-1])
    cv2.imwrite('%s/pd-%06d.png'%(args.testdir,i),img[:,512:1024,::-1])
    cv2.imwrite('%s/cd-%06d.png'%(args.testdir,i),img[:,1024:,::-1])
    #trimesh.Trimesh(vertices=np.asarray(Y0[0].cpu()/10), faces=mesh2.faces,vertex_colors=np.asarray(color_cd[0].cpu())).export('0.obj')

    cds.append(np.asarray(cd.cpu()))
    norms.append(np.asarray(norm.cpu()))
    frames.append(img)
    print('%04d: %.2f, %.2f'%(i, cd,1-norm))
print('ALL: %.2f, %.2f'%(np.mean(cds),1-np.mean(norms)))
imageio.mimsave('tmp/output.gif', frames, duration=5./len(frames))
