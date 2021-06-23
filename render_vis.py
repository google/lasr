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
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ["PYOPENGL_PLATFORM"] = "egl" #opengl seems to only work with TPU
sys.path.insert(0,'third_party')

import subprocess
import imageio
import glob
from ext_utils.badja_data import BADJAData
from ext_utils.joint_catalog import SMALJointInfo
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

parser = argparse.ArgumentParser(description='render mesh')
parser.add_argument('--testdir', default='',
                    help='path to test dir')
parser.add_argument('--seqname', default='camel',
                    help='sequence to test')
parser.add_argument('--watertight', default='no',
                    help='watertight remesh')
parser.add_argument('--outpath', default='/data/gengshay/output.gif',
                    help='output path')
parser.add_argument('--cam_type', default='perspective',
                    help='camera model, orthographic or perspective')
parser.add_argument('--append_img', default='no',
                    help='whether append images before the seq')
parser.add_argument('--append_render', default='yes',
                    help='whether append renderings')
parser.add_argument('--nosmooth', dest='smooth', action='store_false',
                    help='whether to smooth vertex colors and positions')
parser.add_argument('--gray', dest='gray', action='store_true',
                    help='whether to render with gray texture')
parser.add_argument('--overlay', dest='overlay',action='store_true',
                    help='whether to overlay with the input')
parser.add_argument('--vis_bones', dest='vis_bones',action='store_true',
                    help='whether show transparent surface and vis bones')
parser.add_argument('--freeze', dest='freeze', action='store_true',
                    help='freeze object at frist frame')
args = parser.parse_args()

renderer_softflf = sr.SoftRenderer(image_size=256,dist_func='hard' ,aggr_func_alpha='hard',
               camera_mode='look_at',perspective=False, aggr_func_rgb='hard',
               light_mode='vertex', light_intensity_ambient=1.,light_intensity_directionals=0.)

def preprocess_image(img,mask,imgsize):
    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
    if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
        mask = cv2.resize(mask, img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)[:,:,None]
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

def draw_joints_on_image(rgb_img, joints, visibility, region_colors, marker_types):
    joints = joints[:, ::-1] # OpenCV works in (x, y) rather than (i, j)

    disp_img = rgb_img.copy()    
    for joint_coord, visible, color, marker_type in zip(joints, visibility, region_colors, marker_types):
        if visible:
            joint_coord = joint_coord.astype(int)
            cv2.drawMarker(disp_img, tuple(joint_coord), color.tolist(), marker_type, 30, thickness = 10)
    return disp_img

def remesh(mesh):
    mesh.export('tmp/input.obj')
    print(subprocess.check_output(['./Manifold/build/manifold', 'tmp/input.obj', 'tmp/output.obj', '10000']))
    mesh = trimesh.load('tmp/output.obj',process=False)
    return mesh

def main():
    print(args.testdir)
    # store all the data
    all_anno = []
    all_mesh = []
    all_bone = []
    all_cam = []
    all_fr = []
    
    config = configparser.RawConfigParser()
    config.read('configs/%s.config'%args.seqname)
    datapath = str(config.get('data', 'datapath'))
    init_frame = int(config.get('data', 'init_frame'))
    end_frame = int(config.get('data', 'end_frame'))
    dframe = int(config.get('data', 'dframe'))
    for name in sorted(glob.glob('%s/*'%datapath))[init_frame:end_frame][::dframe]:
        rgb_img = cv2.imread(name)
        sil_img = cv2.imread(name.replace('JPEGImages', 'Annotations').replace('.jpg', '.png'),0)[:,:,None]
        all_anno.append([rgb_img,sil_img,0,0,name])
        seqname = name.split('/')[-2]
        fr = int(name.split('/')[-1].split('.')[-2])
        all_fr.append(fr)
        print('%s/%d'%(seqname, fr))
       
        try:
            try: 
                mesh = trimesh.load('%s/pred%d.ply'%(args.testdir, fr),process=False)
            except: 
                mesh = trimesh.load('%s/pred%d.obj'%(args.testdir, fr),process=False)
            trimesh.repair.fix_inversion(mesh)
            if args.watertight=='yes':
                mesh = remesh(mesh) 
            if args.gray:
                mesh.visual.vertex_colors[:,:3]=64
            if args.overlay:
                mesh.visual.vertex_colors[:,:2]=0
                mesh.visual.vertex_colors[:,2]=255
            all_mesh.append(mesh)
            cam = np.loadtxt('%s/cam%d.txt'%(args.testdir,fr))
            all_cam.append(cam)
            all_bone.append(trimesh.load('%s/gauss%d.ply'%(args.testdir, fr),process=False))
        except: print('no mesh found')


    # add bones?
    num_original_verts = []
    num_original_faces = []
    if args.vis_bones:
        for i in range(len(all_mesh)):
            all_mesh[i].visual.vertex_colors[:,-1]=192 # alpha
            num_original_verts.append( all_mesh[i].vertices.shape[0])
            num_original_faces.append( all_mesh[i].faces.shape[0]  )  
            all_mesh[i] = trimesh.util.concatenate([all_mesh[i], all_bone[i]])

    # store all the results
    input_size = all_anno[0][0].shape[:2]
    output_size = (int(input_size[0] * 480/input_size[1]), 480)# 270x480
    frames=[]
    if args.append_img=="yes":
        if args.append_render=='yes':
            if args.freeze: napp_fr = 30
            else:                  napp_fr = int(len(all_anno)//5)
            for i in range(napp_fr):
                frames.append(cv2.resize(all_anno[0][0],output_size[::-1])[:,:,::-1])
        else:
            for i in range(len(all_anno)):
                #frames.append(cv2.resize(all_anno[i][1],output_size[::-1])*255) # silhouette
                frames.append(cv2.resize(all_anno[i][0],output_size[::-1])[:,:,::-1]) # frame
                #strx = sorted(glob.glob('%s/*'%datapath))[init_frame:end_frame][::dframe][i]# flow
                #strx = strx.replace('JPEGImages', 'FlowBW')
                #flowimg = cv2.imread('%s/vis-%s'%(strx.rsplit('/',1)[0],strx.rsplit('/',1)[1]))
                #frames.append(cv2.resize(flowimg,output_size[::-1])[:,:,::-1]) 
    theta = 7*np.pi/9
    light_pose = np.asarray([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]])
    if args.freeze:
        size = 150
    else:
        size = len(all_anno)
    for i in range(size):
        if args.append_render=='no':break
        # render flow between mesh 1 and 2
        if args.freeze:
            print(i)
            refimg = all_anno[0][0]
            img_size = max(refimg.shape)
            refmesh = all_mesh[0]
            refmesh.vertices -= refmesh.vertices.mean(0)[None]
            refmesh.vertices /= 1.2*np.abs(refmesh.vertices).max()
            refcam = all_cam[0].copy()
            refcam[:3,:3] = refcam[:3,:3].dot(cv2.Rodrigues(np.asarray([0.,-i*2*np.pi/size,0.]))[0])
            refcam[:2,3] = 0  # trans xy
            refcam[2,3] = 20 # depth
            if args.cam_type=='perspective':
                refcam[3,2] = refimg.shape[1]/2 # px py
                refcam[3,3] = refimg.shape[0]/2 # px py
                refcam[3,:2] = 8*img_size/2 # fl
            else:
                refcam[3,2] = refimg.shape[1]/2 # px py
                refcam[3,3] = refimg.shape[1]/2 # px py
                refcam[3,:2] =0.5 * img_size/2 # fl
        else:
            refimg, refsil, refkp, refvis, refname = all_anno[i]
            print('%s'%(refname))
            img_size = max(refimg.shape)
            renderer_softflf.rasterizer.image_size = img_size
            refmesh = all_mesh[i]
            refcam = all_cam[i]
        currcam = np.concatenate([refcam[:3,:4],np.asarray([[0,0,0,1]])],0)
        if i==0:
            initcam = currcam.copy()
        
        refface = torch.Tensor(refmesh.faces[None]).cuda()
        verts = torch.Tensor(refmesh.vertices[None]).cuda()
        Rmat =  torch.Tensor(refcam[None,:3,:3]).cuda()
        Tmat =  torch.Tensor(refcam[None,:3,3]).cuda()
        ppoint =refcam[3,2:4]
        scale = refcam[3,0]
        verts = obj_to_cam(verts, Rmat, Tmat,nmesh=1,n_hypo=1,skin=None)
        if args.cam_type != 'perspective':
            verts[:,:,1] = ppoint[1]+verts[:,:, 1]*scale[0]
            verts[:,:,0] = ppoint[0]+verts[:,:, 0]*scale[0]
            verts[:,:,2] += (5+verts[:,:,2].min())


        r = OffscreenRenderer(img_size, img_size)
        if args.overlay:
            bgcolor=[0., 0., 0.]
        else:
            bgcolor=[1.,1.,1.]
        scene = Scene(ambient_light=0.4*np.asarray([1.,1.,1.,1.]), bg_color=bgcolor)
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)

        colors = refmesh.visual.vertex_colors
        colors= np.concatenate([0.6*colors[:,:3].astype(np.uint8), colors[:,3:]],-1)  # avoid overexposure

        smooth=args.smooth
        if args.freeze:
            tbone = 0
        else:
            tbone = i
        if args.vis_bones:
            mesh = trimesh.Trimesh(vertices=np.asarray(verts[0,:num_original_verts[tbone],:3].cpu()), faces=np.asarray(refface[0,:num_original_faces[tbone]].cpu()),vertex_colors=colors[:num_original_verts[tbone]])
            meshr = Mesh.from_trimesh(mesh,smooth=smooth)
            meshr._primitives[0].material.RoughnessFactor=.5
            scene.add_node( Node(mesh=meshr ))
            mesh2 = trimesh.Trimesh(vertices=np.asarray(verts[0,num_original_verts[tbone]:,:3].cpu()), faces=np.asarray(refface[0,num_original_faces[tbone]:].cpu()-num_original_verts[tbone]),vertex_colors=colors[num_original_verts[tbone]:])
            mesh2=Mesh.from_trimesh(mesh2,smooth=smooth)
            mesh2._primitives[0].material.RoughnessFactor=.5
            scene.add_node( Node(mesh=mesh2))
        else: 
            mesh = trimesh.Trimesh(vertices=np.asarray(verts[0,:,:3].cpu()), faces=np.asarray(refface[0].cpu()),vertex_colors=colors)
            meshr = Mesh.from_trimesh(mesh,smooth=smooth)
            meshr._primitives[0].material.RoughnessFactor=.5
            scene.add_node( Node(mesh=meshr ))

        if not args.overlay:
            floor_mesh = trimesh.load('./database/misc/wood.obj',process=False)
            floor_mesh.vertices = np.concatenate([floor_mesh.vertices[:,:1], floor_mesh.vertices[:,2:3], floor_mesh.vertices[:,1:2]],-1 )
            xfloor = 10*mesh.vertices[:,0].min() + (10*mesh.vertices[:,0].max()-10*mesh.vertices[:,0].min())*(floor_mesh.vertices[:,0:1] - floor_mesh.vertices[:,0].min())/(floor_mesh.vertices[:,0].max()-floor_mesh.vertices[:,0].min()) 
            yfloor = floor_mesh.vertices[:,1:2]; yfloor[:] = (mesh.vertices[:,1].max())
            zfloor = 0.5*mesh.vertices[:,2].min() + (10*mesh.vertices[:,2].max()-0.5*mesh.vertices[:,2].min())*(floor_mesh.vertices[:,2:3] - floor_mesh.vertices[:,2].min())/(floor_mesh.vertices[:,2].max()-floor_mesh.vertices[:,2].min())
            floor_mesh.vertices = np.concatenate([xfloor,yfloor,zfloor],-1)
            floor_mesh = trimesh.Trimesh(floor_mesh.vertices, floor_mesh.faces, vertex_colors=255*np.ones((4,4), dtype=np.uint8))
            scene.add_node( Node(mesh=Mesh.from_trimesh(floor_mesh))) # overrides the prev. one
       
        if args.cam_type=='perspective': 
            cam = IntrinsicsCamera(
                    scale,
                    scale,
                    ppoint[0],
                    ppoint[1],
                    znear=1e-3,zfar=1000)
        else:
            cam = pyrender.OrthographicCamera(xmag=1., ymag=1.)
        cam_pose = -np.eye(4); cam_pose[0,0]=1; cam_pose[-1,-1]=1
        cam_node = scene.add(cam, pose=cam_pose)
        direc_l_node = scene.add(direc_l, pose=light_pose)
        if args.vis_bones:
            color, depth = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
        else:
            color, depth = r.render(scene,flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.SKIP_CULL_FACES)
        r.delete()
        color = color[:refimg.shape[0],:refimg.shape[1],:3]
        if args.overlay:
            color = cv2.addWeighted(color, 0.5, refimg[:,:,::-1], 0.5, 0)
        color = cv2.resize(color, output_size[::-1])

        frames.append(color)
    imageio.mimsave('%s'%args.outpath, frames, duration=5./len(frames))
if __name__ == '__main__':
    main()
