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

from absl import flags, app
import sys
sys.path.insert(0,'third_party')

import numpy as np
import skimage.io as io
import torch
import os
import glob
import pdb
import cv2
import matplotlib.pyplot as plt
import soft_renderer as sr

from nnutils import predictor as pred_util
from nnutils.geom_utils import label_colormap
from ext_utils import fusion
from ext_utils import image as img_util
import configparser
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

flags.DEFINE_integer('img_size', 256, 'image size the network was trained on.')
flags.DEFINE_string('evolve', 'no', 'wether to visualize different epochs.')
flags.DEFINE_integer('batch_size', 4, 'Size of minibatches')
flags.DEFINE_integer('num_train_epoch', 0, 'Number of training iterations')
flags.DEFINE_string('checkpoint_dir', './',
                    'Directory where networks are saved')
flags.DEFINE_string('name', 'exp_name', 'Experiment Name')
opts = flags.FLAGS

citylabs = label_colormap()

def preprocess_image(img_path, img_size=256):
    img = cv2.imread(img_path)[:,:,::-1] / 255.

    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)

    mask = cv2.imread(img_path.replace('JPEGImages', 'Annotations').replace('.jpg','.png'),0)
    if mask.shape[0]!=img.shape[0] or mask.shape[1]!=img.shape[1]:
        mask = cv2.resize(mask, img.shape[:2][::-1])
    mask = np.expand_dims(mask, 2)
    
    color = img[mask[:,:,0].astype(bool)].mean(0)
    img =   img*(mask>0).astype(float) + (1-color )[None,None,:]*(1-(mask>0).astype(float))
    img_black =   img*(mask>0).astype(float) + (1-(mask>0).astype(float))

    # crop box
    indices = np.where(mask>0); xid = indices[1]; yid = indices[0]
    center = ( (xid.max()+xid.min())//2, (yid.max()+yid.min())//2)
    length = ( (xid.max()-xid.min())//2, (yid.max()-yid.min())//2)
    maxlength = int(1.2*max(length))
    length = (maxlength,maxlength)

    x0,y0=np.meshgrid(range(2*length[0]),range(2*length[0]))
    x0=(x0+(center[0]-length[0])).astype(np.float32)
    y0=(y0+(center[1]-length[0])).astype(np.float32)
    img = cv2.remap(img,x0,y0,interpolation=cv2.INTER_LINEAR,borderValue=(1-color))
    img_black = cv2.remap(img_black,x0,y0,interpolation=cv2.INTER_LINEAR,borderValue=img_black[0,0])

    maxw=256;maxh=256
    img = cv2.resize(img ,  (maxw,maxh), interpolation=cv2.INTER_LINEAR)
    img_black = cv2.resize(img_black ,  (maxw,maxh), interpolation=cv2.INTER_LINEAR)
    alp = 2*length[0]/maxw

    # Transpose the image to 3xHxW
    img = np.transpose(img, (2, 0, 1))
    img_black = np.transpose(img_black, (2, 0, 1))

    pps  = np.asarray([float( center[0] - length[0] ), float( center[1] - length[1]  )])
    return img, alp, img_black, pps


def visualize(img, outputs, predictor,ipath,saveobj=False,epoch=None):
    vert = outputs['verts'][0]
    vert_vp2 = outputs['verts_vp2'][0]
    vert_vp3 = outputs['verts_vp3'][0]

    if epoch is None:
        epoch=int(ipath.split('/')[-1].split('.')[0])
    #if True:
    if saveobj or predictor.opts.n_mesh>1:
        save_dir = os.path.join(predictor.opts.checkpoint_dir, predictor.opts.name)
        fusion.meshwrite('%s/pred%d.ply'%(save_dir, epoch), np.asarray(vert.cpu()), np.asarray(predictor.faces.cpu()[0]), colors=255*outputs['tex'].cpu())
        fusion.meshwrite('%s/vp2pred%d.ply'%(save_dir, epoch), np.asarray(vert_vp2.cpu()), np.asarray(predictor.faces.cpu()[0]), colors=255*outputs['tex'].cpu())
        fusion.meshwrite('%s/vp3pred%d.ply'%(save_dir, epoch), np.asarray(vert_vp3.cpu()), np.asarray(predictor.faces.cpu()[0]), colors=255*outputs['tex'].cpu())
        if predictor.bones_3d is not None:
            colormap = torch.Tensor(citylabs[:predictor.bones_3d.shape[1]]).cuda() # 5x3
            fusion.meshwrite('%s/bone%d.ply'%(save_dir, epoch), np.asarray(predictor.bones_3d[0].cpu()), np.zeros((0,3)),colors=colormap)
            # gaussian
            skin = predictor.gauss_skin[0,:,:,0]
            skin_colors = skin.T
            skin_colors = (skin_colors[:,:,None] * colormap[None]).sum(1)
            fusion.meshwrite('%s/gauss%d.ply'%(save_dir, epoch), np.asarray(predictor.gaussian_3d[0].cpu()),predictor.sphere.faces,
            #            colors=np.asarray(skin_colors.cpu()))
                        colors=np.asarray(colormap[None].repeat(predictor.nsphere_verts,1,1).permute(1,0,2).reshape(-1,3).cpu()) )
            
        # camera
        RT = np.asarray(torch.cat([predictor.Rmat, predictor.Tmat],-1).cpu())
        #K = np.asarray(torch.cat([predictor.ppoint, predictor.scale],-1).cpu())
        K = np.asarray(torch.cat([predictor.uncrop_scale[0,0,:], predictor.uncrop_pp],-1).view(-1,4).cpu())
        RTK = np.concatenate([RT,K],0)
        #RTK = np.concatenate([RT,K.T],-1)
        np.savetxt('%s/cam%d.txt'%(save_dir, epoch),RTK)
    mask_pred = np.asarray(predictor.mask_pred[0][0].detach().cpu())*255
    
    vp1 = np.asarray(predictor.texture_render.data[0].permute(1,2,0).cpu())
    vp2 = np.asarray(predictor.texture_vp2.data[0].permute(1,2,0).cpu())
    vp3 = np.asarray(predictor.texture_vp3.data[0].permute(1,2,0).cpu())

    img = np.transpose(img, (1, 2, 0))

    redImg = np.zeros(img.shape, np.uint8)
    redImg[:,:] = (0, 0, 255)
    redMask = (redImg * mask_pred[:,:,np.newaxis]/255).astype(np.uint8)
    #if opts.n_mesh>1:
    #    redMask = np.asarray(predictor.part_render[0].permute(1,2,0).cpu()*255, dtype=np.uint8)
    redMask = cv2.addWeighted(redMask, 0.5, (255*img).astype(np.uint8), 1, 0, (255*img).astype(np.uint8))

    plt.ioff()
    plt.figure(figsize=(16,4))
    plt.clf()
    plt.subplot(141)
#    for k in range(outputs['joints'].shape[0]):
#        if predictor.pidx is None or predictor.pidx-1==k:
#            if predictor.pidx is not None: csize=40
#            else: csize=3
#            redMask = cv2.circle(redMask,tuple(128+128*np.asarray(outputs['joints'].cpu())[k,:2]),csize,citylabs[k].tolist(),3)
    plt.imshow(redMask)
    if opts.evolve=='yes':
        plt.gca().set_title('input/rendered mask [epoch %d]'%opts.num_train_epoch)
    else:
        plt.gca().set_title('input/rendered mask [frame %d]'%epoch)
    plt.axis('off')
    plt.subplot(142)
    plt.imshow(vp1)
    plt.gca().set_title('front view')
    plt.axis('off')
    plt.subplot(143)
    plt.imshow(vp2)
    plt.gca().set_title('right view')
    plt.axis('off')
    plt.subplot(144)
    plt.imshow(vp3)
    plt.gca().set_title('top view')
    plt.axis('off')
    plt.gca().set_facecolor('white')
    plt.draw()
    if opts.evolve=='yes':
        plt.savefig('%s/render-%05d.png'%(save_dir, opts.num_train_epoch),facecolor = plt.gca().get_facecolor(), transparent = True)
    else:
        plt.savefig('%s/render-%s.png'%(save_dir, ipath.split('/')[-1].split('.')[0]))
    plt.close()
    
    plt.figure(figsize=(16,16))
    plt.clf()
    for i in range(len(predictor.skin_vis)):
        plt.subplot(6,7,i+1)
        skinvis = np.asarray(predictor.skin_vis[i][0].permute(1,2,0).cpu())
        skinvis = cv2.circle(skinvis.copy(),tuple(128+128*np.asarray(outputs['joints'].cpu())[i,:2]),1,(0.,1.,1.),3)
        plt.imshow(skinvis)
        plt.axis('off')
    plt.draw()
    plt.savefig('%s/renderskin-%s.png'%(save_dir, ipath.split('/')[-1].split('.')[0]))
    plt.close()

    # visualize skin
    config = configparser.RawConfigParser()
    config.read('configs/%s.config'%opts.dataname)
    canonical_frame = int(config.get('data', 'can_frame'))
    if epoch==canonical_frame:
        if not predictor.skin is None:
            skin = predictor.skin[0,:,:,0]
            skin_colors = skin.T
            # color palette
            colormap = torch.Tensor(citylabs[:skin.shape[0]]).cuda() # 5x3
            skin_colors = (skin_colors[:,:,None] * colormap[None]).sum(1)/256.
            sr.Mesh(predictor.pred_v[0], predictor.faces, textures=255*skin_colors.cpu(),texture_type='vertex').save_obj('%s/clusters.obj'%save_dir,save_texture=True)
            fusion.meshwrite('%s/cpoints.ply'%save_dir, np.asarray(predictor.model.ctl_ts.detach().cpu()), np.asarray(predictor.model.faces.cpu())[:0],colors=colormap)
            np.save('%s/skin.npy'%save_dir, np.asarray(skin.cpu())) # BxN
        

def main(_):
    config = configparser.RawConfigParser()
    config.read('configs/%s.config'%opts.dataname)
    datapath = str(config.get('data', 'datapath'))
    canonical_frame = int(config.get('data', 'can_frame'))
    dframe = int(config.get('data', 'dframe'))
    init_frame = int(config.get('data', 'init_frame'))
    end_frame = int(config.get('data', 'end_frame'))
    
    if opts.evolve=='yes':
        for i,ipath in enumerate(sorted(glob.glob('%s/*'%datapath))):
            if i!=canonical_frame:continue
            for j in range(0,200):
                opts.num_train_epoch=j
                predictor = pred_util.MeshPredictor(opts)
                img,alp,imgb,pp = preprocess_image(ipath, img_size=opts.img_size)
 
                batch = {'img': torch.Tensor(np.expand_dims(img, 0))}
 
                print('frame-id:%d'%i)
                #predictor.model.train()
                with torch.no_grad():
                    #outputs = predictor.predict(batch,is_cano=False)
                    outputs = predictor.predict(batch,alp,pp,frameid=i)
 
                visualize(imgb, outputs, predictor,ipath,saveobj=True,epoch=j)
    else:
        # temporal
        predictor = pred_util.MeshPredictor(opts)
        for i,ipath in enumerate(sorted(glob.glob('%s/*'%datapath))):
            if (i%dframe!=init_frame%dframe) or (i<init_frame) or (end_frame>=0 and i >= end_frame):continue
            img,alp,imgb,pp = preprocess_image(ipath, img_size=opts.img_size)

            batch = {'img': torch.Tensor(np.expand_dims(img, 0))}

            print('frame-id:%d'%i)
            with torch.no_grad():
                outputs = predictor.predict(batch,alp,pp,frameid=i)

            visualize(imgb, outputs, predictor,ipath,saveobj=True)


if __name__ == '__main__':
    opts.batch_size = 1
    app.run(main)
