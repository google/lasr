# MIT License
# 
# Copyright (c) 2018 akanazawa
# Copyright (c) 2021 Google LLC
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
import kornia
import configparser
import soft_renderer as sr
from nnutils.geom_utils import pinhole_cam, obj_to_cam, render_flow_soft_3
from nnutils.geom_utils import label_colormap
citylabs = label_colormap()

class MeshNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100):
        super(MeshNet, self).__init__()
        self.opts = opts
        self.symmetric = opts.symmetric
        self.symmetric_texture = opts.symmetric_texture

        # Mean shape.
        if osp.exists('tmp/sphere_%d.npy'%(opts.subdivide)):
            sphere = np.load('tmp/sphere_%d.npy'%(opts.subdivide),allow_pickle=True)[()]
            verts = sphere[0]
            faces = sphere[1]
        else:
            verts, faces = mesh.create_sphere(opts.subdivide)
            if not os.path.exists('tmp'): os.mkdir('tmp')
            np.save('tmp/sphere_%d.npy'%(opts.subdivide),[verts,faces])
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]

        if self.symmetric:
            verts, faces, num_indept, num_sym, num_indept_faces, num_sym_faces,_ = mesh.make_symmetric(verts, faces, opts.symidx)
            num_sym_output = num_indept + num_sym
            if opts.only_mean_sym:
                print('Only the mean shape is symmetric!')
                self.num_output = num_verts
            else:
                self.num_output = num_sym_output
            self.num_sym = num_sym
            self.num_indept = num_indept
            self.num_indept_faces = num_indept_faces
            self.num_sym_faces = num_sym_faces
            self.mean_v = nn.Parameter(torch.Tensor(verts[:num_sym_output]).cuda())
           
            # texture pred
            self.texture_type = 'vertex'
            if self.opts.opt_tex=='yes':
                self.tex = nn.Parameter(torch.normal(torch.zeros(num_sym_output,3).cuda(),1))
            else:
                self.tex = torch.normal(torch.zeros(num_sym_output,3).cuda(),1)
        
            # Needed for symmetrizing..
            self.flip =Variable(torch.ones(1, 3),requires_grad=False)
            self.flip[0, opts.symidx] = -1

        else:
            self.num_output = num_verts
            self.mean_v = nn.Parameter(torch.Tensor(verts)[None])
            self.texture_type = 'vertex'
            if self.opts.opt_tex=='yes':
                self.tex = nn.Parameter(torch.normal(torch.zeros(1,num_verts,3).cuda(),1))
            else:
                self.tex = torch.normal(torch.zeros(num_verts,3).cuda(),1)

        config = configparser.RawConfigParser()
        config.read('data/%s.config'%opts.dataname)
        self.mean_v.data = self.mean_v.data.repeat(opts.n_hypo,1,1)
        self.tex.data = self.tex.data.repeat(opts.n_hypo,1,1)   # id, hypo, F, 3

        verts_np = verts
        faces_np = faces
        self.faces = Variable(torch.LongTensor(faces), requires_grad=False)

        self.encoder = nb.Encoder(input_shape, n_blocks=4, nz_feat=nz_feat)
        self.code_predictor = nb.CodePredictor(nz_feat=nz_feat, num_verts=self.num_output,n_mesh=opts.n_mesh,n_hypo = opts.n_hypo)

    def forward(self, batch_input):
        pass
    
    def symmetrize(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        if self.symmetric:
            if V.dim() == 2:
                # No batch
                V_left = self.flip.cuda() * V[-self.num_sym:]
                verts=torch.cat([V, V_left], 0)
                verts[:self.num_indept,self.opts.symidx]=0
                return verts 
            else:
                pdb.set_trace()
                # With batch
                V_left = self.flip * V[:, -self.num_sym:]
                verts = torch.cat([V, V_left], 1)
                verts[:,:self.num_indept, 0] = 0
                return verts
        else:
            return V
    
    def symmetrize_color(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        # No batch
        if self.symmetric:
            V_left = V[-self.num_sym:]
            verts=torch.cat([V, V_left], 0)
        else: verts = V
        return verts 

    def symmetrize_color_faces(self, tex_pred):
        if self.symmetric:
            tex_left = tex_pred[-self.num_sym_faces:]
            tex = torch.cat([tex_pred, tex_left], 0)
        else: tex = tex_pred
        return tex
    
    def get_mean_shape(self,local_batch_size):
        mean_v = torch.cat([self.symmetrize(i)[None] for i in self.mean_v],0)
        faces = self.faces
        if self.texture_type=='surface':
            tex = torch.cat([self.symmetrize_color_faces(i) for i in self.tex],0)
        else:
            tex = torch.cat([self.symmetrize_color(i)[None] for i in self.tex],0)
        
        faces = faces.repeat(2*local_batch_size,1,1)
        mean_v = mean_v[None].repeat(2*local_batch_size,1,1,1).view(2*local_batch_size*mean_v.shape[0],-1,3)
        if self.texture_type=='surface':
            tex = tex[np.newaxis].repeat(2*local_batch_size,1,1,1).sigmoid()
        else:
            tex = tex[np.newaxis].repeat(2*local_batch_size,1,1,1).sigmoid().view(2*local_batch_size*tex.shape[0],-1,3)
        return mean_v, tex, faces
