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
from nnutils.geom_utils import label_colormap
from nnutils import mesh_net
from nnutils import geom_utils
from dataloader import vid as vid_data


#-------- tranining class ---------#
#----------------------------------#
class Trainer():
    def __init__(self, opts):
        self.opts = opts
        self.local_rank = opts.local_rank
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.name)
        if opts.local_rank==0:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            log_file = os.path.join(self.save_dir, 'opts.log')
            with open(log_file, 'w') as f:
                for k in dir(opts):
                    f.write('{}: {}\n'.format(k, opts.__getattr__(k)))


    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, local_rank=None):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    # helper loading function that can be used by subclasses
    def load_network(self, network, model_path=None):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def define_model(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_dataset(self):
        opts = self.opts
        self.data_module = vid_data

        self.dataloader,_ = self.data_module.data_loader(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def define_criterion(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def set_input(self, batch):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def forward(self):
        '''Should compute self.total_loss. To be implemented by the child class.'''
        raise NotImplementedError

    def save(self, epoch_prefix):
        '''Saves the model.'''
        if self.opts.local_rank==0:
            self.save_network(self.model.module, 'pred', epoch_prefix, local_rank=self.opts.local_rank)
        return

    def get_current_visuals(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_scalars(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def get_current_points(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def init_training(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError

    def train(self):
        '''Should be implemented by the child class.'''
        raise NotImplementedError
