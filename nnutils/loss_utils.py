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
Loss Utils.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from . import geom_utils
import numpy as np

class ARAPLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(ARAPLoss, self).__init__()
        self.nv = vertex.size(0)
        self.nf = faces.size(0)
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = 1
        laplacian[faces[:, 1], faces[:, 0]] = 1
        laplacian[faces[:, 1], faces[:, 2]] = 1
        laplacian[faces[:, 2], faces[:, 1]] = 1
        laplacian[faces[:, 2], faces[:, 0]] = 1
        laplacian[faces[:, 0], faces[:, 2]] = 1

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, dx, x):
        # lap: Nv Nv
        # dx: N, Nv, 3
        diffx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()
        diffdx = torch.zeros(x.shape[0], x.shape[1], x.shape[1]).cuda()
        for i in range(3):
            dx_sub = self.laplacian.matmul(torch.diag_embed(dx[:,:,i])) # N, Nv, Nv)
            dx_diff = (dx_sub - dx[:,:,i:i+1])
            
            x_sub = self.laplacian.matmul(torch.diag_embed(x[:,:,i])) # N, Nv, Nv)
            x_diff = (x_sub - x[:,:,i:i+1])
            
            diffdx += (dx_diff).pow(2)
            diffx +=   (x_diff).pow(2)

        diff = (diffx-diffdx).abs()
        diff = torch.stack([diff[i][self.laplacian.bool()].mean() for i in range(x.shape[0])])
        #diff = diff[self.laplacian[None].repeat(x.shape[0],1,1).bool()]
        return diff
