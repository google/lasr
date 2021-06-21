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

from absl import app
from absl import flags
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import pdb
import time
import numpy as np

import cv2
import kornia
import torch

from nnutils.train_utils import LASRTrainer

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

flags.DEFINE_integer('local_rank', 0, 'for distributed training')
flags.DEFINE_integer('ngpu', 1, 'number of gpus to use')
flags.DEFINE_boolean('use_gtpose', True, 'if true uses gt pose for projection, but camera still gets trained.')
flags.DEFINE_string('sil_path', 'none', 'additional silouette path')
opts = flags.FLAGS
    
def main(_):
    torch.cuda.set_device(opts.local_rank)
    world_size = opts.ngpu
    torch.distributed.init_process_group(
    'nccl',
    init_method='env://',
    world_size=world_size,
    rank=opts.local_rank,
    )
    print('%d/%d'%(world_size,opts.local_rank))

    torch.manual_seed(0)
    torch.cuda.manual_seed(1)
    torch.manual_seed(0)
    trainer = LASRTrainer(opts)
    trainer.init_training()

    trainer.train()

if __name__ == '__main__':
    app.run(main)
