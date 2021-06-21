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

dev=0
ngpu=1
logname=spot3
checkpoint_dir=log/
dataname=spot3
nepoch=10
sil_path=none
address=1253

CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch --master_port $address --nproc_per_node=$ngpu optimize.py --name=$logname-0 --checkpoint_dir $checkpoint_dir --only_mean_sym --nouse_gtpose --subdivide 3 --n_mesh 21 --n_hypo 8 --num_epochs 5 --dataname $dataname  --sil_path $sil_path --ngpu $ngpu --batch_size 1 --opt_tex yes
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch --master_port $address --nproc_per_node=$ngpu optimize.py --name=$logname-1 --checkpoint_dir $checkpoint_dir --nosymmetric --nouse_gtpose --subdivide 3 --n_mesh 26 --n_faces 1600 --n_hypo 1 --num_epochs $nepoch --model_path $checkpoint_dir/$logname-0/pred_net_latest.pth --dataname $dataname  --sil_path $sil_path  --ngpu $ngpu --batch_size 1 --opt_tex yes
