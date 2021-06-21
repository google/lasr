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


davisdir=./database/DAVIS/
res=Full-Resolution
seqname=$1
newname=r${seqname}

rm ./$seqname -rf
# run flow on frames with sufficient motion
CUDA_VISIBLE_DEVICES=0 python preprocess/auto_gen.py --datapath $davisdir/JPEGImages/$res/$seqname/ --loadmodel ./lasr_vcn/vcn_rob.pth  --testres 1

mkdir $davisdir/JPEGImages/$res/$newname
mkdir $davisdir/Annotations/$res/$newname
mkdir $davisdir/FlowFW/$res/$newname
mkdir $davisdir/FlowBW/$res/$newname
cp $seqname/JPEGImages/*   -rf $davisdir/JPEGImages/$res/$newname
cp $seqname/Annotations/* -rf $davisdir/Annotations/$res/$newname
cp $seqname/FlowFW/*           -rf $davisdir/FlowFW/$res/$newname
cp $seqname/FlowBW/*           -rf $davisdir/FlowBW/$res/$newname
rm ./$seqname -rf

# run flow on the full seq
CUDA_VISIBLE_DEVICES=0 python preprocess/auto_gen.py --datapath $davisdir/JPEGImages/$res/$seqname/ --loadmodel ./lasr_vcn/vcn_rob.pth  --testres 1 --flow_threshold 0
mkdir $davisdir/FlowFW/$res/$seqname
mkdir $davisdir/FlowBW/$res/$seqname
cp $seqname/FlowFW/*           -rf $davisdir/FlowFW/$res/$seqname
cp $seqname/FlowBW/*           -rf $davisdir/FlowBW/$res/$seqname
rm ./$seqname -rf
