# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# some functions are modified from nvdiffrast: https://github.com/NVlabs/nvdiffrast

import argparse
import os
import pathlib
import sys
import numpy as np
import torch
import pdb

def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

def q_rnd_m(b=1):
    randnum = np.random.uniform(0.0, 1.0, size=[3*b])
    u, v, w = randnum[:b,None], randnum[b:2*b,None], randnum[2*b:3*b,None]
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.concatenate([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)],-1).astype(np.float32)

def q_scale_m(q, t):
    out = q.copy()
    p=q_unit()
    d = np.dot(p, q.T)
    cond1 = d<0.0
    q[cond1] = -q[cond1]
    d[cond1] = -d[cond1]

    cond2 = d>0.999
    if cond2.sum()>0:
        a = p[None] + t[cond2][:,None] * (q[cond2]-p[None])
        out[cond2] =  a / np.linalg.norm(a,2,-1)[:,None]

    t0 = np.arccos(d)
    tt = t0 * t
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    if (~cond2).sum()>0:
        out[~cond2] =  (s0[:,None]*p[None] + s1[:,None]*q)[~cond2]
    return out
