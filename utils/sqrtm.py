import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as func

def sqrtm_sym(A):
    '''
    Pytorch implementation of square root of a symmetric matrix
    '''

    # assert torch.all(torch.eq(A, A.transpose(-2, -1)))

    _, s, v = torch.svd(A)

    return torch.matmul(v * s.sqrt().unsqueeze(-2), v.transpose(-2, -1))
