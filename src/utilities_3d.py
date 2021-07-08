import torch
import torch.nn as nn 
from torch.nn import functional as F

import numpy as np 
from neighbor_list import computInterListOpt
import time


@torch.jit.script
def smooth_cut_off(r:torch.Tensor, r_cs:torch.Tensor, r_c:torch.Tensor):
  #  todo: this needs to be tested

  mask = (r == 0.)
  zeroDummy = torch.zeros_like(r, dtype = torch.float32)

  ## todo the computation

  # remove the zeros to avoid taking the derivative of something too small
  r_temp = torch.where(mask, zeroDummy, r)

  # 1/r
  s_temp = 1./r_temp
  # 1.r*(0.5*cos(pi*(r - r_cs)/(r_c - r_cs) + 0.5)
  s_temp2 =  torch.where( (r >= r_cs)*(r <= r_c), 
            s_temp*0.5*(torch.cos(torch.tensor(np.pi, dtype = torch.float32)*\
                                    (r_temp-r_cs)/(r_c-r_cs)) + 1.), s_temp)
  s_temp3 = torch.where( (r >= r_c), zeroDummy, s_temp2)

  s_final = torch.where(mask, zeroDummy, s_temp3)

  return s_final


# TODO: check that this work as intended
@torch.jit.script
def gen_coordinates(pos : torch.Tensor, 
                    neighborList : torch.Tensor, 
                    L : torch.Tensor, 
                    ave = torch.tensor([0.0], dtype=torch.float32), 
                    std = torch.tensor([1.0], dtype=torch.float32)):

  # extracting the size information 
  n_snap, n_points, max_num_neighs = neighborList.size()

  assert pos.size()[0] == n_snap
  assert pos.size()[1] == n_points

  # dimension of the input (should be 3D)
  dim = pos.size()[2]

  mask = (neighborList == -1)
  # change the neighbor (because of -1 can not index)
  # we create a new temporal neighbor list 
  # otherwise we modify neighborList outside the scope 
  # of this function
  neighborListTemp = neighborList*(1 - mask.int())

  zeroDummy = torch.zeros(n_snap, n_points, max_num_neighs, dim, dtype=torch.float32)
  # (n_snap, n_points, max_num_neighs, dim) 

  zeroDummy_scalar = torch.zeros(n_snap, n_points, max_num_neighs, dtype=torch.float32)
  # (n_snap, n_points, max_num_neighs)

  temp = pos.unsqueeze(-2).repeat(1, 1, neighborListTemp.shape[-1], 1)
  # (n_snap, n_points, max_num_neighs, dim)

  temp2 = torch.gather(temp, 1, neighborListTemp.unsqueeze(-1).repeat(1,1,1,dim))
  # (n_snap, n_points, max_num_neighs, dim) ? (not sure about the dimensions here)

  temp2Filtered = torch.where(mask.unsqueeze(-1).repeat(1,1,1,dim), 
                              zeroDummy, temp2)
  # (n_snap, n_points, max_num_neighs, dim)

  temp = temp2Filtered - temp
  #(we are supposing that the super cell is a cube)
  tempL = temp - L*torch.round(temp/L) 
  
  Dist = torch.sqrt(torch.sum(torch.square(tempL), dim = -1))
  # (n_snap, n_points, max_num_neighs, 1)

  # carefull of divisions by zero
  DistInv = 1./Dist

  Dist = torch.where(mask,zeroDummy_scalar, Dist) 
  DistInv = torch.where(mask,zeroDummy_scalar, DistInv) 

  return (Dist, tempL*DistInv.unsqueeze(-1))



class DenseChainNet(nn.Module):
# TODO: add a proper documentation
    
  def __init__(self, sizes, act_fn=torch.relu, 
               use_resnet=True, with_weight_skip=False, 
               with_batch_norm=False, **kwargs):
    super().__init__()

    self.act_fn = act_fn
    self.use_resnet = use_resnet
    self.with_weight_skip = with_weight_skip 
    self.with_batch_norm = with_batch_norm

    # if batch normalization is used
    # then set no bias to the dense layers
    self.lin_bias = False if with_batch_norm else True

    # creating the dense layers
    self.layers = nn.ModuleList([nn.Linear(in_f, out_f, bias=self.lin_bias) 
                                 for in_f, out_f in zip(sizes, sizes[1:])])
    
    # creating the batch normalization layers
    if self.with_batch_norm:
      self.batch_norm = nn.ModuleList([nn.BatchNorm1d(out_f) 
                                 for out_f in sizes[1:]])

    if with_weight_skip:
      self.weight_skip = nn.ParameterList(
          [nn.Parameter(torch.normal(torch.ones(out_f), std=0.01)) 
              for out_f in sizes[1:]])
    
    else:
        self.weight_skip = None
  

  def forward(self, x):
    for i, layer in enumerate(self.layers):
      
      # apply the dense network
      tmp = layer(x)

      # adding the batch normalization
      if self.with_batch_norm:
        tmp = self.batch_norm[i](tmp)

      # apply the activation function
      if i < len(self.layers) - 1:
        tmp = self.act_fn(tmp)

      # check if we are using ResNet 
      if self.use_resnet and layer.in_features == layer.out_features:
        if self.with_weight_skip:
          tmp = tmp * self.weight_skip[i]
        x = x + tmp
      else:
        x = tmp
    return x
