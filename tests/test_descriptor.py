import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src')
from neighbor_list import comput_inter_list_type
from utilities_3d import gen_coordinates
from utilities_3d import smooth_cut_off
from utilities_3d import DenseChainNet
from networks_3d import DescriptorModule
from networks_3d import DeepMDsimpleEnergyForces

# we check that the construction of the moduledict is 
# properly performed :)
descript_dim = [1, 2, 4, 8, 16]
fitting_dim = [64, 32, 16, 8]

atom_types = [0, 1, 2]

resnet = False
with_weight_skip = False
with_batch_norm = False
act_fn = torch.relu


module_dict = {}
for i in range(len(atom_types)):
  for j in range(i+1):
    # we build such that j <= i
    module_dict[str(atom_types[j])+
    			str(atom_types[i])] = DenseChainNet(descript_dim, 
                                              act_fn,
                                              resnet, 
                                              with_weight_skip, 
                                              with_batch_norm)

embedding_nets_1 = nn.ModuleDict(module_dict)

## we create the testing example

n_parts = 3
r_in = np.linspace(0., 1., n_parts+1)[:-1] + 0.5/n_parts
x_in, y_in, z_in = np.meshgrid(r_in, r_in, r_in)

r_in = np.concatenate([x_in.reshape((-1,1)), 
                       y_in.reshape((-1,1)),
                       z_in.reshape((-1,1))], axis = -1)

r_in = r_in.reshape((1, *r_in.shape))

atom_type = []
for k in range(n_parts):
    atom_type += [k]*n_parts**2
atom_type = np.array(atom_type, dtype = np.int64).reshape((1,-1))

L = 1. 
radious = 0.45

max_num_neighs_type = np.array([5, 5, 5], dtype = np.int64)
inter_list = comput_inter_list_type(r_in, atom_type, L, radious, max_num_neighs_type)

# we move things to torch tensors
r_in_torch = torch.tensor(r_in, dtype = torch.float32)
inter_list_torch = torch.tensor(inter_list)
L_torch = torch.tensor(L)
input_types = torch.tensor(atom_type).view(1,-1)

n_points = n_parts**3

descriptor = DescriptorModule(n_points, L_torch, 
							  atom_types, max_num_neighs_type, 
							  descript_dim, False, [0.22, 0.45], 16 )

ans = descriptor(r_in_torch, input_types, inter_list_torch)

descriptor_torch = torch.jit.trace(descriptor, (r_in_torch, input_types, inter_list_torch))

DeepMD = DeepMDsimpleEnergyForces(n_points, # 
               L_torch, # we need it to be a tensor
               atom_types, 
               max_num_neighs_type, # this needs to be a tensor too
               descript_dim,
               fitting_dim,
               False,
               [0.4, 0.45],
               8,
               F.relu)

energy, forces = DeepMD(r_in_torch, input_types, inter_list_torch)


# # trying (unsuccesfully to compile DeepMD)
# deep_md_torch = torch.jit.trace(DeepMD, (r_in_torch, input_types, inter_list_torch))

descriptor_torch = torch.jit.trace(DeepMD.descriptor, (r_in_torch, input_types, inter_list_torch))
DeepMD.descriptor = descriptor_torch  # not very kosher... for sure

