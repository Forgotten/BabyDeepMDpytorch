import torch
import torch.nn.functional as F
import numpy as np 
import torch.nn as nn

from utilities_3d import DenseChainNet
from utilities_3d import gen_coordinates
from utilities_3d import gen_coordinates_smooth
from utilities_3d import smooth_cut_off, stable_inverse

class DescriptorModule(torch.nn.Module):

  def __init__(self,
               n_points, # 
               length: torch.Tensor, # we need it to be a tensor
               atom_types = [0, 1, 2, 3], 
               max_num_neighs = [1, 3, 4], # this needs to be a tensor too
               descrip_dim = [1, 2, 4, 8, 16, 32],
               resnet = False,
               cut_offs = [2., 3.],
               dim_sub_net = 16,
               act_fn = F.relu,
               with_weight_skip = False, 
               with_batch_norm = False,
               with_smooth_cut_off = True,
               av = torch.tensor([0.0, 0.0], dtype = torch.float32),
               std = torch.tensor([1.0, 1.0], dtype = torch.float32),
               **kwargs):

    super().__init__()

    # save the different models in a dict
    # we can modify this so it is faster 
    module_dict = {}
    for i in range(len(atom_types)):
      for j in range(i+1):
        # we build such that j <= i
        # each network is index by a string "ij"
        # we make the assumption that we have up to 10 species
        module_dict[str(atom_types[j])+
                    str(atom_types[i])] = DenseChainNet(descrip_dim,
                                                        act_fn,
                                                        resnet, 
                                                        with_weight_skip, 
                                                        with_batch_norm)  

    # saving the dictionay on a moduledict object
    self.embed_nets = nn.ModuleDict(module_dict)

    self.length = length
    # self.descrip_dim    = torch.tensor(descrip_dim, dtype=torch.int32)
    # self.atom_types     = torch.tensor(atom_types, dtype=torch.int32)

    # testing this with lists
    self.descrip_dim    = descrip_dim
    self.atom_types     = atom_types

    self.max_num_neighs = max_num_neighs[:] # shallow copy


    self.embedding_dim = self.descrip_dim[-1]
    self.dim_sub_net = dim_sub_net

    # saving the types of cut-off
    self.with_smooth_cut_off = with_smooth_cut_off

    # saving the cut-offs 
    assert cut_offs[0] <= cut_offs[1]
    self.r_cs = torch.tensor(cut_offs[0])
    self.r_c = torch.tensor(cut_offs[1])

    max_num_neighs_cum = np.zeros(max_num_neighs.shape[0] + 1, 
                                  dtype = np.int32)
    max_num_neighs_cum[1:] = np.cumsum(max_num_neighs).astype(np.int32)

    self.total_num_neighs = max_num_neighs_cum[-1]


    # self.max_num_neighs_cum = torch.tensor(max_num_neighs_cum)
    self.max_num_neighs_cum = max_num_neighs_cum

    self.av = av
    self.std = std

  def forward(self, inputs: torch.Tensor, 
                    input_types: torch.Tensor, 
                    neigh_list: torch.Tensor):

        # inputs     = (n_snapshots, n_points, 3)
        # atom_types = (n_snapshots, n_points)
        # neigh_list = (n_snapshots, n_points, max_num_0, max_num_1, ..., max_num_L)

    device_in = inputs.device

    shape_in = inputs.shape
    n_snap, n_points, dim = inputs.shape
    n_snap_neigh, n_points_neigh, max_num_neighs = neigh_list.shape

    # assert n_snap == n_snap_neigh
    # assert n_points == n_points_neigh 


    # check if we are using a smooth cut-off if not 
    # 
    if self.with_smooth_cut_off:
      # computing the smooth cut-off 
      (s_ij, r_ij) = gen_coordinates(inputs, neigh_list, self.length)
      # s_ij = (n_snpa, n_points, max_nums_neighs)

      s_ij = smooth_cut_off(s_ij, self.r_cs, self.r_c)
      r_ij = r_ij*s_ij.unsqueeze(-1)

    else: 

      (s_ij, r_ij) = gen_coordinates(inputs, neigh_list, self.length)
      # s_ij = (n_snpa, n_points, max_nums_neighs)
      # r_ij = (n_snpa, n_points, max_nums_neighs, 3)

      # compute 1/r (this could be done in the gen_coordinates)
      s_ij = stable_inverse(s_ij) 
      r_ij = r_ij*s_ij.unsqueeze(-1)


    # embedding matrix 
    G = torch.zeros(n_snap, n_points, 
                    self.total_num_neighs, 
                    self.embedding_dim, 
                    device=device_in)

        # we need to do a nested loop
        # the exterior loop will be on the type of atoms centered
    
    for a_type in self.atom_types:
      # build a mask for the input types
      mask = (input_types == a_type)
      # this is the issue here.... 
      idx_type = torch.nonzero(mask[0,:]).reshape(-1)
      # (n_snap, num_atoms[a_type])

      # we extract all the s_ij corresponding to the 
      # 
      s_ij_local = s_ij[:, idx_type, :]
      # (n_snap, number blah,  max_num_neighs_total)

      # this seems to be redundant now
      s_ij_local = s_ij_local.view(shape_in[0], -1, self.total_num_neighs)
      # (n_snap, number blah,  max_num_neighs_total)

      # now we need to put them back 
      for k, (id_start, id_end) in enumerate(zip(self.max_num_neighs_cum[:-1],
                                                 self.max_num_neighs_cum[1:])):
        # print((id_start, id_end))

        # we choose the correct string for the dictionary
        if a_type <= k:
          idx_net = str(a_type) + str(self.atom_types[k])
        else:
          idx_net = str(self.atom_types[k]) + str(a_type)

        s_ij_local_temp = torch.reshape(s_ij_local[:,:,id_start:id_end], (-1,1))
        # todo after find the proper index for the embedding net
        G_ij = self.embed_nets[idx_net](s_ij_local_temp)
        # (n_snap*n_points[k]*max_num_neighs[k], descript_dim )

        G_ij = torch.reshape(G_ij, (n_snap, -1, self.max_num_neighs[k], 
                                     self.embedding_dim))
 
        # we assume the same ordering of the type of atoms in each 
        # batch THIS NEEDS TO BE FIXED!!!
        # storing the values in G
        G[:, idx_type, id_start:id_end,:] = G_ij
        # (n_snap, n_points, max_num_neighs_total, descript_dim )

    # concatenating 
    r_tilde_ij = torch.cat([s_ij.unsqueeze(-1), r_ij], axis = -1)
    # (n_snap, n_points, total_num_neighs, 4 )

    G2 = G[:,:,:,:self.dim_sub_net]
    # (n_snap, n_points, total_num_neighs, dim_sub_net )

    R = torch.matmul(torch.transpose(r_tilde_ij, 2, 3), G2)
    # (n_snap, n_points, 4, dim_sub_net)

    R1 = torch.matmul(r_tilde_ij, R)
    # (n_snap, n_points, total_num_neighs, dim_sub_net)

    D = torch.matmul(torch.transpose(G, 2, 3), R1)
    # (n_snap, n_points, descript_dim, dim_sub_net)

    return D



class DescriptorModuleRadial(torch.nn.Module):

  def __init__(self,
               n_points, # 
               length: torch.Tensor, # we need it to be a tensor
               atom_types = [0, 1, 2, 3], 
               max_num_neighs = [1, 3, 4], # this needs to be a tensor too
               descrip_dim = [1, 2, 4, 8, 16, 32],
               resnet = False,
               cut_offs = [2., 3.],
               dim_sub_net = 16,
               act_fn = F.relu,
               with_weight_skip = False, 
               with_batch_norm = False,
               with_smooth_cut_off = True,
               av = torch.tensor([0.0, 0.0], dtype = torch.float32),
               std = torch.tensor([1.0, 1.0], dtype = torch.float32),
               **kwargs):

    super().__init__()

    # save the different models in a dict
    # we can modify this so it is faster 
    module_dict = {}
    for i in range(len(atom_types)):
      for j in range(i+1):
        # we build such that j <= i
        # each network is index by a string "ij"
        # we make the assumption that we have up to 10 species
        module_dict[str(atom_types[j])+
                    str(atom_types[i])] = DenseChainNet(descrip_dim,
                                                        act_fn,
                                                        resnet, 
                                                        with_weight_skip, 
                                                        with_batch_norm)  

    # saving the dictionay on a moduledict object
    self.embed_nets = nn.ModuleDict(module_dict)

    self.length = length
    # self.descrip_dim    = torch.tensor(descrip_dim, dtype=torch.int32)
    # self.atom_types     = torch.tensor(atom_types, dtype=torch.int32)

    # testing this with lists
    self.descrip_dim    = descrip_dim
    self.atom_types     = atom_types

    self.max_num_neighs = max_num_neighs[:] # shallow copy


    self.embedding_dim = descrip_dim[-1]

    self.dim_sub_net = dim_sub_net

    self.with_smooth_cut_off = with_smooth_cut_off
    # saving the cut-offs 
    assert cut_offs[0] <= cut_offs[1]
    self.r_cs = torch.tensor(cut_offs[0])
    self.r_c = torch.tensor(cut_offs[1])

    max_num_neighs_cum = np.zeros(max_num_neighs.shape[0] + 1, 
                                  dtype = np.int32)
    max_num_neighs_cum[1:] = np.cumsum(max_num_neighs).astype(np.int32)

    self.total_num_neighs = max_num_neighs_cum[-1]

    # self.max_num_neighs_cum = torch.tensor(max_num_neighs_cum)
    self.max_num_neighs_cum = max_num_neighs_cum

    # normalization factors for s_ij
    self.av = av
    self.std = std

  def forward(self, inputs: torch.Tensor, 
                    input_types: torch.Tensor, 
                    neigh_list: torch.Tensor):

        # inputs     = (n_snapshots, n_points, 3)
        # atom_types = (n_snapshots, n_points)
        # neigh_list = (n_snapshots, n_points, max_num_0, max_num_1, ..., max_num_L)

    device_in = inputs.device

    shape_in = inputs.shape
    n_snap, n_points, dim = inputs.shape
    n_snap_neigh, n_points_neigh, max_num_neighs = neigh_list.shape

    # assert n_snap == n_snap_neigh
    # assert n_points == n_points_neigh 

    (s_ij, r_ij) = gen_coordinates(inputs, neigh_list, self.length)
    # s_ij = (n_snpa, n_points, max_nums_neighs)

    # computing the smooth cut-off 
    if self.with_smooth_cut_off:
      s_ij = smooth_cut_off(s_ij, self.r_cs, self.r_c)

    else:
      s_ij = stable_inverse(s_ij)


    # normalizing the input to help with convergence
    s_ij = (s_ij > 0.0)*(s_ij - self.av[0])/self.std[0]

    # embedding matrix 
    G = torch.zeros(n_snap, n_points, 
                    self.total_num_neighs, 
                    self.embedding_dim, device=device_in)

        # we need to do a nested loop
        # the exterior loop will be on the type of atoms centered
    
    for a_type in self.atom_types:
      # build a mask for the input types
      mask = (input_types == a_type)

      # this is an issue that needs to be properly fixed, 
      # we need to input this 
      idx_type = torch.nonzero(mask[0,:]).reshape(-1)
      # (n_snap, num_atoms[a_type])

      # we extract all the s_ij corresponding to the 
      # 
      s_ij_local = s_ij[:, idx_type, :]
      # (n_snap, number blah,  max_num_neighs_total)

      # this seems to be redundant now
      s_ij_local = s_ij_local.view(shape_in[0], -1, self.total_num_neighs)
      # (n_snap, number blah,  max_num_neighs_total)

      # now we need to put them back 
      for k, (id_start, id_end) in enumerate(zip(self.max_num_neighs_cum[:-1],
                                                 self.max_num_neighs_cum[1:])):
        # print((id_start, id_end))

        # we choose the correct string for the dictionary
        if a_type <= k:
          idx_net = str(a_type) + str(self.atom_types[k])
        else:
          idx_net = str(self.atom_types[k]) + str(a_type)

        s_ij_local_temp = torch.reshape(s_ij_local[:,:,id_start:id_end],
                                        (-1,1))
        
        # use the embedding net
        G_ij = self.embed_nets[idx_net](s_ij_local_temp)
        # (n_snap*n_points[k]*max_num_neighs[k], descript_dim )

        # putting the zero the not interacting ones 
        G_ij = G_ij*s_ij_local_temp

        G_ij = torch.reshape(G_ij, (n_snap, -1, self.max_num_neighs[k], 
                                     self.embedding_dim))
 
        # we assume the same ordering of the type of atoms in each 
        # batch THIS NEEDS TO BE FIXED!!!
        # storing the values in G
        G[:, idx_type, id_start:id_end,:] = G_ij
        # (n_snap, n_points, max_num_neighs_total, descript_dim )

    # we average the contribution of each all elements
    D = torch.sum(G, axis=2)

    return D
#Do we need an embedding module too? 


class DeepMDsimpleEnergyForces(torch.nn.Module):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               n_points, # 
               length: torch.Tensor, # we need it to be a tensor
               atom_types = [0, 1, 2, 3], 
               max_num_neighs = [1, 3, 4], # this needs to be a tensor too
               descript_dim = [1, 2, 4, 8, 16],
               fitting_dim = [64, 32, 16, 8, 8, 8, 8],
               resnet = False,
               cut_offs = [2., 3.],
               dim_sub_net = 8,
               act_fn = F.relu, 
               with_weight_skip=False, 
               with_batch_norm=False,
               with_smooth_cut_off = True,
               av = torch.tensor([0.0, 0.0], dtype = torch.float32),
               std = torch.tensor([1.0, 1.0], dtype = torch.float32),
               **kwargs):
    super().__init__()


    self.length = length
    # self.descript_dim    = torch.tensor(descript_dim, dtype=torch.int32)
    # self.atom_types     = torch.tensor(atom_types, dtype=torch.int32)

    # testing this with lists
    self.descript_dim   = descript_dim
    self.atom_types     = atom_types
    self.dim_sub_net    = dim_sub_net

    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descriptor_dim = descript_dim[-1]

    self.fitting_dim = fitting_dim.insert(0, descript_dim[-1]*dim_sub_net)
    self.total_num_neighs = np.sum(np.array(max_num_neighs))

    # print(fitting_dim)
    self.resnet = resnet
    # we may need to use the tanh here
    self.descriptor  = DescriptorModule(n_points, # 
                                        length, # we need it to be a tensor
                                        atom_types, 
                                        max_num_neighs, # this needs to be a tensor too
                                        descript_dim,
                                        resnet,
                                        cut_offs,
                                        dim_sub_net,
                                        act_fn, 
                                        with_weight_skip, 
                                        with_batch_norm,
                                        with_smooth_cut_off)

    # we may need to use the tanh especially here

    
    self.fittingNetwork = DenseChainNet(fitting_dim, 
                                        act_fn,
                                        resnet, 
                                        with_weight_skip, 
                                        with_batch_norm)

    self.linfitNet      = torch.nn.Linear(fitting_dim[-1], 1)    


  def forward(self, inputs: torch.Tensor, 
                    input_types: torch.Tensor, 
                    neigh_list: torch.Tensor):
      # we watch the inputs 

    n_snap, n_points, dim = inputs.shape

    inputs.requires_grad = True

    # in this case we are only considering the distances
    D = self.descriptor(inputs, input_types, neigh_list)
    # ((n_snap, n_points, descripto, dim_sub_net), 

    D = D.view(-1, self.descriptor_dim*self.dim_sub_net)
    # (Nsamples*Npoints, maxNumNeighs, descriptorDim)

    # print(D.shape)
    fit = self.fittingNetwork(D)
    fit = self.linfitNet(fit)
    # print(F1.size())

    Energy = torch.sum(fit.view(-1, n_points), dim=1)
    
    (Forces,) = torch.autograd.grad(-torch.sum(Energy), 
                                     inputs, create_graph=True, 
                                     allow_unused=True) 

    return Energy, Forces
 

class DeepMDradialEnergyForces(torch.nn.Module):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               n_points, # 
               length: torch.Tensor, # we need it to be a tensor
               atom_types = [0, 1, 2, 3], 
               max_num_neighs = [1, 3, 4], # this needs to be a tensor too
               descript_dim = [1, 2, 4, 8, 16],
               fitting_dim = [64, 32, 16, 8, 8, 8, 8],
               resnet = False,
               cut_offs = [2., 3.],
               dim_sub_net = 8,
               act_fn = F.relu, 
               with_weight_skip=False, 
               with_batch_norm=False,
               with_smooth_cut_off=True,
               av = torch.tensor([0.0, 0.0], dtype = torch.float32),
               std = torch.tensor([1.0, 1.0], dtype = torch.float32),
               **kwargs):
    super().__init__()


    self.length = length
    # self.descript_dim    = torch.tensor(descript_dim, dtype=torch.int32)
    # self.atom_types     = torch.tensor(atom_types, dtype=torch.int32)

    # testing this with lists
    self.descript_dim   = descript_dim
    self.atom_types     = atom_types
    self.dim_sub_net    = dim_sub_net

    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descriptor_dim = descript_dim[-1]

    self.fitting_dim = fitting_dim.insert(0, descript_dim[-1])
    self.total_num_neighs = np.sum(np.array(max_num_neighs))

    # print(fitting_dim)
    self.resnet = resnet
    # we may need to use the tanh here
    self.descriptor  = DescriptorModuleRadial(n_points, # 
                                        length, # we need it to be a tensor
                                        atom_types, 
                                        max_num_neighs, # this needs to be a tensor too
                                        descript_dim,
                                        resnet,
                                        cut_offs,
                                        dim_sub_net,
                                        act_fn,
                                        with_weight_skip, 
                                        with_batch_norm,
                                        with_smooth_cut_off)

    # we may need to use the tanh especially here

    
    self.fittingNetwork = DenseChainNet(fitting_dim, 
                                        act_fn,
                                        resnet, 
                                        with_weight_skip, 
                                        with_batch_norm)

    self.linfitNet      = torch.nn.Linear(fitting_dim[-1], 1)    


  def forward(self, inputs: torch.Tensor, 
                    input_types: torch.Tensor, 
                    neigh_list: torch.Tensor):
      # we watch the inputs 

    n_snap, n_points, dim = inputs.shape

    inputs.requires_grad = True

    # in this case we are only considering the distances
    D = self.descriptor(inputs, input_types, neigh_list)
    # ((n_snap, n_points, descripto, dim_sub_net), 

    D = D.view(-1, self.descriptor_dim)
    # (Nsamples*Npoints, maxNumNeighs, descriptorDim)

    # print(D.shape)
    fit = self.fittingNetwork(D)
    fit = self.linfitNet(fit)
    # print(F1.size())

    Energy = torch.sum(fit.view(-1, n_points), dim=1)
    
    (Forces,) = torch.autograd.grad(-torch.sum(Energy), 
                                     inputs, create_graph=True, 
                                     allow_unused=True) 

    return Energy, Forces
 