# script to check that the network works as intended
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src')
from neighbor_list import comput_inter_list_type
from utilities_3d import gen_coordinates
from utilities_3d import smooth_cut_off
from utilities_3d import DenseChainNet
from networks_3d import DescriptorModule
from networks_3d import DeepMDradialEnergyForces
from data_gen_3d import gen_data_per_3d_mixed


plot_bool = False

np.random.seed(1234)
# we check that the construction of the moduledict is 
# properly performed :)
descript_dim = [1, 2, 4, 8, 16]
fitting_dim = [64, 32, 16, 8]
atom_types = [0, 1, 2]

## we create the testing example
n_val = 100

# number of particles
n_parts = 1
n_points_cell = 2
n_snaps = 1600 + n_val
n_points = n_parts**3
L = 1.
radious = 1.0


pointsArray,\
potentialArray,\
forcesArray = gen_data_per_3d_mixed(n_parts, n_points_cell, 10.0, 8.0, 
                          n_snaps, 0.2, L/n_parts, 
                          1.0, 0.0)

if plot_bool:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    idx_plot = 10
    ax.scatter(pointsArray[idx_plot,:,0],
               pointsArray[idx_plot,:,1], 
               pointsArray[idx_plot,:,2], marker='.',  s=5)

    ax.quiver(pointsArray[idx_plot,:,0],
              pointsArray[idx_plot,:,1], 
              pointsArray[idx_plot,:,2], 
              forcesArray[idx_plot,:,0],
              forcesArray[idx_plot,:,1], 
              forcesArray[idx_plot,:,2], length=1, color = 'black')


    plt.show()

mean_pot, std_pot = potentialArray.mean(), potentialArray.std()
mean_for, std_for = forcesArray.mean(), forcesArray.std()

print("Potential mean {:.6f}\t std {:.6f}".format(mean_pot, std_pot))
print("Forces mean {:.6f}\t std {:.6f}".format(mean_for, std_for))

potentialArray = (potentialArray - mean_pot)/std_pot
forcesArray = forcesArray/std_pot


# backwards compatibility with only one species of atom
r_in_torch = torch.tensor(pointsArray, dtype=torch.float32)
input_types_torch = torch.tensor(np.zeros(pointsArray.shape[:-1]), 
                                 dtype=torch.int64)

# define the data (we need to detach it from the computation)
energy_train = torch.tensor(potentialArray.reshape((-1,)), 
                            dtype = torch.float32)  
forces_train = torch.tensor(forcesArray, 
                            dtype = torch.float32)  

L_torch = torch.tensor(L, dtype = torch.float32)

descript_dim = [1, 2, 4, 8, 16, 32]
# fitting_dim = [64, 32, 16, 8]
fitting_dim = [32, 32, 32, 32]

max_num_neighs_type = np.array([2])
# only one type
a_types = [0]

av = torch.tensor([0.0, 0.0], dtype = torch.float32)
std = torch.tensor([1.0, 1.0], dtype = torch.float32)

inter_list = comput_inter_list_type(pointsArray, 
                                    input_types_torch.numpy(), 
                                    L, radious, max_num_neighs_type)

inter_list_torch = torch.tensor(inter_list, dtype= torch.int64)

r_c = 1.0
r_cs = 0.5

DeepMD = DeepMDradialEnergyForces(n_points, # 
               L_torch, # we need it to be a tensor
               a_types, 
               max_num_neighs_type, # this needs to be a tensor too
               descript_dim,
               fitting_dim,
               False,
               [r_cs, r_c],
               4,
               torch.selu, # use other activations/ torch.selu
               False, 
               False, # batch normalization at each layer
               True, # smooth-cut off
               av,
               std)  
                       


# DeepMD = torch.jit.script(DeepMDradialEnergyForces(n_points, # 
#                L_torch, # we need it to be a tensor
#                a_types, 
#                max_num_neighs_type, # this needs to be a tensor too
#                descript_dim,
#                fitting_dim,
#                False,
#                [r_cs, r_c],
#                8,
#                F.relu)).to(device)

# moving the model to a GPU

print("Moving the model to a GPU (if available)")
ngpu = 0
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu") 

print(device)

DeepMD = DeepMD.to(device)

r_in_cuda,\
input_types_cuda,\
inter_list_cuda= r_in_torch[:2,:,:].to(device), \
                  input_types_torch[:2,:].to(device), \
                  inter_list_torch[:2,:,:].to(device)

energyNN, forcesNN = DeepMD(r_in_cuda, input_types_cuda, inter_list_cuda)

r_in_val,\
input_types_val,\
inter_list_val= r_in_torch[-n_val:,:,:].to(device), \
                  input_types_torch[-n_val:,:].to(device), \
                  inter_list_torch[-n_val:,:,:].to(device)

energy_val = energy_train[-n_val:].to(device)
forces_val = forces_train [-n_val:,:,:].to(device)

### setting up the optimization parameters
print("Setting up the optimization parameters")

# specify loss function
criterion = torch.nn.MSELoss(reduction='mean')

# specifying the learning rate
learningRate =1.e-3
lr_steps = 3
gamma = 0.99

# specify optimizer
optimizer = torch.optim.Adam(DeepMD.parameters(), lr=learningRate)

# specidy scheduler
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=lr_steps, gamma=gamma)


# creating the data sets (we don't consider testing so far)
datasetTrain = torch.utils.data.TensorDataset(r_in_torch, 
                                              input_types_torch,
                                              energy_train,
                                              forces_train) 

batchSize = 16

# creating the data loader
dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, 
                                              batch_size=batchSize,
                                              shuffle=True,
                                              num_workers=0)#, 
                                              #pin_memory=True)

Nepochs = 1000
## these parameters need to be adjusted 
weightE_init = 10.
weightE_final = 0.00001
 
weightF_init = 0.001
weightF_final = 10.

print("Starting the training loop")
for epoch in range(1, Nepochs+1):
    # monitor training loss
    train_loss = 0.0
    DeepMD.train()
    # monitoring time elapsed
    start = time.time()

    # this shouldn't be linear, it should be exponential
    weights_rate = epoch/Nepochs
    weightE = weightE_init*(1-weights_rate) + weightE_final*weights_rate
    weightF = weightF_init*(1-weights_rate) + weightF_final*weights_rate


    # rate_E = np.power(weightE_final/weightF_init, 1/Nepochs)
    # rate_F = np.power(weightF_final/weightF_init, 1/Nepochs) 

    # weightE = weightE_init*np.power(rate_E,epoch)
    # weightF = weightF_init*np.power(rate_F,epoch)

    ###################
    # train the model #
    ###################
    for pos, atom_type, energy_batch, forces_batch in dataloaderTrain:

        # computing the interaction list (via numba)
        neighbor_list = comput_inter_list_type(pos.numpy(), atom_type.numpy(), 
                                    L, radious, max_num_neighs_type)
        # moving list to pytorch and moving to device
        neighbor_list = torch.tensor(neighbor_list).to(device)
        #send to the device (either cpu or gpu)
        pos, atom_type, energy_batch, forces_batch = pos.to(device), \
                                         atom_type.to(device), \
                                         energy_batch.to(device), \
                                         forces_batch.to(device)
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        (energyNN, forcesNN) = DeepMD(pos, atom_type, neighbor_list)
        # calculate the loss for the enery
        lossE = criterion(energyNN, energy_batch)
        # calculate the loss for the enery
        lossF = criterion(forcesNN, forces_batch)
        # adding the losses together
        loss =   weightF*lossF  + weightE*lossE
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()
     
    # monitoring the elapsed time          
    end = time.time()

    # advance the step in the scheduler
    exp_lr_scheduler.step() 

    # print avg training statistics 
    train_loss = train_loss/len(dataloaderTrain)
    print('Epoch: {} \tLearning rate: {:.6f}'.format(epoch, 
        optimizer.param_groups[0]['lr']), flush=True)

    print('Training Loss: {:.6f}\t Time per epoch: {:.3f} [s]'.format(
        train_loss,
        end-start), flush=True)

    if epoch % 10 == 0:

        DeepMD.eval()

        energyNN, forcesNN = DeepMD(r_in_val, input_types_val, inter_list_val)

        err_e = torch.sum(torch.square(energyNN - energy_val))/\
                torch.sum(torch.square(energy_val))

        err_f = torch.sum(torch.square(forcesNN - forces_val))/\
                torch.sum(torch.square(forces_val)) 

        print('Validation Loss Energy: {:.3e}\t Forces: {:.3e} [s]'.format(
        torch.sqrt(err_e).cpu().detach().numpy(),
        torch.sqrt(err_f).cpu().detach().numpy()), flush=True)

        print('L2 norm Energy: {:.3e}\t Forces: {:.3e} [s]'.format(
        torch.sqrt(torch.sum(torch.square(energyNN))).cpu().detach().numpy(),
        torch.sqrt(torch.sum(torch.square(forcesNN))).cpu().detach().numpy()), flush=True)

        print('Weights Energy: {:.3e}\t Forces: {:.3e} [s]'.format(
        weightE,
        weightF), flush=True)


descript_test = DeepMD.descriptor(pos, atom_type, neighbor_list)

input_types = atom_type 

a_type = DeepMD.atom_types[0]

mask = (input_types == a_type)


### play ground for the descriptor
inputs, input_types, neigh_list =\
r_in_cuda, input_types_cuda, inter_list_cuda

n_snap, n_points, dim = inputs.shape
device_in = inputs.device
shape_in = inputs.shape

(s_ij, r_ij) = gen_coordinates(inputs, neigh_list, DeepMD.length)


# embedding matrix 
G = torch.zeros(n_snap, n_points, 
                DeepMD.descriptor.total_num_neighs, 
                DeepMD.descriptor.embedding_dim, device=device_in)

    # we need to do a nested loop
    # the exterior loop will be on the type of atoms centered

for a_type in DeepMD.descriptor.atom_types:
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
  s_ij_local = s_ij_local.view(shape_in[0], -1, DeepMD.descriptor.total_num_neighs)
  # (n_snap, number blah,  max_num_neighs_total)

  # now we need to put them back 
  for k, (id_start, id_end) in enumerate(zip(DeepMD.descriptor.max_num_neighs_cum[:-1],
                                             DeepMD.descriptor.max_num_neighs_cum[1:])):
    # print((id_start, id_end))

    # we choose the correct string for the dictionary
    if a_type <= k:
      idx_net = str(a_type) + str(DeepMD.descriptor.atom_types[k])
    else:
      idx_net = str(DeepMD.descriptor.atom_types[k]) + str(a_type)

    s_ij_local_temp = torch.reshape(s_ij_local[:,:,id_start:id_end],
                                    (-1,1))
    
    # use the embedding net
    G_ij = DeepMD.descriptor.embed_nets[idx_net](s_ij_local_temp)
    # (n_snap*n_points[k]*max_num_neighs[k], descript_dim )

    G_ij = G_ij*s_ij_local_temp

    G_ij = torch.reshape(G_ij, (n_snap, -1, DeepMD.descriptor.max_num_neighs[k], 
                                 DeepMD.descriptor.embedding_dim))


    # we assume the same ordering of the type of atoms in each 
    # batch THIS NEEDS TO BE FIXED!!!
    # storing the values in G
    G[:, idx_type, id_start:id_end,:] = G_ij
    # (n_snap, n_points, max_num_neighs_total, descript_dim )

# we average the contribution of each all elements
D = torch.sum(G, axis=2)