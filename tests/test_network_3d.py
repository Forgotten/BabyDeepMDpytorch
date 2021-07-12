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
from networks_3d import DeepMDsimpleEnergyForces

# we check that the construction of the moduledict is 
# properly performed :)
descript_dim = [1, 2, 4, 8, 16]
fitting_dim = [64, 32, 16, 8]
atom_types = [0, 1, 2]

## we create the testing example

# number of particles
n_parts = 3
n_snaps = 10
n_points = n_parts**3
L = 1
r_in =  L*np.random.rand(n_snaps, n_parts**3, 3)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(r_in[0,:,0], r_in[0,:,1], r_in[0,:,2], marker='.',  s=10)
ax.scatter(r_in[1,:,0], r_in[1,:,1], r_in[1,:,2], marker='.',  s=10)
plt.show()

# creating the atom type array
atom_type = []
for k in range(n_parts):
    atom_type += [k]*n_parts**2
atom_type = np.array(atom_type, dtype=np.int64).reshape(1, n_parts**3)

atom_type = atom_type*np.ones((n_snaps, 1), dtype = np.int64)


#creating the data
L_torch = torch.tensor(L)
r_in_torch = torch.tensor(r_in)
r_in_torch.requires_grad = True 
atom_type_torch = torch.tensor(atom_type)


dist = r_in_torch.view(n_snaps, 1, n_points, 3)\
     - r_in_torch.view(n_snaps, n_points,1, 3)

mask = dist > 1e-3
dist = torch.where(mask,dist, torch.zeros_like(dist, 
                                               dtype=torch.float64))
dist_per =  dist - L_torch*torch.round(dist/L_torch) 

dist_per = torch.sqrt(torch.sum(torch.square(dist_per), axis = -1 ))

atom_type_torch = 0.5*atom_type_torch + 2
atom_type_torch_expand = atom_type_torch.view(n_snaps, 1, n_points)\
                       * atom_type_torch.view(n_snaps, n_points,1)

# mask = dist_per > 1e-3
# dist_per = torch.where(mask,dist_per, torch.zeros_like(dist_per, dtype = torch.float64) )

pot = -torch.exp(-atom_type_torch_expand*dist_per )

upper_potential = torch.triu(pot, 1)
energy = torch.sum(upper_potential, axis=[1,2])

(forces,) = torch.autograd.grad(-torch.sum(energy), r_in_torch)

radious = 0.30

max_num_neighs_type = np.array([6, 6, 6], dtype=np.int64)
inter_list = comput_inter_list_type(r_in, atom_type, 
                                    L, radious, max_num_neighs_type)


r_in_torch = torch.tensor(r_in, dtype=torch.float32)
input_types = torch.tensor(atom_type, dtype=torch.int64)
inter_list_torch = torch.tensor(inter_list, dtype=torch.int64)

descript_dim = [1, 2, 4, 8, 16]
fitting_dim = [64, 32, 16, 8]
a_types = [0, 1, 2]

r_c = 0.30
r_cs = 0.25

DeepMD = DeepMDsimpleEnergyForces(n_points, # 
               L_torch, # we need it to be a tensor
               a_types, 
               max_num_neighs_type, # this needs to be a tensor too
               descript_dim,
               fitting_dim,
               False,
               [r_cs, r_c],
               8,
               F.relu)


# model = torch.jit.script(DeepMDsimpleEnergyForces(n_points, # 
#                L_torch, # we need it to be a tensor
#                a_types, 
#                max_num_neighs_type, # this needs to be a tensor too
#                descript_dim,
#                fitting_dim,
#                False,
#                [r_cs, r_c],
#                8,
#                F.relu)).to(device)

# energyNN, forcesNN = DeepMD(r_in_torch, input_types, inter_list_torch)
# yey! at lease this produces something! 

### setting up the optimization parameters

ngpu = 0 
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu") 

DeepMD = DeepMD.to(device)

# specify loss function
criterion = torch.nn.MSELoss(reduction='mean')

# specifying the learning rate
learningRate =1.e-3

# specify optimizer
optimizer = torch.optim.Adam(DeepMD.parameters(), lr=learningRate)


# define the data (we need to detach it from the computation)
r_in_torch = torch.tensor(r_in, dtype = torch.float32)  
energy_train = torch.tensor(energy.detach().numpy(), dtype = torch.float32)  
forces_train = torch.tensor(forces.detach().numpy(), dtype = torch.float32)  


# creating the data sets (we don't consider testing so far)
datasetTrain = torch.utils.data.TensorDataset(r_in_torch, 
                                              input_types,
                                              energy_train,
                                              forces_train) 

batchSize = 2 

# creating the data loader
dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, 
                                              batch_size=batchSize,
                                              shuffle=True,
                                              num_workers=4)

Nepochs = 1000

for epoch in range(1, Nepochs+1):
    # monitor training loss
    train_loss = 0.0
    DeepMD.train()
    # monitoring time elapsed
    start = time.time()

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
        loss =  lossF  + lossE
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()
            
    # monitoring the elapsed time          
    end = time.time()
    # print avg training statistics 
    train_loss = train_loss/len(dataloaderTrain)
    print('Epoch: {} \tTraining Loss: {:.10f} \t Time per epoch: {:.3f} [s]'.format(
        epoch, 
        train_loss, end-start
        ), flush=True)
