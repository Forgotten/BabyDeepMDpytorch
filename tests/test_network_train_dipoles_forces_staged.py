# script to check that the network works as intended
# we train a network using dipole type of data 
# using a multi-stage optimization loop with different batchsizes


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
from data_gen_3d import gen_data_exp_per_3d_charge

ngpu = 1 
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu") 
print(device)

# number of particles
n_cells = 3
n_points_cell = 1
n_val = 100
n_snaps = 3200 + n_val
n_points = n_cells**3*(2*n_points_cell)
L = 3.

min_delta = 0.5

mu = 10.0
# structured 
r_in,\
charge_array,\
potential_array,\
forces_array = gen_data_exp_per_3d_charge(n_cells, n_points_cell, 
                                          n_snaps, L/n_cells, 
                                          min_delta, mu = mu)

plot_bool = False 

if plot_bool:

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(r_in[0,:,0], r_in[0,:,1], r_in[0,:,2], marker='.',  s=10)
    ax.scatter(r_in[1,:,0], r_in[1,:,1], r_in[1,:,2], marker='.',  s=10)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    idx_plot = 10
    ax.scatter(r_in[idx_plot,:,0],
               r_in[idx_plot,:,1], 
               r_in[idx_plot,:,2], marker='.',  s=5)

    ax.quiver(r_in[idx_plot,:,0],
              r_in[idx_plot,:,1], 
              r_in[idx_plot,:,2], 
              forces_array[idx_plot,:,0],
              forces_array[idx_plot,:,1], 
              forces_array[idx_plot,:,2], length=1, color = 'black')


    plt.show()


# creating the atom type array
atom_type = np.array(charge_array == 1.,\
                     dtype=np.int64).reshape((n_snaps, n_points))

#creating the data
L_torch = torch.tensor(L)
r_in_torch = torch.tensor(r_in, dtype=torch.float64)
r_in_torch.requires_grad = True 
atom_type_torch = torch.tensor(atom_type)


##################################################################
####   Setting up the training data   ############################
##################################################################

radious = L/2

max_num_neighs_type = np.array([n_points//2, n_points//2], dtype=np.int64)

r_in_torch = torch.tensor(r_in, dtype=torch.float32)
input_types_torch = torch.tensor(atom_type, 
                                 dtype=torch.int64)

# define the data (we need to detach it from the computation)
energy_train = torch.tensor(potential_array.reshape((-1,)), 
                            dtype = torch.float32)  
forces_train = torch.tensor(forces_array, 
                            dtype = torch.float32)  


###### Validation data 

inter_list_val_np = comput_inter_list_type(r_in[-n_val:,:,:], atom_type[-n_val:,:], 
                                    L, radious, max_num_neighs_type)

inter_list_torch = torch.tensor(inter_list_val_np, dtype=torch.int64)


r_in_val,\
input_types_val,\
inter_list_val= r_in_torch[-n_val:,:,:].to(device), \
                  input_types_torch[-n_val:,:].to(device), \
                  inter_list_torch[-n_val:,:,:].to(device)

energy_val = energy_train[-n_val:].to(device)
forces_val = forces_train [-n_val:,:,:].to(device)


####### Parameters of the Model

descript_dim = [1, 2, 4, 8, 16, 32]
fitting_dim = [64, 64, 64, 64]
a_types = [0, 1]

av = torch.tensor([0.0, 0.0], dtype = torch.float32)
std = torch.tensor([1.0, 1.0], dtype = torch.float32)

r_c = radious
r_cs = 4/5*(r_c)

DeepMD = DeepMDsimpleEnergyForces(n_points, # 
               L_torch, # we need it to be a tensor
               a_types, 
               max_num_neighs_type, # this needs to be a tensor too
               descript_dim,
               fitting_dim,
               False,
               [r_cs, r_c],
               2,
               torch.nn.GELU(), # torch.selu/ nn.SiLU()/ torch.tanh / torch.relu / torch.nn.GELU()
               True, 
               False, # batch normalization at each layer
               True, # smooth-cut off
               av,
               std)  

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

### Setting up the optimization parameters

DeepMD = DeepMD.to(device)

### setting up the optimization parameters
print("Setting up the optimization parameters")

# specify loss function
criterion = torch.nn.MSELoss(reduction='mean')

# specifying the learning rate
learningRate =1.e-3
lr_steps = 5
gamma = 0.98

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


batch_sizes_array = [2, 4, 8, 16, 32, 64, 128]
n_epochs_array = [12, 25, 50, 100, 200, 400, 800]


for ii, (batch_size, n_epochs) in enumerate(zip(batch_sizes_array,\
                                                n_epochs_array)):

    print("Starting stage %d"%ii)
    print("batch size = %d"%batch_size)
    # creating the data loader
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, 
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)

    ## these parameters need to be adjusted 
    weightE_init = 0.
    weightE_final = 0.0
     
    weightF_init = 1.0
    weightF_final = 1.0

    print("Starting the training loop")
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        DeepMD.train()
        # monitoring time elapsed
        start = time.time()

        # this shouldn't be linear, it should be exponential
        weights_rate = epoch/n_epochs
        weightE = weightE_init*(1-weights_rate) + weightE_final*weights_rate
        weightF = weightF_init*(1-weights_rate) + weightF_final*weights_rate

        # rate_E = np.power(weightE_final/weightF_init, 1/n_epochs)
        # rate_F = np.power(weightF_final/weightF_init, 1/n_epochs) 

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


torch.save(DeepMD, 'model_deep_md_forces_dipole.pt')