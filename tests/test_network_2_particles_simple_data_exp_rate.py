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
from data_gen_3d import gen_data_per_3d_mixed


plot_bool = False

np.random.seed(1234)

## we create the testing example
n_val = 100

# number of particles
n_parts = 1
n_points_cell = 2
n_snaps = 800 + n_val
n_points = (n_parts**3)*n_points_cell
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

descript_dim = [1, 2, 4, 8, 16]
fitting_dim = [64, 64, 64, 64]

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

DeepMD = DeepMDsimpleEnergyForces(n_points, # 
               L_torch, # we need it to be a tensor
               a_types, 
               max_num_neighs_type, # this needs to be a tensor too
               descript_dim,
               fitting_dim,
               False,
               [r_cs, r_c],
               4,
               torch.nn.GELU(), # torch.selu/ nn.SiLU()/ torch.tanh / torch.relu / torch.nn.GELU()
               False, 
               False, # batch normalization at each layer
               True, # smooth-cut off
               av,
               std)  
                       

# DeepMD = torch.jit.script(DeepMDsimpleEnergyForces(n_points, # 
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
ngpu = 1
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
lr_steps = 4
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
                                              num_workers=4)

Nepochs = 1000
## these parameters need to be adjusted 
weightE_init = 0.2
weightE_final = 8
 
weightF_init = 100
weightF_final = 1

print("Starting the training loop")
for epoch in range(1, Nepochs+1):
    # monitor training loss
    train_loss = 0.0
    DeepMD.train()
    # monitoring time elapsed
    start = time.time()

    rate_E = np.power(weightE_final/weightE_init, 1/Nepochs)
    rate_F = np.power(weightF_final/weightF_init, 1/Nepochs) 

    weightE = weightE_init*np.power(rate_E,epoch)
    weightF = weightF_init*np.power(rate_F,epoch)

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
