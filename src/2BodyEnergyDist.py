# typical imports
# we have a simplified deep MD using only the radial information
# and the inverse of the radial information. We don't allow the particules to be
# too close, we allow biases in the pyramids and we multiply the outcome by 
# the descriptor income (in order to preserve the zeros)
# This version supports an inhomogeneous number of particules, however we need to 
# provide a neighboor list. 

import torch
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import h5py
import sys
import json

from data_gen_1d import genDataYukawaPer
from neighbor_list import computInterListOpt
from networks import DeepMDsimpleEnergy
from utilities import genCoordinates
from utilities import trainEnergy


import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

nameScript = sys.argv[0].split('/')[-1]

# we are going to give all the arguments using a Json file
nameJson = sys.argv[1]
print("=================================================")
print("Executing " + nameScript + " following " + nameJson, flush = True)
print("=================================================")


# opening Json file # TODO:write a function to manipulate all this
jsonFile = open(nameJson) 
data = json.load(jsonFile)   

# loading the input data from the json file

# These one are deprecated we will continue using them for 
# no to call old data sets
Ncells = data["Ncells"]                  # number of cells
Np = data["Np"]                          # number of particules per cell
Nsamples = data["Nsamples"]              # number of samples 
Lcell = data["lengthCell"]               # lenght of each cell
mu = data["mu"]                          # the parameter mu of the potential
minDelta = data["minDelta"]
filterNet = data["filterNet"]
fittingNet = data["fittingNet"]
seed = data["seed"]
batchSize = data["batchSize"]
epochsPerStair = data["epochsPerStair"]
learningRate = data["learningRate"]
decayRate = data["decayRate"]
dataFolder = data["dataFolder"]
loadFile = data["loadFile"]
Nepochs = data["numberEpoch"]

# number of Gpus to be used
ngpu = 1
# defining device for the computation 
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# we need to add a few new parameters
maxNumNeighs = data["maxNumNeighbors"]
radious = data["radious"]

# the ones not used yet
potentialType = data["potentialType"]

print("We are using the random seed %d"%(seed))
torch.manual_seed(seed)

dataFile = dataFolder + "data_"+ potentialType + \
                        "_Ncells_" + str(Ncells) + \
                        "_Np_" + str(Np) + \
                        "_mu_" + str(mu) + \
                        "_minDelta_%.4f"%(minDelta) + \
                        "_Nsamples_" + str(Nsamples) + ".h5"

checkFolder  = "checkpoints/"
checkFile = checkFolder + "checkpoint_" + nameScript + \
                          "potential_"+ potentialType + \
                          "_Json_" + nameJson + \
                          "_Ncells_" + str(Ncells) + \
                          "_Np_" + str(Np) + \
                          "_mu_" + str(mu) + \
                          "_minDelta_%.4f"%(minDelta) + \
                          "_Nsamples_" + str(Nsamples)

print("Using data in %s"%(dataFile))

# TODO: add the path file for this one
assert potentialType == "Periodic"
print("We only consider periodic potentials in this case")

# if the file doesn't exist we create it
if not path.exists(dataFile):
  # TODO: encapsulate all this in a function
  print("Data file does not exist, we create a new one")

  pointsArray, \
  potentialArray, \
  forcesArray  = genDataYukawaPer(Ncells, Np, mu, Nsamples, minDelta, Lcell)
  
  hf = h5py.File(dataFile, 'w') 
  
  hf.create_dataset('points', data=pointsArray)   
  hf.create_dataset('potential', data=potentialArray) 
  hf.create_dataset('forces', data=forcesArray)
  
  hf.close()

# extracting the data
hf = h5py.File(dataFile, 'r')

pointsArray = hf['points'][:]
forcesArray = hf['forces'][:]
potentialArray = hf['potential'][:]

# normalization of the data

if loadFile: 
  # if we are loading a file, the normalization needs to be 
  # properly defined 
  forcesMean = data["forcesMean"]
  forcesStd = data["forcesStd"]
else: 
  forcesMean = np.mean(forcesArray)
  forcesStd = np.std(forcesArray)

print("mean of the forces is %.8f"%(forcesMean))
print("std of the forces is %.8f"%(forcesStd))

potentialArray /= forcesStd
forcesArray -= forcesMean
forcesArray /= forcesStd

lengthCell = torch.tensor(Ncells*Lcell, 
                          dtype = torch.float32, 
                          requires_grad=False) 

# Npoints = torch.tensor(Ncells*Np, 
#                        dtype = torch.float32, 
#                        requires_grad=False) 

Npoints = Ncells*Np

pointsArrayTorch = torch.tensor(pointsArray, 
                                dtype = torch.float32)
potentialArrayTorch = torch.tensor(potentialArray, 
                                   dtype = torch.float32)
# creating the data sets (we don't consider testing so far)
datasetTrain = torch.utils.data.TensorDataset(pointsArrayTorch, 
                                              potentialArrayTorch) 

# creating the data loader
dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, 
                                              batch_size=batchSize,
                                              shuffle=True,
                                              num_workers=4)

# computing an estimate of ave and std
pointsArrayTorchSmall = pointsArrayTorch[:16, :]

pointsnumpy = pointsArrayTorchSmall.numpy()

neighbor_list = computInterListOpt(pointsnumpy, Lcell*Ncells,  
                                            radious, maxNumNeighs)

neighbor_list = torch.tensor(neighbor_list)

(dist, distInv) = genCoordinates(pointsArrayTorchSmall, 
                                  neighbor_list, lengthCell)

# we compute the mean and std (only the positive values)
# given that the zero values are just padding. 
ave = torch.stack([torch.mean(dist[dist>0]), 
                  torch.mean(distInv[distInv>0])])

std = torch.stack([torch.std(dist[dist>0]), 
                  torch.std(distInv[distInv>0])])

# ## compute the mean and std for the samples (or simply)
# # load them from a file 
# ave = torch.tensor([0.0, 0.0], dtype=torch.float32)
# std = torch.tensor([1.0, 1.0], dtype=torch.float32)

print("building the model")
# building the model 
model = DeepMDsimpleEnergy(Npoints, lengthCell, maxNumNeighs,
                           filterNet, fittingNet, True,
                           ave, std).to(device)

model = torch.jit.script(DeepMDsimpleEnergy(Npoints, lengthCell, maxNumNeighs,
                           filterNet, fittingNet, True,
                           ave, std)).to(device)

# specify loss function
criterion = torch.nn.MSELoss(reduction='mean')

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TODO add the custom training procedure

trainEnergy(model, optimizer, 
            criterion, dataloaderTrain, Nepochs, Lcell*Ncells,radious, maxNumNeighs, device)
# for epoch in range(1, Nepochs+1):
#     # monitor training loss
#     train_loss = 0.0
#     model.train()
#     ###################
#     # train the model #
#     ###################
#     for pos, energy in dataloaderTrain:

#         # computing the interaction list (via numba)
#         neighbor_list = computInterListOpt(pos.numpy(), Lcell*Ncells,  
#                                             radious, maxNumNeighs)
#         # moving list to pytorch and moving to device
#         neighbor_list = torch.tensor(neighbor_list).to(device)
#         #send to the device (either cpu or gpu)
#         pos, energy = pos.to(device), energy.to(device)
#         # clear the gradients of all optimized variables
#         optimizer.zero_grad()
#         # forward pass: compute predicted outputs by passing inputs to the model
#         energyNN = model(pos, neighbor_list)
#         # calculate the loss
#         loss = criterion(energyNN, energy)
#         # backward pass: compute gradient of the loss with respect to model parameters
#         loss.backward()
#         # perform a single optimization step (parameter update)
#         optimizer.step()
#         # update running training loss
#         train_loss += loss.item()
            
#     # print avg training statistics 
#     train_loss = train_loss/len(dataloaderTrain)
#     print('Epoch: {} \tTraining Loss: {:.6f}'.format(
#         epoch, 
#         train_loss
#         ), flush=True)



#### tests


