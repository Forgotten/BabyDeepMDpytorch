import torch
import numpy as np 
from neighbor_list import computInterListOpt
# def genCoordinates(pos : torch.Tensor, 
# 				   neighborList: torch.Tensor, 
# 				   L : torch.Tensor, 
# 				   ave = torch.tensor([0.0, 0.0], dtype=torch.float32), 
# 				   std = torch.tensor([1.0, 1.0], dtype=torch.float32)):

# 	# extracting the size information 
# 	Nsamples, Npoints, maxNumNeighs = neighborList.size()

# 	assert pos.size()[0] == Nsamples
# 	assert pos.size()[1] == Npoints

# 	mask = (neighborList == -1)
# 	# change the neighbor (because of -1 can not index)
# 	neighborList[mask] = 0
# 	temp = pos.unsqueeze(-1).repeat(1,1, neighborList.shape[-1])
# 	temp = torch.gather(temp, 1, neighborList) - temp
# 	tempL = temp - L*torch.round(temp/L)
# 	Dist = (torch.abs(tempL) - ave[0])/std[0]
# 	DistInv = (1/(torch.abs(tempL))- ave[1])/std[1]

# 	Dist[mask] = 0.0
# 	DistInv[mask] = 0.0

# 	return (Dist, DistInv)

@torch.jit.script
def genCoordinates(pos : torch.Tensor, 
				   	  neighborList: torch.Tensor, 
				      L : torch.Tensor, 
				      ave = torch.tensor([0.0, 0.0], dtype=torch.float32), 
				   std = torch.tensor([1.0, 1.0], dtype=torch.float32)):

	# extracting the size information 
	Nsamples, Npoints, maxNumNeighs = neighborList.size()

	assert pos.size()[0] == Nsamples
	assert pos.size()[1] == Npoints

	mask = (neighborList == -1)
	# change the neighbor (because of -1 can not index)
	neighborList *= (1 - mask.int())
	temp = pos.unsqueeze(-1).repeat(1,1, neighborList.shape[-1])
	temp = torch.gather(temp, 1, neighborList) - temp
	tempL = temp - L*torch.round(temp/L)
	Dist = (torch.abs(tempL) - ave[0])/std[0]
	DistInv = (1/(torch.abs(tempL))- ave[1])/std[1]

	zeroDummy = torch.zeros_like(Dist)
	Dist = torch.where(mask, zeroDummy, Dist) 
	DistInv = torch.where(mask, zeroDummy, DistInv) 

	return (Dist, DistInv)


def trainEnergy(model, optimizer, criterion, 
				dataloaderTrain, Nepochs, 
				L,radious, maxNumNeighs, device):
	print("Entering the training Stage")
	
	for epoch in range(1, Nepochs+1):
	  # monitor training loss
	  train_loss = 0.0
	  model.train()
	  ###################
	  # train the model #
	  ###################
	  for pos, energy in dataloaderTrain:
	
	    # computing the interaction list (via numba)
	    neighbor_list = computInterListOpt(pos.numpy(), L,  
	                                        radious, maxNumNeighs)
	    # moving list to pytorch and moving to device
	    neighbor_list = torch.tensor(neighbor_list).to(device)
	    #send to the device (either cpu or gpu)
	    pos, energy = pos.to(device), energy.to(device)
	    # clear the gradients of all optimized variables
	    optimizer.zero_grad()
	    # forward pass: compute predicted outputs by passing inputs to the model
	    energyNN = model(pos, neighbor_list)
	    # calculate the loss
	    loss = criterion(energyNN, energy)
	    # backward pass: compute gradient of the loss with respect to model parameters
	    loss.backward()
	    # perform a single optimization step (parameter update)
	    optimizer.step()
	    # update running training loss
	    train_loss += loss.item()
	            
	  # print avg training statistics 
	  train_loss = train_loss/len(dataloaderTrain)
	  print('Epoch: {} \tTraining Loss: {:.6f}'.format(
	      epoch, 
	      train_loss
	      ), flush=True)

