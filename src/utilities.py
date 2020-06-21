import torch
import numpy as np 
from neighbor_list import computInterListOpt
import time 

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
	# we create a new temporal neighbor list 
	# otherwise we modify neighborList outside the scope 
	# of this function
	neighborListTemp = neighborList*(1 - mask.int())

	zeroDummy = torch.zeros_like(neighborList, dtype=torch.float32)

	temp = pos.unsqueeze(-1).repeat(1,1, neighborListTemp.shape[-1])
	temp2 = torch.gather(temp, 1, neighborListTemp)  
	temp2Filtered = torch.where(mask, zeroDummy, temp2) 
	temp = temp2Filtered - temp
	tempL = temp - L*torch.round(temp/L)
	Dist = (torch.abs(tempL) - ave[0])/std[0]
	DistInv = (1./(torch.abs(tempL))- ave[1])/std[1]

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

	  # monitoring time elapsed
	  start = time.time()
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
	            
	  # monitoring the elapsed time          
	  end = time.time()
	  # print avg training statistics 
	  train_loss = train_loss/len(dataloaderTrain)
	  print('Epoch: {} \tTraining Loss: {:.10f} \t Time per epoch: {:.3f} [s]'.format(
	      epoch, 
	      train_loss, end-start
	      ), flush=True)

