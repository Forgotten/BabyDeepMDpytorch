import torch
import numpy as np 


def genCoordinates(pos, neighborList, L, ave = torch.tensor([0.0, 0.0], dtype=torch.float32), 
									     std = torch.tensor([1.0, 1.0], dtype=torch.float32)):

	# extracting the size information 
	Nsamples, Npoints, maxNumNeighs = neighborList.size()

	assert pos.size()[0] == Nsamples
	assert pos.size()[1] == Npoints

	mask = (neighborList == -1)
	# change the neighbor (because of -1 can not index)
	neighborList[mask] = 0
	temp = pos.unsqueeze(-1).repeat(1,1, neighborList.shape[-1])
	temp = torch.gather(temp, 1, neighborList) - temp
	tempL = temp - L*torch.round(temp/L)
	Dist = (torch.abs(tempL) - ave[0])/std[0]
	DistInv = (1/(torch.abs(tempL))- ave[1])/std[1]
	Dist[mask] = 0.0
	DistInv[mask] = 0.0

	return (Dist, DistInv)
