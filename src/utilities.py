import torch
import numpy as np 


def genCoordinates(pos, neighborList, L, ave = torch.tensor([0.0, 0.0], dtype=torch.float32), 
									     std = torch.tensor([1.0, 1.0], dtype=torch.float32)):

	# extracting the size information 
	Nsamples, Npoints, maxNumNeighs = neighborList.size()

	assert pos.size()[0] == Nsamples
	assert pos.size()[1] == Npoints

	# we alocate the distance tensor
	Dist = torch.tensor(np.zeros((Nsamples, Npoints, maxNumNeighs)), 
					    dtype= torch.float32, 
					    device = pos.device) 
	# we alocate the inverse distance tensor
	DistInv = torch.tensor(np.zeros((Nsamples, Npoints, maxNumNeighs)), 
						   dtype= torch.float32, 
						   device = pos.device) 

	# nested loop (this needs to be properly vectorized)
	for ii in range(Nsamples):
		for jj in range(Npoints):
			for kk in range(maxNumNeighs):
				if neighborList[ii,jj,kk]!= -1: 
					temp = pos[ii,neighborList[ii,jj,kk]] - pos[ii,jj]
					tempL = temp - L*torch.round(temp/L)
					Dist[ii, jj, kk] =  (torch.abs(tempL) - ave[0])/std[0]
					DistInv[ii, jj, kk] = (1/(torch.abs(tempL))- ave[1])/std[1]

	return (Dist, DistInv)