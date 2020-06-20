import numpy as np 
import matplotlib.pyplot as plt
import torch
 
import sys
sys.path.insert(1, '../src')

from data_gen_1d import genDataYukawaPer


from neighbor_list import computInterListOpt
from networks import _ResidueModuleDense

from data_gen_2d import potentialPer
from data_gen_2d import forces

from utilities import computInterList2Dv2
from utilities import genDistInvPerNlist2D

from utilities import computInterList2D
from utilities import genDistInvPerNlist2DSimple

from utilities import genDistInvPerNlist2Dwhere
 
from utilities import genDistInvPerNlist2Dwherev2

Ncells = 10
Np = 2 
mu = 10 
Nsamples = 2 
minDelta = 0.1
Lcell = 1.0 
L = Nce
Npoints = Ncells*Np

radious = 1.5
maxNumNeighs = 8

points, pot, forces = genDataYukawaPer(Ncells, Np, mu, Nsamples, minDelta, Lcell)

neighList = computInterListOpt(points, L,  radious, maxNumNeighs)



positions = torch.tensor(points, dtype = torch.float32, requires_grad=True)

neighborList = torch.tensor(neighList, dtype = torch.int32)
# Nsamples, Npoint, MaxNumNeighs

# we build the Distance tensor


Dist = torch.tensor(np.zeros((Nsamples, Npoints, maxNumNeighs)), 
					dtype= torch.float32) 
DistInv = torch.tensor(np.zeros((Nsamples, Npoints, maxNumNeighs)), dtype= torch.float32)

mean = torch.tensor([0.0, 0.0], dtype=torch.float32)
std = torch.tensor([1.0, 1.0], dtype=torch.float32)

for ii in range(Nsamples):
	for jj in range(Npoints):
		for kk in range(maxNumNeighs):
			if neighborList[ii,jj,kk]!= -1: 
				temp = positions[ii,neighborList[ii,jj,kk]] - positions[ii,jj]
				tempL = temp - L*torch.round(temp/L)
				Dist[ii, jj, kk] =  (torch.abs(tempL) - mean[0])/std[0]
				DistInv[ii, jj, kk] = (1/(torch.abs(tempL))- mean[1])/std[1]


Dist2 = Dist.view(-1, 1)

Pyramid = _PyramidNetwork([1, 2, 4, 8, 16], resNet = True) 

S = Pyramid(Dist2)

Omega = S*Dist2

Omega2 = Omega.view(Nsamples, Npoints, maxNumNeighs, 16)

D = torch.sum(Omega2, dim = 2)

F1 = torch.nn.Linear(16, 1, bias = True)

Elocal = F1(D)

Etotal = torch.sum(torch.squeeze(Elocal),dim = 1)

grads = torch.autograd.grad([Etotal[0], Etotal[1]], positions)[0]   



(dd1, ddinv) = genCoordinates(positions, neighborList)  


model = DeepMDsimpleEnergy(Npoints, maxNumNeighs, [2, 4, 8, 16, 32], [16, 8, 4, 2, 1], resNet = True)

E = model(positions, neighborList)