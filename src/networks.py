import torch
import torch.nn.functional as F
import numpy as np 
from utilities import genCoordinates 


class _ResidueModuleDense(torch.nn.Module):
    # implements the 

    def __init__(self, size_in, size_out, resNet=False, actfn = F.relu):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self._resNet = resNet
        self.actfn = actfn

        # print("value of Resnet")
        # print(self._resNet)

        if self._resNet : 
            self.weigth_skip = torch.nn.Parameter(torch.ones(1))
        else:
            self.weigth_skip = torch.zeros(1)

        self.linear = torch.nn.Linear(size_in, size_out)

    def forward(self, x):
        if self._resNet:
            if self.size_out == self.size_in :
                return self.weigth_skip*x + self.actfn(self.linear(x))

            # elif self.size_out ==  self.size_in/2:
            #     return  self.weigth_skip*0.5*torch.sum(x.view(x.size()[0],\
            #                                            -1,2), 2) + \
            #             self.actfn(self.linear(x))

            # elif self.size_out ==  2*self.size_in:
            #     return  self.weigth_skip*torch.cat([x,x], dim = 1) + \
            #              self.actfn(self.linear(x))

            else: 
                print("the dimensions don't match for resNet module")
                return self.actfn(self.linear(x))

        else:    
            return self.actfn(self.linear(x))


class _PyramidNetwork(torch.nn.Module):

    def __init__(self, sizes, resNet=False, actfn = F.relu):
        super().__init__()

        self.numLayers = len(sizes)-1
        self.layers = []

        for ii in range(self.numLayers):
            self.layers.append(_ResidueModuleDense(sizes[ii], sizes[ii+1], 
                                                   resNet = resNet, 
                                                   actfn = actfn))

        self.seqLayers = torch.nn.Sequential(*self.layers) 


    def forward(self, x):
        return self.seqLayers(x)


class DeepMDsimpleEnergy(torch.nn.Module):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               Npoints,
               length: torch.Tensor, # we need it to be a tensor
               maxNumNeighs = 4,
               descripDim = [1, 2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               resNet = False,
               av = torch.tensor([0.0, 0.0], dtype = torch.float32),
               std = torch.tensor([1.0, 1.0], dtype = torch.float32),
               **kwargs):
    super().__init__()

    # this should be done on the fly, for now we will keep it here
    self.Npoints = Npoints 
    self.length = length
    # maximum number of neighbors
    self.maxNumNeighs = maxNumNeighs
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]

    self.resNet = resNet
    # we may need to use the tanh here
    self.layerPyramid   = _PyramidNetwork(descripDim, 
                                          resNet = self.resNet,
                                          actfn = torch.tanh)
    self.layerPyramidInv  = _PyramidNetwork(descripDim, 
                                            resNet = self.resNet, 
                                            actfn = torch.tanh)
    
    # we may need to use the tanh especially here
    self.fittingNetwork = _PyramidNetwork(fittingDim, 
                                          resNet = resNet, 
                                          actfn = torch.tanh)

    self.linfitNet      = torch.nn.Linear(fittingDim[-1], 1)    


  def forward(self, inputs: torch.Tensor, neighList: torch.Tensor):
      # we watch the inputs 

    # in this case we are only considering the distances
    (dist, distInv) = genCoordinates(inputs, neighList, self.length,
                                     self.av, self.std)

    # ((Nsamples*Npoints*maxNumNeighs), \
    #  (Nsamples*Npoints*maxNumNeighs))

    L1   = self.layerPyramid(dist.view(-1,1))*dist.view(-1,1)
    # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
    L2   = self.layerPyramidInv(distInv.view(-1,1))*distInv.view(-1,1)
    # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
    LL = torch.cat((L1, L2), dim = 1)
    # (Nsamples*Npoints*maxNumNeighs, 2*descriptorDim)
    Dtemp = LL.view(-1, self.maxNumNeighs, 
                        2*self.descriptorDim)
    # (Nsamples*Npoints, maxNumNeighs, descriptorDim)
    D = torch.sum(Dtemp, dim = 1)
    # (Nsamples*Npoints, descriptorDim)

    F2 = self.fittingNetwork(D)
    F1 = self.linfitNet(F2)
    # print(F1.size())

    Energy = torch.sum(F1.view(-1, self.Npoints), dim = 1, keepdim= True)
    
    return Energy



class DeepMDsimpleEnergyForces(torch.nn.Module):
  """Combines the encoder and decoder into an end-to-end model for training."""

  def __init__(self,
               Npoints,
               length: torch.Tensor, # we need it to be a tensor
               maxNumNeighs = 4,
               descripDim = [1, 2, 4, 8, 16, 32],
               fittingDim = [16, 8, 4, 2, 1],
               resNet = False,
               av = torch.tensor([0.0, 0.0], dtype = torch.float32),
               std = torch.tensor([1.0, 1.0], dtype = torch.float32),
               **kwargs):
    super().__init__()

    # this should be done on the fly, for now we will keep it here
    self.Npoints = Npoints 
    self.length = length
    # maximum number of neighbors
    self.maxNumNeighs = maxNumNeighs
    # we normalize the inputs (should help for the training)
    self.av = av
    self.std = std
    self.descripDim = descripDim
    self.fittingDim = fittingDim
    self.descriptorDim = descripDim[-1]

    self.resNet = resNet
    # we may need to use the tanh here
    self.layerPyramid   = _PyramidNetwork(descripDim, 
                                          resNet = self.resNet,
                                          actfn = torch.tanh)
    self.layerPyramidInv  = _PyramidNetwork(descripDim, 
                                            resNet = self.resNet, 
                                            actfn = torch.tanh)
    
    # we may need to use the tanh especially here
    self.fittingNetwork = _PyramidNetwork(fittingDim, 
                                          resNet = resNet, 
                                          actfn = torch.tanh)

    self.linfitNet      = torch.nn.Linear(fittingDim[-1], 1)    


  def forward(self, inputs: torch.Tensor, neighList: torch.Tensor):
      # we watch the inputs 

    inputs.requires_grad = True
    # in this case we are only considering the distances
    (dist, distInv) = genCoordinates(inputs, neighList, self.length,
                                     self.av, self.std)

    # ((Nsamples*Npoints*maxNumNeighs), \
    #  (Nsamples*Npoints*maxNumNeighs))

    L1   = self.layerPyramid(dist.view(-1,1))*dist.view(-1,1)
    # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
    L2   = self.layerPyramidInv(distInv.view(-1,1))*distInv.view(-1,1)
    # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
    LL = torch.cat((L1, L2), dim = 1)
    # (Nsamples*Npoints*maxNumNeighs, 2*descriptorDim)
    Dtemp = LL.view(-1, self.maxNumNeighs, 
                        2*self.descriptorDim)
    # (Nsamples*Npoints, maxNumNeighs, descriptorDim)
    D = torch.sum(Dtemp, dim = 1)
    # (Nsamples*Npoints, descriptorDim)

    F2 = self.fittingNetwork(D)
    F1 = self.linfitNet(F2)
    # print(F1.size())

    Energy = torch.sum(F1.view(-1, self.Npoints), dim = 1, keepdim= True)
    
    (Forces,) = torch.autograd.grad(-torch.sum(Energy), inputs, create_graph=True, allow_unused=True) 

    return Energy, Forces
 

#  ##


# pointsArrayTorchSmall = pointsArrayTorch[:16, :]

# pointsnumpy = pointsArrayTorchSmall.numpy()

# neighbor_list = computInterListOpt(pointsnumpy, Lcell*Ncells,  
#                                             radious, maxNumNeighs)

# neighbor_list = torch.tensor(neighbor_list)

# inputs = pointsArrayTorchSmall.to(device)
# neighList = neighbor_list.to(device)

# inputs.requires_grad = True

# energyNN = model(inputs, neighList)

#     # in this case we are only considering the distances
#     (dist, distInv) = genCoordinates(inputs, neighList, model.length,
#                                      model.av, model.std)

#     # ((Nsamples*Npoints*maxNumNeighs), \
#     #  (Nsamples*Npoints*maxNumNeighs))

#     L1   = model.layerPyramid(dist.view(-1,1))*dist.view(-1,1)
#     # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
#     L2   = model.layerPyramidInv(distInv.view(-1,1))*distInv.view(-1,1)
#     # (Nsamples*Npoints*maxNumNeighs, descriptorDim)
#     LL = torch.cat((L1, L2), dim = 1)
#     # (Nsamples*Npoints*maxNumNeighs, 2*descriptorDim)
#     Dtemp = LL.view(-1, model.maxNumNeighs, 
#                         2*model.descriptorDim)
#     # (Nsamples*Npoints, maxNumNeighs, descriptorDim)
#     D = torch.sum(Dtemp, dim = 1)
#     # (Nsamples*Npoints, descriptorDim)

#     F2 = model.fittingNetwork(D)
#     F1 = model.linfitNet(F2)
#     # print(F1.size())

#     Energy = torch.sum(F1.view(-1, model.Npoints), dim = 1, keepdim= True)
    
#     Forces = torch.autograd.grad(Energy[0], inputs)

#     return Energy
