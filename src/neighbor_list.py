# functions to compute the interaction lists using a 
# periodic distance. The functions are optimized to 
# be used with numba

### Question: can we rewrite this in pytorch? 
# otherwise just keep it in python-numba

import numpy as np 
from numba import jit 

@jit(nopython=True)
def computInterListOpt(Rinnumpy, L,  radious, maxNumNeighs):
  Nsamples, Npoints = Rinnumpy.shape

  # computing the distance
  DistNumpy = np.abs(Rinnumpy.reshape(Nsamples,Npoints,1) \
              - Rinnumpy.reshape(Nsamples,1,Npoints))

  # periodicing the distance
  out = np.zeros_like(DistNumpy)
  np.round(DistNumpy/L, 0, out)
  DistNumpy = DistNumpy - L*out

  # add the padding and loop over the indices 
  Idx = np.zeros((Nsamples, Npoints, maxNumNeighs), dtype=np.int32) -1 
  
  for ii in range(0,Nsamples):
    for jj in range(0, Npoints):
      ll = 0 
      for kk in range(0, Npoints):
        if jj!= kk and np.abs(DistNumpy[ii,jj,kk]) < radious:
          # checking that we are not going over the max number of
          # neighboors, if so we break the loop
          if ll >= maxNumNeighs:
            print("Number of neighboors is larger than the max number allowed")
            break
          Idx[ii,jj,ll] = kk
          ll += 1 

  return Idx


# optimized function fo compute the distance
# TODO: use scikit-learn KD tree to optimize this one 
@jit(nopython=True)
def computInterList2DOpt(Rinnumpy, L,  radious, maxNumNeighs):
  # function to compute the interaction lists 
  # this function in agnostic to the dimension of the data.
  Nsamples, Npoints, dimension = Rinnumpy.shape

  # computing the relative coordinates
  DistNumpy = Rinnumpy.reshape(Nsamples,Npoints,1, dimension) \
              - Rinnumpy.reshape(Nsamples,1, Npoints,dimension)

  # periodicing the distance
  # working around some quirks of numba with the np.round function
  out = np.zeros_like(DistNumpy)
  np.round(DistNumpy/L, 0, out)
  DistNumpy = DistNumpy - L*out

  # computing the distance
  DistNumpy = np.sqrt(np.sum(np.square(DistNumpy), axis = -1))

  # add the padding and loop over the indices 
  Idx = np.zeros((Nsamples, Npoints, maxNumNeighs), dtype=np.int32) -1 
  for ii in range(0,Nsamples):
    for jj in range(0, Npoints):
      ll = 0 
      for kk in range(0, Npoints):
        if jj!= kk and np.abs(DistNumpy[ii,jj,kk]) < radious:
          # checking that we are not going over the max number of
          # neighboors, if so we break the loop
          if ll >= maxNumNeighs:
            print("Number of neighboors is larger than the max number allowed")
            break
          Idx[ii,jj,ll] = kk
          ll += 1 

  return Idx