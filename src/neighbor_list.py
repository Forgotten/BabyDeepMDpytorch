# functions to compute the interaction lists using a 
# periodic distance. The functions are optimized to 
# be used with numba

### Question: can we rewrite this in pytorch? 
# otherwise just keep it in python-numba

import numpy as np 
from numba import jit 

@jit(nopython=True)
def computInterListOpt(Rinnumpy, L, radious, maxNumNeighs):
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


# optimized function fo compute the distance
# TODO: use scikit-learn KD tree to optimize this one 
@jit(nopython=True)
def comput_inter_list(r_in_numpy, L, radious, max_num_neighs):
  # TODO: add documentation
  # function to compute the interaction lists 
  # this function in agnostic to the dimension of the data.
  n_samples, n_points, dimension = r_in_numpy.shape

  # computing the relative coordinates
  dist_numpy = r_in_numpy.reshape(n_samples, n_points, 1, dimension) \
              - r_in_numpy.reshape(n_samples, 1, n_points, dimension)

  # periodicing the distance
  # working around some quirks of numba with the np.round function
  out = np.zeros_like(dist_numpy)
  np.round(dist_numpy/L, 0, out)
  dist_numpy = dist_numpy - L*out

  # computing the distance
  dist_numpy = np.sqrt(np.sum(np.square(dist_numpy), axis = -1))

  # add the padding and loop over the indices 
  Idx = np.zeros((n_samples, n_points, max_num_neighs), dtype=np.int32)-1 

  for ii in range(0, n_samples):
    for jj in range(0, n_points):
      ll = 0 
      for kk in range(0, n_points):
        if jj!= kk and np.abs(dist_numpy[ii,jj,kk]) < radious:
          # checking that we are not going over the max number of
          # neighboors, if so we break the loop
          if ll >= max_num_neighs:
            print("Number of neighboors is larger than the max number allowed")
            break
          Idx[ii,jj,ll] = kk
          ll += 1 

  return Idx



# optimized function fo compute the distance
# TODO: use scikit-learn KD tree to optimize this one 
@jit(nopython=True)
def comput_inter_list_type(r_in_numpy, atom_type, 
                           L, radious, max_num_neighs_type):
  # TODO: add documentation
  # r_in_numpy (n_samples, n_points, dims)
  # atom_type  (n_samples, n_points )
  # function to compute the interaction lists 
  # atom type is an array with integer entries
  # this function in agnostic to the dimension of the data.
  n_samples, n_points, dimension = r_in_numpy.shape

  # computing the relative coordinates
  dist_numpy = r_in_numpy.reshape(n_samples, n_points, 1, dimension) \
              - r_in_numpy.reshape(n_samples, 1, n_points, dimension)

  # periodicing the distance
  # working around some quirks of numba with the np.round function
  out = np.zeros_like(dist_numpy)
  np.round(dist_numpy/L, 0, out)
  dist_numpy = dist_numpy - L*out

  # computing the distance
  dist_numpy = np.sqrt(np.sum(np.square(dist_numpy), axis = -1))

  # add the padding and loop over the indices 
  Idx = np.zeros((n_samples, n_points, np.sum(max_num_neighs_type)), dtype=np.int64)-1 

  indx = np.zeros(len(max_num_neighs_type)+1, dtype = np.int64)
  indx[1:] = np.cumsum(max_num_neighs_type).astype(np.int64)
  

  for ii in range(0, n_samples):
    for jj in range(0, n_points):
      ll = np.zeros(len(max_num_neighs_type), dtype = np.int64)
      for kk in range(0, n_points):
        if jj!= kk and np.abs(dist_numpy[ii,jj,kk]) < radious:
          # checking that we are not going over the max number of
          # neighboors, if so we break the loop
          if ll[atom_type[ii,kk]] >= max_num_neighs_type[atom_type[ii,kk]]:
            print("Number of neighboors is larger than the max number allowed")
            continue
          
          Idx[ii,jj,ll[atom_type[ii,kk]]+indx[atom_type[ii,kk]]] = kk
          ll[atom_type[ii,kk]] += 1 

  return Idx