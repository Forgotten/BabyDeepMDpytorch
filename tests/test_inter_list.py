import numpy as np 
import matplotlib.pyplot as plt
import torch

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../src')
from neighbor_list import comput_inter_list_type
from utilities_3d import gen_coordinates
from utilities_3d import smooth_cut_off


n_parts = 3

r_in = np.linspace(0., 1., n_parts+1)[:-1]
x_in, y_in, z_in = np.meshgrid(r_in, r_in, r_in)

r_in = np.concatenate([x_in.reshape((-1,1)), 
                       y_in.reshape((-1,1)),
                       z_in.reshape((-1,1))], axis = -1)

r_in = r_in.reshape((1, *r_in.shape))

atom_type = []
for k in range(n_parts):
    atom_type += [k]*n_parts**2
atom_type = np.array(atom_type, dtype = np.int64)

L = 1. 
radious = 0.45

max_num_neighs_type = np.array([5, 5, 5], dtype = np.int64)

inter_list = comput_inter_list_type(r_in, atom_type, L, radious,max_num_neighs_type)

# plotting in 3D to check that the computation is correct

idx_center = 13

xs = []
ys = []
zs = []

for j in range(inter_list.shape[-1]):
    if inter_list[0, idx_center, j] != -1:
        xs.append(r_in[0, inter_list[0, idx_center, j], 0])
        ys.append(r_in[0, inter_list[0, idx_center, j], 1])
        zs.append(r_in[0, inter_list[0, idx_center, j], 2])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(r_in[0,:,0], r_in[0,:,1], r_in[0,:,2], marker='.',  s=5)

ax.scatter([r_in[0, idx_center, 0]],
           [r_in[0, idx_center, 1]], 
           [r_in[0, idx_center, 2]], marker='o',  s=20)

ax.scatter(xs, ys, zs, marker='^',  s=20)

plt.show()

# testing the smooth cut-off
r_cs = torch.tensor(1.5) # smooth cut-off
r_c = torch.tensor(2.5)  # cut-off

r_test = torch.tensor(np.linspace(0,3, 100), dtype=torch.float32)
ans = smooth_cut_off(r_test, r_cs, r_c)
plt.plot(r_test.numpy(), ans.numpy())
plt.show()

# prepare for the gen_coordinates

r_in_torch = torch.tensor(r_in, dtype = torch.float32)
inter_list_torch = torch.tensor(inter_list)
L_torch = torch.tensor(L)

r_in_torch.requires_grad = True



# comput the generalized coordinated from the interaction list
s_ij, r_ij = gen_coordinates(r_in_torch, inter_list_torch,
                            L_torch)

ds_dr = torch.autograd.grad(-torch.sum(s_ij[0][0]), 
                                     r_in_torch, create_graph=True, 
                                     allow_unused=True)


# apply the smooth cut-off
s_ij = smooth_cut_off(s_ij, torch.tensor(radious/2), torch.tensor(radious) )
r_ij = r_ij*s_ij.unsqueeze(-1)

