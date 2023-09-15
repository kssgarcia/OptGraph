# %% Initialization
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation

import solidspy.assemutil as ass # Solidspy 1.1.0
import solidspy.postprocesor as pos 
import meshio

from beams import *
# from utils.SIMP_utils_wrench import *
from utils.SIMP_utils_wrench2 import *

np.seterr(divide='ignore', invalid='ignore')

mesh = meshio.read("wrench2.msh")
points = mesh.points
cells = mesh.cells

nodes = np.zeros((points.shape[0], 5))
nodes[:,0] = np.arange(0,points.shape[0])
nodes[:,1:3] = points[:,:2]
nodes[np.unique(cells[2].data.flatten()), -2:] = -1
nodes[np.unique(cells[3].data.flatten()), -2:] = -1

n_loads = np.unique(np.concatenate((cells[0].data.flatten(), cells[1].data.flatten()))).shape[0]
loads = np.zeros((n_loads,3))
loads[:,0] = np.unique(np.concatenate((cells[0].data.flatten(), cells[1].data.flatten())))
loads[:,-1] = 1/n_loads

els = np.zeros((cells[-1].data.shape[0], 7), dtype=int)
els[:,0] = np.arange(0,cells[-1].data.shape[0], dtype=int)
els[:, 1:3] = [1,0]
els[:,-4:] = cells[-1].data

mats = np.zeros((els.shape[0], 3))
mats[:] = [1,0.28,1]

BC = np.argwhere(nodes[:,-1]==-1)

IBC, UG, _ = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

# Calculate centers and volumes
nels = els.shape[0]
niter = 60
centers, areas = center_els(nodes, els)
mats[:,-1] = areas

# Initialize the design variables
r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4
penal = 3
Emin=1e-9
Emax=1.0
volfrac = 0.5
change = 10
g = 0

# Initialize the density, sensitivity and the iteration history
rho = volfrac * np.ones(nels,dtype=float)
sensi_rho = np.ones(nels)
rho_old = rho.copy()
d_c = np.ones(nels)
d_v = areas.copy()
rho_data = []

# %%

for i in range(1):

    if change < 0.01:
        print('Convergence reached')

    # Change density 
    mats[:,2] = Emin+rho**penal*(Emax-Emin)

    IBC, UG, _ = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
    UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

    # Sensitivity analysis
    params = tuple(mats[0, :])
    elcoor = nodes[els[:, -4:], 1:3]
    
    elast_quad4_fixed = partial(uel.elast_quad4, params=params)
    kloc = np.array(list(map(elast_quad4_fixed, elcoor)))[:,0]

    sensi_rho = np.zeros((nels,))
    for i in range(nels):
        sensi_rho[i] = (np.dot(UC[els[:,-4:]].reshape(nels,8)[i], kloc[i]) * UC[els[:,-4:]].reshape(nels,8)[i]).sum(0)

    d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
    d_c[:] = density_filter(centers, r_min, rho, d_c)

    # Optimality criteria
    rho_old[:] = rho
    rho[:], g = optimality_criteria(nels, rho, d_c, g)

    # Compute the change
    change = np.linalg.norm(rho.reshape(nels,1)-rho_old.reshape(nels,1),np.inf)


# %% 

mask = rho > 0.5
print(mask)
mask_els = protect_els(els[np.invert(mask)], els.shape[0], loads, BC)
mask = np.bitwise_or(mask, mask_els)
del_node(nodes, els[mask], loads, BC)
els = els[mask]
print(els.shape)

E_nodes, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)
pos.fields_plot(els, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)

# %%
colors = sensi_rho/sensi_rho.max()
plot_mesh_with_colors(els, nodes, colors)

# %%

colors = rho.copy()/rho.max()

plt.figure()
plt.scatter(centers[:,0], centers[:,1], c=colors)
plt.show()