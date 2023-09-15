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
from utils.ESO_utils import *

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
UCI, E_nodesI, S_nodesI = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

# %%
niter = 10
RR = 0.001 # Initial removal ratio
ER = 0.001 # Removal ratio increment
ELS = None

for i in range(1):

    if not is_equilibrium(nodes, mats, els, loads) : break # Check equilibrium/volume and stop if not

    # FEM analysis
    IBC, UG, _ = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
    UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses

    # Compute Sensitivity number
    sensi_number = sensi_el(nodes, mats, els, UC) # Sensitivity number
    mask_del = sensi_number < RR # Mask of elements to be removed
    mask_els = protect_els(els, loads, BC) # Mask of elements to do not remove
    mask_del *= mask_els # Mask of elements to be removed and not protected
    ELS = els # Save last iteration elements
    
    # Remove/add elements
    els = np.delete(els, mask_del, 0) # Remove elements
    del_node(nodes, els) # Remove nodes

    RR += ER


# %% 

E_nodes, S_nodes = pos.strain_nodes(nodes, ELS, mats[:,:2], UC)
pos.fields_plot(ELS, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)

# %%
print(sensi_number.shape)
print(UC.shape)

plot_mesh(els, nodes, UC)
