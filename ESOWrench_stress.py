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

niter = 100
RR = 0.001 # Initial removal ratio
ER = 0.002 # Removal ratio increment
ELS = None

# %%

for i in range(niter):

    if not is_equilibrium(nodes, mats, els, loads) : break # Check equilibrium/volume and stop if not

    # FEM analysis
    IBC, UG, _ = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
    UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses
    E_els, S_els = strain_els(els, E_nodes, S_nodes) # Calculate strains and stresses in elements
    vons = np.sqrt(S_els[:,0]**2 - (S_els[:,0]*S_els[:,1]) + S_els[:,1]**2 + 3*S_els[:,2]**2)

    # Remove/add elements
    RR_el = vons/vons.max() # Relative stress
    mask_del = RR_el < RR # Mask for elements to be deleted
    mask_els = protect_els(els, loads, BC) # Mask for elements to be protected
    mask_del *= mask_els  
    els = np.delete(els, mask_del, 0) # Delete elements
    del_node(nodes, els) # Delete nodes that are not connected to any element

    RR += ER


# %% 

E_nodes, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)
pos.fields_plot(els, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)

# %%

# FEM analysis
IBC, UG, _ = preprocessing(nodes, mats, els, loads) # Calculate boundary conditions and global stiffness matrix
UC, E_nodes, S_nodes = postprocessing(nodes, mats[:,:2], els, IBC, UG) # Calculate displacements, strains and stresses
E_els, S_els = strain_els(els, E_nodes, S_nodes) # Calculate strains and stresses in elements
vons = np.sqrt(S_els[:,0]**2 - (S_els[:,0]*S_els[:,1]) + S_els[:,1]**2 + 3*S_els[:,2]**2)


centers, areas = center_els(nodes, els)

plt.figure()
plt.scatter(centers[:,0], centers[:,1], c=vons/vons.max())
plt.show()