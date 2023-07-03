# %% Initialization
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation

import solidspy.assemutil as ass # Solidspy 1.1.0
import solidspy.postprocesor as pos 
import solidspy.preprocesor as msh
import meshio

from beams import *
from SIMP_utils_copy import *


mesh = meshio.read("wrench.msh")
points = mesh.points
cells = mesh.cells

nodes = np.zeros((points.shape[0], 5))
nodes[:,0] = np.arange(0,points.shape[0])
nodes[:,1:3] = points[:,:2]
nodes[cells[1].data[:,0], -2:] = -1
nodes[cells[2].data[:,0], -2:] = -1

loads = np.zeros((1,3))
loads[0,0] = cells[0].data[:, 0][7]
loads[0,-1] = 1000

els = np.zeros((cells[-1].data.shape[0], 7), dtype=int)
els[:,0] = np.arange(0,cells[-1].data.shape[0], dtype=int)
els[:, 1:3] = [1,0]
els[:,-4:] = cells[-1].data

mats = np.zeros((els.shape[0], 3))
mats[:] = [2.068e11,0.28,1]

BC = np.argwhere(nodes[:,-1]==-1)


np.seterr(divide='ignore', invalid='ignore')

# Calculate centers and volumes
nels = els.shape[0]
niter = 60
centers = center_els(nodes, els)

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
d_v = np.ones(nels)
rho_data = []


for i in range(niter):

    if change < 0.01:
        print('Convergence reached')

    if i == 10:
        break

    # Change density 
    mats[:,2] = Emin+rho**penal*(Emax-Emin)

    IBC, UG, rhs_vec = preprocessing(nodes, mats, els, loads)
    UC, *_ = postprocessing(nodes, mats[:,:2], els, IBC, UG, strain_sol=False)

    # Sensitivity analysis
    sensi_rho[:] = sensitivity_els(nodes, mats, els, UC, nels)
    d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
    obj = ((Emin+rho**penal*(Emax-Emin))*sensi_rho).sum()
    d_c[:] = density_filter(centers, r_min, rho, d_c)
    d_v[:] = np.ones(nels)

    # Optimality criteria
    rho_old[:] = rho
    rho[:], g = optimality_criteria(nels, rho, d_c, d_v, g)

    # Compute the change
    change = np.linalg.norm(rho.reshape(nels,1)-rho_old.reshape(nels,1),np.inf)

    if i%5 == 0:
        rho_data.append(-rho)

# Remove/add elements
# %%
mask = rho > 0.5
mask_els = protect_els(els[np.invert(mask)], els.shape[0], loads, BC)
mask = np.bitwise_or(mask, mask_els)
del_node(nodes, els[mask], loads, BC)
els = els[mask]

# %% 
E_nodes, S_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)
pos.fields_plot(els, nodes, UC, E_nodes=E_nodes, S_nodes=S_nodes)

# %% Animation
fig, ax = plt.subplots()
im = ax.imshow(np.zeros((ny,nx)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))

def update(frame):
    rho_frame = rho_data[frame]
    im.set_array(rho_frame)
    return im,
ani = animation.FuncAnimation(fig, update, frames=len(rho_data), interval=200, blit=True)
output_file = "animation.gif"
ani.save(output_file, writer="pillow")

# %% Plot
plt.ion() 
fig,ax = plt.subplots()
im = ax.imshow(-rho.reshape(ny, nx), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
fig.show()
