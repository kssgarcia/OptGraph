import numpy as np
import solidspy.assemutil as ass    
import solidspy.postprocesor as pos 
import solidspy.solutil as sol      
import solidspy.uelutil as uel 
from scipy.spatial.distance import cdist

from functools import partial

import matplotlib.pyplot as plt


def preprocessing(nodes, mats, els, loads):
    """
    Compute IBC matrix and the static solve.
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
        
    Returns
    -------
    bc_array : ndarray 
        Boundary conditions array
    disp : ndarray 
        Static displacement solve
    rh_vec : ndarray 
        Vector of loads
    """   

    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)
    print("Number of elements: {}".format(els.shape[0]))

    # System assembly
    stiff_mat, _ = ass.assembler(els, mats, nodes[:, :3], neq, assem_op)
    rhs_vec = ass.loadasem(loads, bc_array, neq)

    # System solution
    disp = sol.static_sol(stiff_mat, rhs_vec)
    if not np.allclose(stiff_mat.dot(disp)/stiff_mat.max(),
                       rhs_vec/stiff_mat.max()):
        print("The system is not in equilibrium!")
    return bc_array, disp, rhs_vec


def postprocessing(nodes, mats, els, bc_array, disp):
    """
    Compute the nodes displacements, strains and stresses.
    
    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    bc_array : ndarray 
        Boundary conditions array
    disp : ndarray 
        Static solve
        
    Returns
    -------
    disp_complete : ndarray 
        Displacements at elements.
    strain_nodes : ndarray 
        Strains at elements
    stress_nodes : ndarray 
        Stresses at elements
    """   
    
    disp_complete = pos.complete_disp(bc_array, nodes, disp)
    strain_nodes, stress_nodes = None, None
    strain_nodes, stress_nodes = pos.strain_nodes(nodes, els, mats, disp_complete)
    
    return disp_complete, strain_nodes, stress_nodes

def optimality_criteria(nels, rho, d_c, g):
    """
    Optimality criteria method.

    Parameters
    ----------
    nels : int
        Number of elements.
    rho : ndarray
        Array with the density of each element.
    d_c : ndarray
        Array with the derivative of the compliance.
    g : float
        Volume constraint.

    Returns
    -------
    rho_new : ndarray
        Array with the new density of each element.
    gt : float
        Volume constraint.

    """
    l1=0
    l2=1e9
    move=0.2
    rho_new=np.zeros(nels)
    while (l2-l1)/(l1+l2)>1e-3: 
        lmid=0.5*(l2+l1)
        rho_new[:]= np.maximum(0.0,np.maximum(rho-move,np.minimum(1.0,np.minimum(rho+move,rho*np.sqrt(-d_c/lmid)))))
        gt=g+np.sum(((rho_new-rho)))
        if gt>0 :
            l1=lmid
        else:
            l2=lmid
    return (rho_new,gt)


def sensitivity_els(nodes, mats, els, UC, nels):
    """
    Calculate the sensitivity number for each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    mats : ndarray
        Array with models materials
    els : ndarray
        Array with models elements
    UC : ndarray
        Displacements at nodes
    nx : float
        Number of elements in x direction.
    ny : float
        Number of elements in y direction.

    Returns
    -------
    sensi_number : ndarray
        Sensitivity number for each element.
    """   
    sensi_number = np.zeros(els.shape[0])
    params = mats[1, :]
    elcoor = nodes[els[:, -4:], 1:3]
    
    elast_quad4_fixed = partial(uel.elast_quad4, params=params)
    kloc = np.array(list(map(elast_quad4_fixed, elcoor)))[:,0]
    print(kloc.shape)

    sensi_number = (np.dot(UC[els[:,-4:]].reshape(nels,8),kloc) * UC[els[:,-4:]].reshape(nels,8) ).sum(1)

    return sensi_number

def volume(els, length, height, nx, ny):
    """
    Volume calculation.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements.
    length : ndarray
        Length of the beam.
    height : ndarray
        Height of the beam.
    nx : float
        Number of elements in x direction.
    ny : float
        Number of elements in y direction.

    Return 
    ----------
    V: float

    """

    dy = length / nx
    dx = height / ny
    V = dx * dy * np.ones(els.shape[0])

    return V

def density_filter(centers, r_min, rho, d_rho):
    """
    Performe the sensitivity filter.
    
    Parameters
    ----------
    centers : ndarray
        Array with the centers of each element.
    r_min : float
        Minimum radius of the filter.
    rho : ndarray
        Array with the density of each element.
    d_rho : ndarray
        Array with the derivative of the density of each element.
        
    Returns
    -------
    densi_els : ndarray
        Sensitivity of each element with filter
    """
    dist = cdist(centers, centers, 'euclidean')
    delta = r_min - dist
    H = np.maximum(0.0, delta)
    densi_els = (rho*H*d_rho).sum(1)/(H.sum(1)*np.maximum(0.001,rho))

    return densi_els

def center_els(nodes, els):
    """
    Calculate the center of each element.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes.
    els : ndarray
        Array with models elements.
        
    Returns
    -------
    centers : ndarray
        Centers of each element.
    areas : ndarray
        Areas of each element.
    """
    centers = np.zeros((els.shape[0], 2))
    areas = np.zeros(els.shape[0])
    for i, el in enumerate(els):
        n = nodes[el[-4:], 1:3]
        x1, y1 = n[0]
        x2, y2 = n[1]
        x3, y3 = n[2]
        x4, y4 = n[3]
        cx = (x1 + x2 + x3 + x4) / 4
        cy = (y1 + y2 + y3 + y4) / 4
        centers[i] = [cx, cy]
        a = 0.5 * abs((x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1))
        areas[i] = a
    return centers, areas

def plot_mesh(elements, nodes, disp, E_nodes=None):
    """
    Plot contours for model

    Get from: https://github.com/AppliedMechanics-EAFIT/SolidsPy/blob/master/solidspy/solids_GUI.py

    Parameters
    ----------
    nodes : ndarray (float)
        Array with number and nodes coordinates:
         `number coordX coordY BCX BCY`
    elements : ndarray (int)
        Array with the node number for the nodes that correspond
        to each element.
    disp : ndarray (float)
        Array with the displacements.
    E_nodes : ndarray (float)
        Array with strain field in the nodes.

    """
    # Check for structural elements in the mesh
    struct_pos = 5 in elements[:, 1] or \
             6 in elements[:, 1] or \
             7 in elements[:, 1]
    if struct_pos:
        # Still not implemented visualization for structural elements
        print(disp)
    else:
        pos.plot_node_field(disp, nodes, elements, title=[r"$u_x$", r"$u_y$"],
                        figtitle=["Horizontal displacement",
                                  "Vertical displacement"])
        if E_nodes is not None:
            pos.plot_node_field(E_nodes, nodes, elements,
                            title=[r"",
                                   r"",
                                   r"",],
                            figtitle=["Strain epsilon-xx",
                                      "Strain epsilon-yy",
                                      "Strain gamma-xy"])

def protect_els(els, nels, loads, BC):
    """
    Compute an mask array with the elements that don't must be deleted.
    
    Parameters
    ----------
    els : ndarray
        Array with models elements
    nels : ndarray
        Number of elements
    loads : ndarray
    BC : ndarray 
        Boundary conditions nodes
        
    Returns
    -------
    mask_els : ndarray 
        Array with the elements that don't must be deleted.
    """   
    mask_els = np.zeros(nels, dtype=bool)
    protect_nodes = np.hstack((loads[:,0], BC[:,0])).astype(int)
    protect_index = None
    for p in protect_nodes:
        protect_index = np.argwhere(els[:, -4:] == p)[:,0]
        mask_els[els[protect_index,0]] = True
        
    return mask_els

def del_node(nodes, els, loads, BC):
    """
    Retricts nodes dof that aren't been used and free up the nodes that are in use.
    
    Parameters
    ----------
    nodes : ndarray
        Array with models nodes
    els : ndarray
        Array with models elements
    loads : ndarray
        Array with models loads
    BC : ndarray 
        Boundary conditions nodes

    Returns
    -------
    """   
    protect_nodes = np.hstack((loads[:,0], BC[:,0])).astype(int)
    for n in nodes[:,0]:
        if n not in els[:, -4:]:
            nodes[int(n), -2:] = -1
        elif n not in protect_nodes and n in els[:, -4:]:
            nodes[int(n), -2:] = 0


def plot_mesh_with_colors(nodes, els, colors):
    """
    Plot the mesh defined by the given nodes and elements, with the colors of each element defined by the given colors.

    Parameters
    ----------
    nodes : ndarray
        Array of shape (n_nodes, 5) containing the node information. The first column contains the index of each node, the
        second and third columns contain the x and y coordinates, respectively, and the fourth and fifth columns indicate
        whether each node is constrained in the x or y direction.
    els : ndarray
        Array of shape (n_els, 7) containing the element information. The first column contains the index of each element,
        and the next four columns contain the indices of the four nodes that make up each element.
    colors : ndarray
        Array of shape (n_els, 1) containing the color of each element, as a value between 0 and 1.

    Returns
    -------
    None.
    """
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Loop over the elements and plot each one with its corresponding color
    for i in range(els.shape[0]):
        # Get the indices of the nodes that make up the current element
        node_indices = els[i, -4:]

        # Get the x and y coordinates of the nodes
        x = nodes[node_indices, 1]
        y = nodes[node_indices, 2]

        # Get the color of the current element
        color = colors[i]

        # Plot the current element with its corresponding color
        ax.fill(x, y, color=color, edgecolor='k')

    # Set the axis limits
    ax.set_xlim([np.min(nodes[:, 1]), np.max(nodes[:, 1])])
    ax.set_ylim([np.min(nodes[:, 2]), np.max(nodes[:, 2])])

    # Set the axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Show the plot
    plt.show()