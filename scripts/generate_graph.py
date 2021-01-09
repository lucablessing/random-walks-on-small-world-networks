import numpy as np
import networkx as nx
from distances import dist_2d_lattice


def Kleinberg_2d(L, r, seed=42):
    '''
    constructor of a Kleinber graph model

    Arguments:
    - L: number of points on line
    - r: parameter for the long range decay
    - seed: inital number used to generate the random edges

    Return:
    - G: Kleinber graph
    '''
    np.random.seed(seed)
    G = nx.grid_2d_graph(L, L, periodic=True)
    Z = 0
    # computation of Z from (0,0) node
    for y in G:
        if y != (0, 0):
            Z = Z + dist_2d_lattice((0, 0), y, L)**(-r)
    # creation of new long range random edges
    for x in G:
        for y in G:
            if x > y:  # to avoid self loop and also double count on the edges
                if np.random.uniform(0, 1) < (dist_2d_lattice(x, y, L)**(-r)/Z):
                    G.add_edge(x, y)
    return G


def small_world_2d(L, r, seed=42):
    '''
    constructor of a Small world graph model

    Arguments:
    - L: number of points on line
    - r: parameter for the long range decay
    - seed: inital number used to generate the random edges

    Return:
    - G: small world graph
    '''
    np.random.seed(seed)
    G = nx.grid_2d_graph(L, L, periodic=True)
    Z = 0
    # computation of Z from (0,0) node
    for y in G:
        if y != (0, 0):
            d = dist_2d_lattice((0, 0), y, L)
            # Z is computed ONLY over the long range distance
            if d > 1:
                Z = Z+d**(-r)

    # creation of new long range random edges
    for x in G:
        for y in G:
            if x > y:  # to avoid self loop and also double count on the edges
                if np.random.uniform(0, 1) < (dist_2d_lattice(x, y, L)**(-r)/Z):
                    G.add_edge(x, y)
    return G


def small_world_2d_new(L, r, seed=None):
    '''
    constructor of a Small world graph model

    Arguments:
    - L: number of points on line
    - r: parameter for the long range decay
    - seed: inital number used to generate the random edges

    Return:
    - G: small world graph
    '''
    if seed is not None:
        np.random.seed(seed)

    np.random.seed(seed)
    G = nx.grid_2d_graph(L, L, periodic=True)

    # computation of Z from (0,0) node
    Z = 0
    y_nodes = np.array(G.nodes())
    x_repeat = np.array((0, 0))[np.newaxis, :].repeat(y_nodes.shape[0], axis=0)
    factor = np.floor(2.*abs(y_nodes-x_repeat)/L)
    dist_2d = np.sum(abs(factor*L - abs(y_nodes-x_repeat)), axis=1)
    #  is computed ONLY over the long range distance, d > 1
    Z = np.sum(dist_2d[dist_2d > 1]**(-r))

    # creation of new long range random edges
    for i, x in enumerate(list(G.nodes())[:-1]):
        # consider only nodes > x to avoid self loops and double count on the edges
        y_list = list(G.nodes())[i+1:]
        # calculate distance array
        abs_diff = abs(np.array(G.nodes())[i+1:]-np.array(x)[np.newaxis, :].repeat(int(L*L-i-1), axis=0))
        factor = np.floor(2.*abs_diff/L)
        dist_2d = np.sum(abs(factor*L - abs_diff), axis=1)
        # create list with all dges from x to any node in y_list
        x_array = np.empty(1, object)
        x_array[...] = [x]
        edge_list = np.empty((len(y_list), 2), object)
        edge_list[..., 1] = y_list
        edge_list[..., 0] = list(x_array.repeat(len(y_list)))
        # choose edges with prob = d(x,y)**(-r)/Z
        edge_list = list(edge_list[np.where(np.random.uniform(0, 1, int(L*L-i-1)) < dist_2d**(-r)/Z)[0]])
        # add edges to G
        G.add_edges_from(edge_list)

    return G


def sparse_lattice_graph(L, fraction_of_nodes_to_remove=0.1, seed=None, periodic_lattice=True):
    '''
    Constructor of a sparse lattice graph, where a part of the nodes are deleted
    by removing all their incidend edges.

    Arguments:
    - L: number of points on line
    - fraction_of_nodes_to_remove: fraction of nodes to be removed, default = 0.1
    - seed: inital number used to generate the random edges, default = None

    Return:
    - G: small world graph
    '''
    if seed is not None:
        np.random.seed(seed)

    G = nx.grid_2d_graph(L, L, periodic=periodic_lattice)

    node_list = list(G.nodes())
    node_arr = np.empty(L*L, dtype=object)
    node_arr[...] = node_list
    edge_list = list(G.edges(
        node_arr[np.random.choice(L*L, size=int(fraction_of_nodes_to_remove*L*L), replace=False)]))
    G.remove_edges_from(edge_list)

    return G
