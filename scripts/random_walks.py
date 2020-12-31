import numpy as np
import networkx as nx


def simple_2d_random_walk(G, L, max_steps, start_node=(0, 0), use_random_starting_node=False, seed=None):
    '''
        Perform a simple random walk on a given graph G, which should be based on a 2d lattice. In one step the random
        walker with equal probability traverses any of the incident edges of the current node to a neighboring node.

        Arguments:
        - G: graph, based on a 2d lattice
        - L: side length L of the 2d lattice base of G
        - max_steps: number of steps the random walker takes in total
        - start_node: node to start random walk from, default=(0,0)
        - use_random_starting_node: start on a randomly chosen node, default=False
        - seed: use a seed for np.random, default=None

        Return:
        - path: list of edges the random walker traversed (format: [(node1, node2), (node2, node3)])
    '''
    if seed is not None:
        np.random.seed(seed)

    if use_random_starting_node:
        current_node = tuple(np.random.randint(0, L, 2))
    else:
        current_node = start_node

    path = []
    # for-loop over max_steps
    for t in range(max_steps):
        # randomly choose from neighbors of current node
        new_node = tuple(list(
            G.neighbors(current_node)
            )[np.random.randint(0, nx.degree(G, current_node))])
        # append new node to path
        path.append((current_node, new_node))
        current_node = new_node
    return path


def lazy_2d_random_walk(G, L, max_steps, start_node=(0, 0), use_random_starting_node=False, seed=None):
    '''
        Perform a lazy random walk on a given graph G, which should be based on a 2d lattice. In one step the random
        walker stays at the current node with probability 1/2 and with probability 1/2d_n traverses any of the d_n
        incident edges of the current node to a neighboring node.

        Arguments:
        - G: graph, based on a 2d lattice
        - L: side length L of the 2d lattice base of G
        - max_steps: number of steps the random walker takes in total
        - start_node: node to start random walk from, default=(0,0)
        - use_random_starting_node: start on a randomly chosen node, default=False
        - seed: use a seed for np.random, default=None

        Return:
        - path: list of edges the lazy random walker traversed (format: [(node1, node2), (node2, node3)])
    '''
    if seed is not None:
        np.random.seed(seed)

    if use_random_starting_node:
        current_node = tuple(np.random.randint(0, L, 2))
    else:
        current_node = start_node

    path = []
    # for-loop over max_steps
    for t in range(max_steps):
        # stay at current node with p=1/2 or move to a neighboring
        # node with each p=1/2d_n (d_n degree of current node)
        if np.random.uniform(0, 1) <= 0.5:
            new_node = current_node
        else:
            new_node = tuple(list(G.neighbors(current_node))[np.random.randint(0, nx.degree(G, current_node))])
        # append new node to path
        path.append((current_node, new_node))
        current_node = new_node
    return path
