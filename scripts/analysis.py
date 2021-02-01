import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count

from random_walks import simple_2d_random_walk, lazy_2d_random_walk

def get_path_forG(G, L, max_steps=int(1e2), max_walks=int(1e3),  start_node=(0,0), use_random_starting_node=False,
             random_walk_function=lazy_2d_random_walk):
    '''
    Given a Graph G and a type of random walk, it returns a list containing the nodes visited
    for a random path of time max_steps for each max_walks random path
    
    Arguments:
    - G: graph
    - L: number of nodes on line
    - r: parameter for the long range decay
    - max_steps: time of the random path
    - max_walks: number of random path (walk) considered
    - start_node: initial node
    - use_random_starting_node: if True the starting node is random
    - random_walk_function: simple_2d_random_walk_parallel for default
    
    Return:
    - np.array(paths):
        - paths[i]: list of visited nodes for the i-th walk
        - paths[i][-1]: last node visited by the i-th path
    '''
    # seed list
    seed_list = np.random.permutation(np.arange(0, max_walks)).tolist()

    # list of random walks (dim = (max_walks, max_steps))
    pool = Pool(cpu_count())
    paths = np.array(pool.starmap_async(random_walk_function, [(G, L, max_steps, start_node, use_random_starting_node, seed) for seed in seed_list]).get())[:,:,1]
    pool.close()
    #starting_paths.append(paths)
    
    return paths

def increase_walks_forG(G, L, paths, max_steps=int(1e2), max_walks=int(1e3), start_node=(0,0), use_random_starting_node=False,
             random_walk_function=lazy_2d_random_walk):
    '''
    This increases the walks of the paths got from the get_path_forG
    
    Arguments:
    - G: graph
    - L: number of nodes on line
    - r: parameter for the long range decay
    - max_steps: time of the random path
    - max_walks: number of random path (walk) considered
    - start_node: initial node
    - use_random_starting_node: if True the starting node is random
    - random_walk_function: simple_2d_random_walk_parallel for default
    
    Return:
    - np.array(paths):
        - paths[i]: list of visited nodes for the i-th walk
    '''
    # seed list
    seed_list = np.random.permutation(np.arange(0, max_walks)).tolist()

    # list of random walks (dim = (max_walks, max_steps))
    pool = Pool(cpu_count())
    new_paths = np.array(pool.starmap_async(random_walk_function, 
                        [(G, L, max_steps, start_node, use_random_starting_node, seed) 
                         for seed in seed_list]).get())[:,:,1]
    pool.close()
    increased_paths=np.append(paths,new_paths,axis=0)
    return np.array(increased_paths)

def get_new_start(G,L,paths,max_steps, max_walks, find_node):
    """
    INPUT:
        paths = is the initial paths
        max_steps = is the fixed number of steps that we want to
        max_walks = is the fixed number of paths that we want to
        find_node = is the starting node that we are looking for
    """
    new_paths=[]
    for path in paths:
        # finding the node
        idx=np.argmax((path==find_node).sum(axis=1))
        """
        idx==0 could also means that we didn't find the find_node in the path, therefore to exclude this possibility
        if the first node reached is not our find_node this means that idx=0 tells us there's no find_node in the path
        """
        if idx==0 :
            if (path[0]!=np.array(find_node)).any():
                # this means that I didn't find the node, hence I continue
                continue
        new_path=path[idx:] # new path
        
        # now we add new steps
        if len(new_path)!=max_steps :
            new_steps=max_steps-len(new_path)
            # new path with starting node (new_path[-1][0],new_path[-1][1]))
            new_path_steps=np.array(lazy_2d_random_walk(G,L,max_steps=new_steps,
                                                    start_node=(new_path[-1][0],new_path[-1][1])))[:,1]
            # here we merge the two paths
            new_path=np.append(new_path,new_path_steps,axis=0)
        new_paths.append(new_path)
    new_paths=np.array(new_paths)
    if not new_path.shape[0]==max_walks:
        # here we add new walks, precisely max_walks-new_paths.shape[0]
        new_paths=increase_walks_forG(G,L,new_paths,max_steps=max_steps, 
                              max_walks=max_walks-new_paths.shape[0],start_node=find_node)
    return new_paths

def get_mixing_time(G, L, prob, stationary, max_steps=int(1e2), max_walks=int(1e3)):
    '''
    It returns the total variance with the stationary distribution. it is computed as a statistical inference
    over a max_walks number of path and it is plotted against the time (steps)
    The total variance vector stops when it reaches 0.25
    
    Arguments:
    - G: graph
    - L: number of nodes on line
    - max_steps: time of the random path
    - max_walks: number of random path (walk) considered
    - regular_graph: if True the graph has the same degree at each nodes
    
    Return:
    - np.array(norm_vector): array containing the total variance over time index
        - norm_vector[i]: total variance at i-th time (step) computed with statistical inference with 
                          max_walks random paths.
    '''
    norm_vector=[]
    # compute L1 norm for every random walk step
    for num_steps in range(max_steps):
        variance=0.5 * np.sum(abs(prob[num_steps] - stationary))
        norm_vector.append(variance)
        # we do not need more data, then we break
        if variance < 0.25:
            break
    # we can use index(norm_vector[-1])+1 as mixing time
    return np.array(norm_vector)

def get_prob_distribution(G, L, max_steps=int(1e2), max_walks=int(1e3)):
    '''
    It returns the probability distribution per steps
    
    Arguments:
    - G: graph
    - L: number of nodes on line
    - max_steps: time of the random path
    - max_walks: number of random path (walk) considered
    
    Return:
    - prob_per_steps: is a list containing max_steps array of probability distribution
        - prob_per_steps[i]: probability distribution associated to the i-th step
    '''
    # get a path
    paths=get_path_forG(G,L,max_steps=max_steps,max_walks=max_walks)
    prob_per_steps=[]
    for num_steps in range(max_steps):
        prob, _ = np.histogram(a=paths[:max_walks,num_steps,0]*L + paths[:max_walks,num_steps,1],
                               bins=np.arange(0,L*L+1), density=True)
        prob=np.array(prob)
        prob_per_steps.append(prob)
    return prob_per_steps

def increase_prob_distribution(G, L, prob, new_max_walks, max_steps=int(1e2)):
    '''
    Increasing the number of walks for the statistic with new_walks 
    It returns the probability distribution per steps
    
    Arguments:
    - G: graph
    - L: number of nodes on line
    - max_steps: time of the random path
    - max_walks: number of random path (walk) considered
    
    Return:
    - prob_per_steps: is a list containing max_steps array of probability distribution
        - prob_per_steps[i]: probability distribution associated to the i-th step
    '''
    paths=get_path_forG(G,L,max_steps=max_steps,max_walks=new_max_walks)
    prob_per_steps=[]
    # compute L1 norm for every random walk step
    for num_steps in range(max_steps):
        add_prob, _ = np.histogram(a=paths[:new_max_walks,num_steps,0]*L + paths[:new_max_walks,num_steps,1],
                               bins=np.arange(0,L*L+1), density=True)
        add_prob=np.array(add_prob)
        # we divide by 2 in order to normilize the new prob distribution
        prob_per_steps.append((add_prob+prob[num_steps])/2)
    return prob_per_steps



def get_L1_norm_over_step_count(G, L, prob, max_steps=int(1e2), max_walks=int(1e3), regular_graph=True):
    '''
    It returns the total variance with the stationary distribution. it is computed as a statistical inference
    over a max_walks number of path and it is plotted against the time (steps)
    
    Arguments:
    - G: graph
    - L: number of nodes on line
    - max_steps: time of the random path
    - max_walks: number of random path (walk) considered
    - regular_graph: if True the graph has the same degree at each nodes
    
    Return:
    - np.array(norm_vector): array containing the total variance over time index
        - norm_vector[i]: total variance at i-th time (step) computed with statistical inference with 
                          max_walks random paths.
    '''
    norm_vector=[]
        
    # compute L1 norm for every random walk step
    for num_steps in range(max_steps):
        norm_vector.append(0.5 * np.sum(abs(prob[num_steps] - stationary)))
        
    return np.array(norm_vector)

def get_L1_norm(G, L, paths, max_steps=int(1e2), max_walks=int(1e3), regular_graph=False):
    '''
    It returns the total variance with the stationary distribution. it is computed as a statistical inference
    over a max_walks number of path and it is plotted against the time (steps)
    
    Arguments:
    - G: graph
    - L: number of nodes on line
    - max_steps: time of the random path
    - max_walks: number of random path (walk) considered
    - regular_graph: if True the graph has the same degree at each nodes
    
    Return:
    - np.array(norm_vector): array containing the total variance over time index
        - norm_vector[i]: total variance at i-th time (step) computed with statistical inference with 
                          max_walks random paths.
    '''
    # compute L1 norm for every random walk step
    norm_vector=[]
    for num_steps in range(max_steps):
        prob, _ = np.histogram(a=paths[:max_walks,num_steps,0]*L + paths[:max_walks,num_steps,1], bins=np.arange(0,L*L+1), density=True)
        # L1 norm using node distribution from random walks and stationary distriubtion
        # pi_v = d_v/(2*E), with d_v the degree of vertex/node v. For the periodic
        # lattice d_v=4 and hence pi_v = 1/ #nodes_in_G = 1/L*L
        if regular_graph:
            norm_vector.append(0.5 * np.sum(abs(prob - np.ones(prob.shape[0])/(L*L))))
        
        # If the graph is not regular (e.g. small world) the stationary distribution is not uniform
        else:
            #definition of stationary probability vector
            list_degree=[]
            for v in G:
                list_degree.append(nx.degree(G,v))
            degree=np.array(list_degree)
            norm_vector.append(0.5 * np.sum(abs(prob - degree/(2*nx.number_of_edges(G)))))
    
    return np.array(norm_vector)


