import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

from random_walks import simple_2d_random_walk, lazy_2d_random_walk

def plot_walk(G, L, total_path, steps=-1, figure_size=(5,5)):
    # take only specified steps of path (-1 -> complete path)
    path = total_path[:steps]
    # color map
    color_map=[]
    for node in G:
        if node == path[0][0]:
            color_map.append('red') #INITIAL POINT RED
        elif node == path[-1][1]: 
            color_map.append('blue') #END POINT BLUE
        else:
            color_map.append('lightgreen')

    # FIGURE
    plt.figure(figsize=figure_size)
    # make graph R to display the path
    R = nx.grid_2d_graph(L,L,periodic=True)
    R.remove_edges_from(list(R.edges))
    R.add_edges_from(path)
    pos = {(x,y):(y,-x) for x,y in G.nodes()}
    
    # draw underlying network
    nx.draw(G, pos=pos, node_color='lightgreen', edge_color='lightgray',
        with_labels=False, node_size=100, width=3)
    # draw path using R
    nx.draw(R, pos=pos, node_color=color_map, with_labels=False,
        node_size=100, edge_color='red', width=2)
    plt.show()
    
def plot_2d_graph(G,labels=False):
    pos = {(x,y):(y,-x) for x,y in G.nodes()}
    nx.draw(G, pos=pos, node_color='lightgreen', edge_color='lightgray',
        with_labels=labels, node_size=100, width=3)
    plt.show()
