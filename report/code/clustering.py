'''
Attempts to cluster graphs based on algorithm described
https://www.safaribooksonline.com/library/view/social-network-analysis/9781449311377/ch04.html
Uses functions from seeder.py

'''

from seeder import makeGraphFromJSON
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

g = makeGraphFromJSON('2.10.20.json')

# find number of connected subgraphs
subgraphs = list(nx.connected_component_subgraphs(g))


# every node assigned to own cluster (think like Kruskal's)

# use 'connection matrix' find closest pair of nodes and merge into cluster

# recompute matrix, treating a cluster as a node

# repeat until all merged

# remember to choose clustering threshhold <-- exactly what does this mean?

from collections import defaultdict
import networkx as nx
import numpy
from scipy.cluster import hierarchy
from scipy.spatial import distance
import matplotlib.pyplot as plt


def create_hc(G, t=1.0):
    """
    This function was taken and slightly modified from
    https://www.safaribooksonline.com/library/view/social-network-analysis/9781449311377/ch04.html
    """
    """
    Creates hierarchical cluster of graph G from distance matrix
    Maksim Tsvetovat ->> Generalized HC pre- and post-processing to work on labelled graphs 
    and return labelled clusters
    The threshold value is now parameterized; useful range should be determined 
    experimentally with each dataset
    """

    """Modified from code by Drew Conway"""

    ## Create a shortest-path distance matrix, while preserving node labels
    labels=list(G.nodes())
    path_length=nx.all_pairs_shortest_path_length(G)
    distances=numpy.zeros((len(G),len(G)))
    i=0
    for u,p in path_length:
        j=0
        for v,d in p.items():
            distances[i][j]=d
            distances[j][i]=d
            if i==j: distances[i][j]=0
            j+=1
        i+=1

    # Create hierarchical cluster
    Y=distance.squareform(distances)
    Z=hierarchy.complete(Y)  # Creates HC using farthest point linkage
    # This partition selection is arbitrary, for illustrive purposes
    membership=list(hierarchy.fcluster(Z,t=t))
    # Create collection of lists for blockmodel
    partition=defaultdict(list)
    for n,p in zip(list(range(len(G))), membership):
        partition[p].append(labels[n])
    return list(partition.values())

def hiclus_blockmodel(G):
    # Extract largest connected component into graph H
    H=list(nx.connected_component_subgraphs(G))[0]
    # Create parititions with hierarchical clustering
    partitions=create_hc(H)
    # Build blockmodel graph
    BM=nx.blockmodel(H,partitions)

    # Draw original graph
    pos=nx.spring_layout(H,iterations=100)
    fig=plt.figure(1,figsize=(6,10))
    ax=fig.add_subplot(211)
    nx.draw(H,pos,with_labels=False,node_size=10)
    plt.xlim(0,1)
    plt.ylim(0,1)

    # Draw block model with weighted edges and nodes sized by
    # number of internal nodes
    node_size=[BM.node[x]['nnodes']*10 for x in BM.nodes()]
    edge_width=[(2*d['weight']) for (u,v,d) in BM.edges(data=True)]
    # Set positions to mean of positions of internal nodes from original graph
    posBM={}
    for n in BM:
        xy=numpy.array([pos[u] for u in BM.node[n]['graph']])
        posBM[n]=xy.mean(axis=0)
    ax=fig.add_subplot(212)
    nx.draw(BM,posBM,node_size=node_size,width=edge_width,with_labels=False)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.axis('off')
