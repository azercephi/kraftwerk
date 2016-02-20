'''
Main file for Pandemaniac
Parses JSON graph adjacecy file into networkx and produces seed nodes.
'''

import json, sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def makeGraphFromJSON(filename):
    """
    Generates undirected networkx graph object from JSON formatted adjacency file.
    """

    # initialize variables
    graph = nx.Graph()
    edgeData= {}
    
    # retrive edge data
    with open(filename) as dataFile:
        edgeData = json.load(dataFile)
    
    # generate graph by adding edges
    for node1 in edgeData:
        # if there are outgoing edges
        if edgeData[node1]:
            for node2 in edgeData[node1]:
                graph.add_edge(node1, node2)
                
    return graph

if __name__ == "__main__":
    
    try:
        # make sure correct number of arguments
        if len(sys.argv) != 2:
            raise IndexError()

        # get filename
        filename = sys.argv[1]
        
        # make sure it's a json file
        if filename.split('.')[1] != "json":
            raise IOError()
        
        # read it in as networkx graph
        graph = makeGraphFromJSON(filename)

        # visualize
        nx.draw(graph, node_size=100)

        plt.show()
    
    except IndexError:
        print >> sys.stderr, "usage: python seeder.py *.json"
        sys.exit(1)

    except IOError:
        print >> sys.stderr, "usage: python seeder.py *.json"
        print >> sys.stderr, "    input must be valid *.json file format"
        sys.exit(1)
