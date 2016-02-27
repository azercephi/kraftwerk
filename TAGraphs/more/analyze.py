import networkx as nx
import matplotlib.pyplot as plt
import pylab as py
from operator import itemgetter
import community
import json
#from random import random
import sys

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

def graphCommunities(filename):
	
	# generate graph
	graph = makeGraphFromJSON(filename)
	name = filename.split(".json")[0]

	# find the groups each node belongs to
	part = community.best_partition(graph)

	# get total number of clusters and set up dictionary
	clusters = {}
	total = max(part.values()) + 1
	for i in range(total):
		clusters[i] = []

	# sort nodes into clusters
	for node, group in part.items():
		clusters[group].append(node)

	# turn clusters into a separate graph
	for c, nodes in clusters.items():
		clusters[c] = graph.subgraph(clusters[c])

	## Now display clustering of original graph by fixing a point from each
	## cluster and letting networkx arrange the rest of nodes

	fixed = {}
	x = 0
	for _, c in clusters.items():
		# find node of highest degree centrality within each cluster
		deg = nx.degree_centrality(c)
		deg = sorted(deg.items(),key=itemgetter(1), reverse=True)[0]
		
		x += 1

	#return clusters, total
	# graph
	fig = plt.figure()
	plt.axis("off")

	#pos = nx.spring_layout(graph, fixed=fixed.keys(), pos=fixed)

	values = [part.get(node) for node in graph.nodes()]

	#nx.draw_networkx(graph, pos=pos, cmap = plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=True)
	nx.draw_spring(graph, cmap = plt.get_cmap('jet'), node_color = values, node_size=30, with_labels=True)
	# save figure
	fig.savefig(name + ".png")
	
def printStats(filename):
	'''
	Converts json adjacency list into networkx to calculate and print the
	graphs's 
	  - average clustering coefficient
	  - overall clustering coefficient
	  - maximum diameter
	  - average diameter
	  - number of paritions using community.best_parition
	  - modularity of community.best_partition
	'''
	g = makeGraphFromJSON(filename)
	
	print "Average Clustering Coefficient: %f" % nx.average_clustering(g)
	print "Overall Clustering Coefficient: %f" % nx.transitivity(g)
	
	connected_subgraphs = list(nx.connected_component_subgraphs(g))
	largest = max(nx.connected_component_subgraphs(g), key=len)
	print "# Connected Components: %d" % len(connected_subgraphs)
	print "    Maximal Diameter: %d" % nx.diameter(largest)
	print "    Average Diameter: %f" % nx.average_shortest_path_length(largest)

	# Find partition that maximizes modularity using Louvain's algorithm
	part = community.best_partition(g)	
	print "# Paritions: %d" % (max(part.values()) + 1)
	print "Louvain Modularity: %f" % community.modularity(part, g)
	
	
if __name__ == "__main__":

	# generate and save images of graphs


	# print graph statistics and pipe into a text file
	for filename in sys.argv[1:]:
		print filename
		printStats(filename)
		print 
		#graphCommunities(filename)
	
	

