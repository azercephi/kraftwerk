'''
Main file for Pandemaniac
Parses JSON graph adjacecy file into networkx and produces seed nodes.

Dependencies:
Scipy: https://www.scipy.org/install.html
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

'''

import json, sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.readwrite import json_graph
import heapq
import time
import community # sudo pip install python-louvain
from operator import itemgetter
from math import sqrt
from networkx.algorithms import approximation as apxa
import itertools
import os
import sys

def reformat(filename):
	"""
	Generates undirected networkx graph object from JSON formatted adjacency file.
	"""

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
	
	# get filename
	filename = sys.argv[1]
	n = int(filename.split('.')[1])
	
	if not os.path.exists('strategy/'+filename):
		os.makedirs('strategy/'+filename)	
	
	with open(filename) as dataFile:
		
		players = json.load(dataFile)		
		ind=1
		for nodes in players["TA_more"]:
			stratfile = 'TA_more_'+str(ind)
			with open('strategy/'+filename+'/'+stratfile, 'w+') as t:
				out= [int(x) for x in nodes]
				out = '['+','.join([str(x) for x in out])+']'
				t.write(str(out))
			ind+=1
			
		
		for player in players.keys():
			seedfile = filename.split('-')[0] + "_" + str(player)
			s = open(seedfile, 'w+')			
			for nodeset in players[player]:
				
				
				
				for node in nodeset:
					s.write(node)
					s.write("\n")
				ind =0


