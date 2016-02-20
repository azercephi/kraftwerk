'''
Main file for Pandemaniac
Parses JSON graph adjacecy file into networkx and produces seed nodes.

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


ROUNDS = 50 # Number of rounds in a game

def write_seeds(filename, seeds):
	"""
	Writes seeds list to filename. seeds is list of (list of nodes ids)
	"""
	outstr =  ''
	
	#for seed_row in seeds:
	for seed in seeds:
		outstr +=str(seed)+'\n'

	with open('seeds/'+filename,'w') as f:
		f.write(outstr[:-1])
		
def write_strategy(filename, seeds):
	"""
	Writes strategy as list for use in sim.py
	"""
	with open('strategy/'+filename,'w') as f:
		f.write(str('['+','.join(seeds)+']')) # gets rid of l=[u'1',u'2'] unicode
		
def gen_weighted_samples(values, power, n):
	"""
	Generates ROUNDS random samples of n points, from values list weighted
	by weight**power.
	values = [(node, weight), (node, weight), ...]
	"""
	
	# Probability distribution of weights
	weights = [x[1]**power for x in values]
	sum_w = sum(weights)
	weights = [w / float(sum_w) for w in weights]
	nodes = [x[0] for x in values]
	
	# Weighted random choice of nodes from top degrees
	seeds = []
	for i in range(ROUNDS):	
		sample = np.random.choice(nodes, n, p=weights, replace=False)
		seeds.append(sorted(sample))
		
	return seeds

def get_seeds(filename, G, n, runtime):
	"""
	Generate n seeds for networkx graph object G using at most 'runtime' seconds
	"""	
		
	deg = nx.degree(G)	
	# Not actually faster up to 5000 nodes.
	#top_deg = heapq.nlargest(10,deg, key=lambda k:deg[k])
	
	top_deg = sorted(deg.items(),key=itemgetter(1), reverse=True)[0:2*n]
	#write_seeds(filename+'top_deg', top_deg[0:2*n] * 50)
	top_nodes = [x[0] for x in top_deg]
	
	
	# Cancel top $n-1$ nodes from TA-degree
	seeds = top_nodes[0:n-1]
	
	# Add node adjacent to node of highest degree
	for edge in G.edges(top_nodes[0]):
		if edge[1] not in seeds:
			 seeds.append(edge[1])
			 break	 
	
	write_seeds(filename+'top_beatdeg', seeds*50)
	write_strategy(filename+'top_beatdeg', seeds)	
	write_strategy(filename+'unweighted_top_deg', top_nodes[0:n])
	return
	
	#seeds = gen_weighted_samples(top_deg, 3, n)
	#write_seeds(filename+'weighted_top_deg', seeds)
	
	
	
	print runtime - time.clock()


	partition = community.best_partition(G)
	#print partition	
	#g = list(nx.connected_component_subgraphs(G))	
	#draw(g[0])
	
	print runtime - time.clock()

	print 'computing betweness centrality'
	bet = nx.betweenness_centrality(G)
	top_bet = sorted(bet.items(),key=itemgetter(1), reverse=True)[0:2*n]
	
	seeds = gen_weighted_samples(top_bet, 3, n)
	write_seeds(filename+'weighted_top_bet', seeds)
	
	print runtime - time.clock()

	
	print 'computing closeness centrality'
	close = nx.closeness_centrality(G)
	top_close = sorted(close.items(),key=itemgetter(1), reverse=True)[0:2*n]
	seeds = gen_weighted_samples(top_close, 3, n)
	write_seeds(filename+'weighted_top_close', seeds)
	
	print runtime - time.clock()

	
def draw(G):
	"""
	Draws a networkx graph object
	"""
	nx.draw(G, pos=nx.spring_layout(G), node_size=100, with_labels=False)
	plt.show()
	
def draw_dict(filename, colors, adjlist):
	"""
	Draws a graph with dict 'colors': {nodeid: color, nodeid:color}
	and adjacency list 'adjlist': {nodeid: [nodeid, nodeid, ...], ...}
	
	"""
	plt.figure()
	# to make graph object consistent across function calls
	nodes = sorted(adjlist.keys()) 
	pos = {}
	deg = dict.fromkeys(nodes, 0) # Dict with 0 values
	
	G = nx.Graph()
	# generate graph by adding edges
	for node1 in nodes:
		# if there are outgoing edges
		if adjlist[node1]:
			for node2 in adjlist[node1]:
				G.add_edge(str(node1), str(node2), {'weight':10})
				deg[node1] += 1
	
	
	#pos=nx.spectral_layout(G, scale=2) # positions for nodes
	#pos=nx.circular_layout(G, scale=2) # positions for nodes
	
	# Lattice structure for nodes
	N = int(sqrt(len(nodes)))
	scale = 3
	pos = {}
	for node in nodes:
		pos[node] = [scale * (int(node) % N), scale * (int(node) / N)]	
		
	# list of strategy names
	colornames = sorted(list(set([x for x in colors.values() if x is not None])))
	
	blue = []
	nocolor= []
	red = [str(x) for x in colors if colors[x] == colornames[0]]
	if len(colornames) > 1:
		blue = [str(x) for x in colors if colors[x] == colornames[1]]
		nocolor = [str(x) for x in colors if (colors[x] != colornames[0] and \
			colors[x] != colornames[1])]
	
	if len(colornames) > 1:
		nx.draw_networkx_nodes(G,pos, nodelist=red, node_color='r',
			node_size=100, alpha=0.8)
		nx.draw_networkx_nodes(G,pos, nodelist=blue, node_color='blue', \
			node_size=100, alpha=0.8)
			
	else: # only one color, use unique color
		nx.draw_networkx_nodes(G,pos, nodelist=red, node_color='g',
			node_size=100, alpha=0.8)		
	nx.draw_networkx_nodes(G,pos, nodelist=nocolor, node_color='white', \
		node_size=100, alpha=0.8)
     
	nx.draw_networkx_edges(G,pos,width=0.5,alpha=0.5)
	
	nx.draw_networkx_labels(G,pos,deg,font_size=6)
		
	
	#plt.show()
	
	if len(colornames) > 1:
		plt.xlabel('red='+str(colornames[0]))
		plt.ylabel('blue='+str(colornames[1]))
	else:
		plt.xlabel('green='+str(colornames[0]))
	plt.savefig('figs/'+filename+'.png')
	


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
	
def save(G, filename, num_players, num_seeds):
	"""
	Saves networkx graph object to JSON formatted adjacency file.
	"""
	with open(str(num_players)+'.'+str(num_seeds)+'.'+filename,'w') as f:
		f.write('{\n')
		for n in G.nodes():
			f.write('\t"'+str(n)+'": ')
			edgstr = '['
			if len(G.edges(n))>0:
				for edge in G.edges(n):
					edgstr += '"'+str(edge[1])+'", '
				edgstr = edgstr[:-2]+ '],\n'
			else:
				edgstr +=']\n'
			if n == G.nodes()[-1]:
				edgstr = edgstr[:-2]
			f.write(edgstr)
		f.write('\n}')
		
def generate_graphs():
	"""
	Generates sample graphs for testing
	"""
	#er=nx.erdos_renyi_graph(1000, 0.05)

	
	ws=nx.watts_strogatz_graph(3000,3,0.1)
	save(ws, 'WS_3000_3_01', 2, 10)
	draw(ws)
	
	ba=nx.barabasi_albert_graph(1000,10)
	save(ba, 'BA_1000_5',2,10)
	draw(ba)
	
	red=nx.random_lobster(1000,0.9,0.10)
	save(red, 'LOB_1000_09', 2, 10)
	draw(red)
		
	exit(1)
	

if __name__ == "__main__":
	#generate_graphs()
	
	now = time.clock()	
	try:
		# make sure correct number of arguments
		if len(sys.argv) < 2:
			raise IndexError()
			
		# num seconds of running time allowed	
		runtime = 180 if len(sys.argv) == 2 else int(sys.argv[2])
		print runtime				

		# get filename
		filename = sys.argv[1]
		
		# make sure it's a json file
		try:
			split = filename.split('.')
			num_players = int(split[0])
			num_seeds = int(split[1])
			id = split[2:]
		except ValueError:
			print >> sys.stderr, "usage: python seeder.py num_players.num_seeds.id [time]"
			exit(1)
		# read it in as networkx graph
		try:
			G = makeGraphFromJSON(filename)
		except IOError:
			print >> sys.stderr, "usage: python seeder.py num_players.num_seeds.id [time]"
			print >> sys.stderr, "    input must be valid json file format"
			sys.exit(1)
		
		seeds = get_seeds(filename+'.seeds.', G, num_seeds, runtime - (time.clock() - now))		

		# visualize
		draw(G)
	
	except IndexError:
		print >> sys.stderr, "usage: python seeder.py num_players.num_seeds.id [time]"
		sys.exit(1)


