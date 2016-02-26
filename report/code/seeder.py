'''
Main file for Pandemaniac
Parses JSON graph adjacecy file into networkx and produces seed nodes.

Dependencies:
Scipy: https://www.scipy.org/install.html
sudo apt-get install python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose

'''

import json, sys, itertools, os, time
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import numpy as np
import community # sudo pip install python-louvain
from operator import itemgetter
from math import sqrt
from networkx.algorithms import approximation as apxa
from analyze import graphCommunities

# Not tested yet
from parallel_betweenness_centrality import betweenness_centrality_parallel


ROUNDS = 50 # Number of rounds in a game

def write_seeds(filename, seeds):
	"""
	Writes seeds list to filename. seeds is list of (list of nodes ids)
	"""
	outstr =  ''

	#for seed_row in seeds:
	for seed in seeds:
		outstr +=str(seed)+'\n'

	if not os.path.exists('seeds/'+filename[0:filename.find('/')]):
		os.makedirs('seeds/'+filename[0:filename.find('/')])


	with open('seeds/'+filename,'w') as f:
		f.write(outstr[:-1])

def write_strategy(filename, seeds):
	"""
	Writes strategy as list for use in sim.py
	"""
	if not os.path.exists('strategy/'+filename[0:filename.find('/')]):
		os.makedirs('strategy/'+filename[0:filename.find('/')])

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
	clusters, total = graphCommunities(G)

	print runtime - time.clock()
	promising_nodes = []

	'''
	####################################################################
	# Strategy: use components to find more important regions of graph
	k_components = apxa.k_components(G)

	print runtime - time.clock()
	#print len(list(large_comp[0])), len(G.nodes())

	print k_components

	for k in [1,2,3,4,5,6,7,8,9,10]:
		if k <= len(k_components):
			subnodes = list(k_components[k-1][0])
			if len(subnodes) >= n: # Don't want tiny subcomponent
				subG = G.subgraph()
				draw_subgraph(G, subG)

				print 'computing large '+str(k)+'subgraph closeness centrality'
				subclose = nx.closeness_centrality(subG)
				subclose = sorted(subclose.items(),key=itemgetter(1), reverse=True)
				#seeds = gen_weighted_samples(subclose, 3, n)
				#write_seeds(filename+'weighted_close', seeds)
				subclose_nodes = [x[0] for x in subclose]
				write_strategy(filename+'sub'+str(k)+'close', subclose_nodes[0:n])
	####################################################################
	'''

	# Use partitions? # Was taking too long
	use_part = False
	if use_part:
		partition = community.best_partition(G)
		print partition
		#g = nx.connected_component_subgraphs(G)

		rev_partition = {}
		for k, v in partition.iteritems():
			rev_partition.setdefault(v, []).append(k)

		for key in rev_partition.keys():
			nodes = rev_partition[key]
			print 'subgraph', nodes
			draw_subgraph(G, nodes)
		exit(1)

	####################################################################
	# Strategy: Use degree (simulates TA-degree)
	deg = dict(nx.degree(G))
	deg = sorted(deg.items(),key=itemgetter(1), reverse=True)[0:2*n]
	deg_nodes = [x[0] for x in deg]
	write_seeds(filename+'/deg', deg_nodes[0:n] * ROUNDS)
	write_strategy(filename+'/deg',  deg_nodes[0:n])
	# Use degree with 20% more nodes (simulates TA-more)
	write_strategy(filename+'/more_deg', deg_nodes[0:int(1.2*n)])
	nodes = [x[0] for x in deg]
	promising_nodes.append(nodes[0:n])
	####################################################################

	print runtime - time.clock()

	####################################################################
	# Strategy: Use eigenvector centrality
	print 'computing ev centrality'

	#ev = nx.eigenvector_centrality(G)
	#ev = sorted(ev.items(),key=itemgetter(1), reverse=True)[0:2*n]
	#ev_nodes = [x[0] for x in ev]
	#write_seeds(filename+'/ev', ev_nodes[0:n] * 50)
	#write_strategy(filename+'/ev',  ev_nodes[0:n])
	#promising_nodes.append(ev_nodes[0:n])
	####################################################################

	print runtime - time.clock()

	####################################################################
	'''try:
		# Strategy: Use current_flow_betweenness_centrality centrality
		print 'computing current_flow_betweenness_centrality'

		cf_bet = nx.current_flow_betweenness_centrality(G)
		cfbet = sorted(cf_bet.items(),key=itemgetter(1), reverse=True)[0:2*n]
		cfbet_nodes = [x[0] for x in cfbet]
		write_seeds(filename+'/cfbet', cfbet_nodes[0:n] * ROUNDS)
		write_strategy(filename+'/cfbet',  cfbet_nodes[0:n])
		promising_nodes.append(cfbet_nodes[0:n])
	except nx.exception.NetworkXError:
		pass'''
	####################################################################

	print runtime - time.clock()

	####################################################################
	# Strategy: Use katz centrality
	'''try:
		print 'computing katz centrality'

		katz = nx.katz_centrality(G)
		katz = sorted(katz.items(),key=itemgetter(1), reverse=True)[0:2*n]
		katz_nodes = [x[0] for x in katz]
		write_seeds(filename+'/katz', katz_nodes[0:n] * ROUNDS)
		write_strategy(filename+'/katz',  katz_nodes[0:n])
		promising_nodes.append(katz_nodes[0:n])
	except nx.exception.NetworkXError:
		pass'''
	####################################################################

	print runtime - time.clock()

	'''
	####################################################################
	# Strategy: Cancel top $n-1$ nodes from TA-degree
	seeds = nodes[0:n-1]

	# Add node adjacent to node of highest degree
	for edge in G.edges(nodes[0]):
		if edge[1] not in seeds:
			 seeds.append(edge[1])
			 break
	write_seeds(filename+'beatdeg', seeds*50)
	write_strategy(filename+'beatdeg', seeds)
	write_strategy(filename+'unweighted_deg', nodes[0:n])
	####################################################################
	'''
	print runtime - time.clock()

	'''
	####################################################################
	# Strategy: Use closeness centrality
	print 'computing closeness centrality'
	close = nx.closeness_centrality(G)
	close = sorted(close.items(),key=itemgetter(1), reverse=True)[0:2*n]
	#seeds = gen_weighted_samples(close, 3, n)
	#write_seeds(filename+'weighted_close', seeds)
	close_nodes = [x[0] for x in close]
	write_seeds(filename+'/close', close_nodes[0:n] * ROUNDS)
	write_strategy(filename+'/close', close_nodes[0:n])
	# Simulates TA-more
	write_strategy(filename+'/more_close', close_nodes[0:int(1.2*n)])
	promising_nodes.append(close_nodes[0:n])
	####################################################################
	'''

	print runtime - time.clock()

	####################################################################
	# Strategy: Use bewteenness centrality
	print 'computing betweness centrality'
	bet = nx.betweenness_centrality(G)
	bet = sorted(bet.items(),key=itemgetter(1), reverse=True)[0:2*n]
	bet_nodes = [x[0] for x in bet]
	write_seeds(filename+'/bet', bet_nodes[0:n] * ROUNDS)
	write_strategy(filename+'/bet', bet_nodes[0:n])
	promising_nodes.append(bet_nodes[0:n])
	####################################################################

	print runtime - time.clock()

	'''
	####################################################################
	# Strategy: Use dispersion centrality
	print 'computing dispersion centrality'
	bet = nx.dispersion(G)
	dis = sorted(bet.items(),key=itemgetter(1), reverse=True)[0:2*n]
	dis_nodes = [x[0] for x in dis]
	write_seeds(filename+'/dis', dis_nodes[0:n] * ROUNDS)
	write_strategy(filename+'/dis', dis_nodes[0:n])
	promising_nodes.append(dis_nodes[0:n])
	####################################################################
	'''

	print runtime - time.clock()

	'''
	####################################################################
	if nx.is_connected(G):
		print 'computing random walk betweenness centrality'
		randwalk = nx.approximate_current_flow_betweenness_centrality(G)
		# approximate much faster than original
		rw = sorted(randwalk.items(),key=itemgetter(1), reverse=True)[0:2*n]
		rw_nodes = [x[0] for x in rw]
		write_seeds(filename+'/rw', rw_nodes)
		write_strategy(filename+'/rw', rw_nodes[0:n])
		promising_nodes.append(rw_nodes[0:n])
	####################################################################
	'''

	print runtime - time.clock()

	print promising_nodes
	filtered_nodes = dict.fromkeys([item for sublist in promising_nodes for item in sublist], 0)
	for node_list in promising_nodes:
		for i in range(len(node_list)):
			filtered_nodes[node_list[i]] += len(node_list)-i # high ranking nodes near front
	filtered_nodes = sorted(filtered_nodes.items(),key=itemgetter(1), reverse=True)

	print filtered_nodes
	if len(filtered_nodes) > n+2:
		filtered_nodes = [x[0] for x in filtered_nodes[0:n+2]]
	else:
		filtered_nodes = [x[0] for x in filtered_nodes]


	hybrid = [str(x) for x in list(set(deg_nodes[0:n] + bet_nodes[0:n] + ev_nodes[0:n]))]

	ind = 1
	for ind in range(1,100):
		hybrid = np.random.permutation(hybrid)
		seeds = [str(x) for x in hybrid]
		write_seeds(filename+'/hybrid'+str(ind), seeds[0:n] * ROUNDS)
		write_strategy(filename+'/hybrid'+str(ind), seeds[0:n])

	ind = 1
	for combo in itertools.combinations(filtered_nodes, n):
		#print combo
		seeds = [str(x) for x in combo]
		write_seeds(filename+'/promise_'+str(ind), seeds[0:n] * ROUNDS)
		write_strategy(filename+'/promise_'+str(ind), seeds[0:n])
		ind+=1


def draw(G):
	"""
	Draws a networkx graph object
	"""
	plt.figure()
	nodes = G.nodes()
	N = int(sqrt(len(nodes)))
	scale = 3
	pos = {}
	for node in nodes:
		pos[node] = [scale * (int(node) % N), scale * (int(node) / N)]

	nx.draw(G, pos, node_size=100, with_labels=False)
	plt.show()

def draw_subgraph(G, nodes):
	"""
	Draws the subgraph of networkx graph object G specified by list of
	nodes
	"""
	plt.figure()
	all_nodes = sorted(G.nodes())
	pos = {}
	deg = dict.fromkeys(all_nodes, 0) # Dict with 0 values

	# Lattice structure for nodes
	N = int(sqrt(len(all_nodes)))
	scale = 3
	pos = {}
	for node in all_nodes:
		pos[node] = [scale * (int(node) % N), scale * (int(node) / N)]

	nx.draw_networkx_nodes(G,pos, nodelist=nodes, node_color='r',
			node_size=100, alpha=0.8)
	nx.draw_networkx_nodes(G,pos, nodelist=[x for x in all_nodes if x not in nodes] \
			, node_color='b', node_size=100, alpha=0.8)

	nx.draw_networkx_edges(G,pos,width=0.5,alpha=0.5, edge_color='b')
	nx.draw_networkx_edges(G.subgraph(nodes),pos,width=0.5,alpha=0.5, edge_color='r')

	plt.xlabel('red is subgraph, blue is graph')
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

	# The below function layouts were not useful for visualization
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

	if len(colornames) > 1:
		plt.xlabel('red='+str(colornames[0]))
		plt.ylabel('blue='+str(colornames[1]))
	else:
		plt.xlabel('green='+str(colornames[0]))
	plt.savefig(filename+'.png')


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


def getClusterSeeds(graph, n):
	''' get seeds from cluster/communities'''
	clusters, total = graphCommunities(graph)

	promising_nodes = []

	for _, c in clusters.items()[0:5]:
		cluster_nodes = []

		####################################################################
		# Strategy: Use degree (simulates TA-degree)
		# find node of highest degree centrality within each cluster
		deg = dict(nx.degree_centrality(c))
		deg = sorted(deg.items(),key=itemgetter(1), reverse=True)[0:2*n]
		deg_nodes = [x[0] for x in deg]
		write_seeds(filename+'/clust_deg', deg_nodes[0:n] * ROUNDS)
		write_strategy(filename+'/clust_deg',  deg_nodes[0:n])
		nodes = [x[0] for x in deg]
		cluster_nodes.append(nodes[0:n])
		####################################################################

		print runtime - time.clock()

		####################################################################
		# Strategy: Use eigenvector centrality
		#print 'computing ev centrality'

		#ev = nx.eigenvector_centrality(c)
		#ev = sorted(ev.items(),key=itemgetter(1), reverse=True)[0:2*n]
		#ev_nodes = [x[0] for x in ev]
		#write_seeds(filename+'/clust_ev', ev_nodes[0:n] * 50)
		#write_strategy(filename+'/clust_ev',  ev_nodes[0:n])
		#cluster_nodes.append(ev_nodes[0:n])
		####################################################################

		####################################################################
		# Strategy: Use closeness centrality
		#print 'computing closeness centrality'
		#close = nx.closeness_centrality(G)
		#close = sorted(close.items(),key=itemgetter(1), reverse=True)[0:2*n]
		#seeds = gen_weighted_samples(close, 3, n)
		#write_seeds(filename+'weighted_close', seeds)
		#close_nodes = [x[0] for x in close]
		#write_seeds(filename+'/clust_close', close_nodes[0:n] * ROUNDS)
		#write_strategy(filename+'/clust_close', close_nodes[0:n])
		#cluster_nodes.append(close_nodes[0:n])
		####################################################################

		print runtime - time.clock()

		####################################################################
		# Strategy: Use bewteenness centrality
		print 'computing betweness centrality'
		bet = nx.betweenness_centrality(G)
		bet = sorted(bet.items(),key=itemgetter(1), reverse=True)[0:2*n]
		bet_nodes = [x[0] for x in bet]
		write_seeds(filename+'/bet', bet_nodes[0:n] * ROUNDS)
		write_strategy(filename+'/bet', bet_nodes[0:n])
		cluster_nodes.append(bet_nodes[0:n])
		####################################################################


		#print cluster_nodes
		filtered_nodes = dict.fromkeys([item for sublist in cluster_nodes \
			for item in sublist], 0)
		for node_list in cluster_nodes:
			for i in range(len(node_list)):
				# high ranking nodes near front
				filtered_nodes[node_list[i]] += len(node_list)-i
		filtered_nodes = sorted(filtered_nodes.items(),key=itemgetter(1), reverse=True)

		if len(filtered_nodes) > n+2:
			filtered_nodes = [str(x[0]) for x in filtered_nodes[0:n+2]]
		else:
			filtered_nodes = [str(x[0]) for x in filtered_nodes]

		promising_nodes.append(filtered_nodes)


	####################################################################
	# Take even number from each cluster
	seeds = []
	for i in range(2*n):
		seeds.append(promising_nodes[i % len(clusters)][i / len(clusters)])
	print seeds
	seeds = list(set(seeds))

	write_seeds(filename+'/cluster', seeds[0:n] * ROUNDS)
	write_strategy(filename+'/cluster',  seeds[0:n])
	####################################################################


	####################################################################
	# Take proportional to number of nodes in each cluster
	seeds2 = []
	total_size = sum([len(g.nodes()) for g in clusters.values()])
	prop= [len(g.nodes()) / float(total_size) for g in clusters.values()]
	max_ind = prop.index(max(prop))
	prop = [int(round(cluster_prop * n)) for cluster_prop in prop]

	for num in prop:
		for i in range(num):
			pot_node = promising_nodes[prop.index(num)][i]
			if pot_node not in seeds2:
				seeds2.append(pot_node)

	seeds2 += promising_nodes[max_ind][0:n]
	seeds2 = list(set(seeds2))[0:n]

	write_seeds(filename+'/cluster2', seeds2[0:n] * ROUNDS)
	write_strategy(filename+'/cluster2',  seeds2[0:n])
	####################################################################

	####################################################################
	# Take proportional to number of edges in each cluster
	seeds3 = []
	total_size = sum([len(g.edges()) for g in clusters.values()])
	prop= [len(g.edges()) / float(total_size) for g in clusters.values()]
	max_ind = prop.index(max(prop))

	prop = [int(round(cluster_prop * n)) for cluster_prop in prop]

	for num in prop:
		for i in range(num):
			pot_node = promising_nodes[prop.index(num)][i]
			if pot_node not in seeds3:
				seeds3.append(pot_node)

	seeds3 += promising_nodes[max_ind][0:n]
	seeds3 = list(set(seeds3))[0:n]
	write_seeds(filename+'/cluster3', seeds3[0:n] * ROUNDS)
	write_strategy(filename+'/cluster3',  seeds3[0:n])
	####################################################################


if __name__ == "__main__":
	#generate_graphs()
	usage = 'usage: python seeder.py num_players.num_seeds.id [time] \n' \
		+ 'input must be valid json file format'

	now = time.clock()

	# make sure correct number of arguments
	if len(sys.argv) < 2:
		print usage
		exit(1)

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
		print >> sys.stderr, usage
		exit(1)
	# read it in as networkx graph
	try:
		G = makeGraphFromJSON(filename)
	except IOError:
		print >> sys.stderr, usage
		sys.exit(1)

	#seeds = get_seeds(filename+'.seeds.', G, num_seeds, runtime - (time.clock() - now))
	seeds = getClusterSeeds( G, num_seeds)

	# visualize
	#draw(G)
