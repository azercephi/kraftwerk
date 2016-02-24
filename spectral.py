import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
import scipy as sp
import scipy.sparse
from seeder import makeGraphFromJSON

g = makeGraphFromJSON('2.10.20.json')

## find adjacency matrix
#A = nx.adjacency_matrix(g)

## find inverse of diagonal degree matrix
#deg = g.degree(g.nodes())
#D = sp.sparse.dia_matrix(np.diag(deg.values()))
#invD = sp.sparse.dia_matrix(np.diag([1.0/d for d in deg.values()]))

## Laplacian
#L = D - A

## identity matrix
#I = sp.sparse.identity(g.number_of_nodes())

## Convert Laplacian into transition matrix
#P = I - invD * L

# find normalized Laplacian matrix
L = nx.normalized_laplacian_matrix(g)
e = numpy.linalg.eigvals(L.A)
print e
print("Largest eigenvalue:", max(e))
print "second smallest" 
print("Smallest eigenvalue:", min(e))



# do power iterations on item