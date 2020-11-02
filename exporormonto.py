# import unittest
# import torch
# from torch import nn
# from torchviz import make_dot, make_dot_from_trace
# import numpy as np
#
# file = open("abc.txt", "r")
# adj = open("adj.txt", "w+")
# data = []
# for line in file:
#     line_data = line.split()
#     data.append((int(line_data[0]), int(line_data[1])))
#
# adjm = np.zeros((data[0][0], data[0][0]), dtype=int)
# for i in range(data[0][1]):
#     adjm[data[i+1][0]][data[i+1][1]] = 1
#     adjm[data[i+1][1]][data[i+1][0]] = 1
#
# for i in range(data[0][0]):
#     for j in range(data[0][0]):
#         adj.write("%d" % adjm[i][j])
#         if j<data[0][0]-1:
#             adj.write(" ")
#     if i<data[0][0]-1:
#         adj.write("\n")

# First networkx library is imported
# along with matplotlib
import networkx as nx
import matplotlib.pyplot as plt


# Defining a Class
class GraphVisualization:

    def __init__(self):

        # visual is a list which stores all
        # the set of edges that constitutes a
        # graph
        self.visual = []

    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        plt.show()

# Driver code
G = GraphVisualization()
G.addEdge(0, 2)
G.addEdge(1, 2)
G.addEdge(1, 3)
G.addEdge(5, 3)
G.addEdge(3, 4)
G.addEdge(1, 0)
G.visualize() 
