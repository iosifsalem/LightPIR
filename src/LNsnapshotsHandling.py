import json
import networkx as nx
import numpy as np
from lightningImpementationlUtilities import strongly_connected_component_subgraphs
import matplotlib.pyplot as plt

small_datasets = [13012019, 13112018, 13092018, 13012020, 13052019]
normal_datasets = [13032019, 13072019, 13092019, 13112019, 13032020, 13052020, 13072020, 13092020, 13112020, 13012021]

for i in range(len(normal_datasets)):
    date = str(normal_datasets[i])

    G = nx.read_gpickle('../snapshots/' + date + '.gpickle')

    k = nx.number_strongly_connected_components(G)
    components = [c for c in strongly_connected_component_subgraphs(G)]

    print(f'date {date}: #nodes = {G.number_of_nodes()}, #channels = {G.number_of_edges()}, #cc = {k}')

    # node_component_distribution = [components[i].number_of_nodes() for i in range(len(components))]
    edge_component_distribution = [components[i].number_of_edges() for i in range(len(components))]

    # largest_comp_index_by_nodes = np.argmax(node_component_distribution)
    largest_comp_index_by_edges = np.argmax(edge_component_distribution)

    # if largest_comp_index_by_nodes != largest_comp_index_by_edges:
    #     print(f'largest_comp_index_by_nodes = {largest_comp_index_by_nodes}, largest_comp_index_by_edges = {largest_comp_index_by_edges}')       
    # largest_comp_index = max(largest_comp_index_by_nodes, largest_comp_index_by_edges)
 
    largest_comp = components[largest_comp_index_by_edges]  
    
    print(f'largest component (cc {largest_comp_index_by_edges}): #nodes = {largest_comp.number_of_nodes()}, #channels = {largest_comp.number_of_edges()}\n')    
    