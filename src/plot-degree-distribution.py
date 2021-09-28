import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from lightningImpementationlUtilities import strongly_connected_component_subgraphs
import calendar

def snapshot_date_parser(date):
    date = str(date)
    month = int(date[2:4])
    month_text = calendar.month_name[month]
    year_str = date[4:]
    return month_text + " " + year_str
    

def plot_degree_distribution(node_degrees):
    # compute degree distribution
    degree_distr = {}
    for node in node_degrees:
        degree = node_degrees[node]
        if degree not in degree_distr:
            degree_distr[degree] = 0
        degree_distr[degree] += 1
        
    degrees = [key for key in degree_distr]
    degrees.sort()

    plt.title(f"Degree distribution of largest scc, {snapshot_date_parser(LN_snapshot_date)}")
    plt.ylabel("number of nodes with degree x")
    plt.xlabel(f"nodes degrees")
    y_axis = [degree_distr[deg] for deg in degrees]
    # plt.xscale('symlog', linthresh=0.02)
    plt.plot(degrees, y_axis, 'o-')
    plt.grid(linestyle=':')
    plt.savefig("../results/degree-distr012021.pdf", format="pdf")
    plt.show()
  
LN_snapshot_date = 13012021 

graph = nx.read_gpickle('../snapshots/' + str(LN_snapshot_date) + '.gpickle')

# replace pkey with an integer to reduce storage size 
nodes = [v for v in graph.nodes]
node_int_to_pkey = {str(i):nodes[i] for i in range(len(nodes))} # assign integers to pkeys 
node_pkey_to_int = {node_int_to_pkey[str(i)]:str(i) for i in range(len(nodes))} # reverse dict of node_int_to_pkey 

# create new graoh replacing node names from public keys to integers 
graph_with_int_nodes = nx.DiGraph()
for edge in graph.edges:
    (u,v) = edge
    int_u, int_v = node_pkey_to_int[u], node_pkey_to_int[v]  
    graph_with_int_nodes.add_edge(int_u, int_v)
    for key in graph[u][v]:
        graph_with_int_nodes[int_u][int_v][key] = graph[u][v][key]

# pick largest connected component
k = nx.number_strongly_connected_components(graph_with_int_nodes) 
print(f'LN snapshot has: #nodes = {graph_with_int_nodes.number_of_nodes()}, #channels = {graph_with_int_nodes.number_of_edges()}, #strongly connected components = {k}')
components = [c for c in strongly_connected_component_subgraphs(graph_with_int_nodes)] 
edge_component_distribution = [components[i].number_of_edges() for i in range(len(components))]
largest_comp_index_by_edges = np.argmax(edge_component_distribution)
global largest_comp 
largest_comp = components[largest_comp_index_by_edges]
print(f"connected components of LN snapshot on {LN_snapshot_date}: {k}")

node_degrees = {node:largest_comp.degree[node] for node in largest_comp.nodes()}
# nodes_sorted_by_degree = [(node, largest_comp.degree[node]) for node in largest_comp.nodes]
# nodes_sorted_by_degree.sort(key=lambda tup: tup[1]) # sort by the second element (degree)

plot_degree_distribution(node_degrees) 