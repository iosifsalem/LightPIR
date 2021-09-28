import re
import networkx as nx
import time
import json

def duration_since(last_time):
    delta = time.time() - last_time  # duration in seconds 
    if delta < 60:
        duration = delta
        measure = 'seconds'
    elif delta < 3600:
        duration = delta/60
        measure = 'minutes'
    else:
        duration = delta/3600
        measure = 'hours'
        
    duration = int(duration*100)/100  # truncate to two decimals 
    return f'{duration} {measure}'
    
def write_and_print(filename, text):
    f = open(filename, "a")
    f.write(text+'\n')
    f.close()
    print(text)

def write_only(filename, text):
    f = open(filename, "a")
    f.write(text+'\n')
    f.close()    
    
# create a networkx graph from a text input
def create_graph(graphfile):
    graph = nx.Graph() # change to nx.DiGraph() for directed 
    f = open(graphfile, 'r') 
    out = f.readlines() 
    # n = int(out[0])
    for i in range(1, len(out)):
        l = re.split('\s+', out[i])  # edge and weight
        u = int(l[0])
        v = int(l[1])
        w = int(l[2])
        graph.add_edge(u, v, weight=w)
    f.close()
    return graph

def create_graph_from_json(json_file):
    with open(json_file, "r") as handle:
        graph_dict = json.load(handle)
        
    graph = nx.Graph() # change to nx.DiGraph() for directed 
        
    count_assymetric_edges = 0
    for edge in graph_dict["edges"]:
        # check if the edge is at least unidirectional 
        if edge["node1_policy"] != "null" or edge["node2_policy"] != "null":
            
            selected_key = "fee_base_msat"  # use base fees as edge weights             
            # selected_key = "fee_rate_milli_msat"  # alternatively: use rates as edge weights 
            
            # count edges with assymetric fees
            if int(edge["node1_policy"][selected_key]) != int(edge["node2_policy"][selected_key]):
                count_assymetric_edges += 1

            w = (int(edge["node1_policy"][selected_key]) + int(edge["node2_policy"][selected_key]))/2
        
            # add uniqe weights to edges by using the channel id as the decimal part of the current weights
            # i.e. integer part is w, decimal part is the channel id 
            constant_to_make_weight_unique = int(edge["channel_id"])*10**-len(edge["channel_id"])
            w += constant_to_make_weight_unique
    
            graph.add_edge(edge["node1_pub"], edge["node2_pub"], weight=w)
        
    print(f"There were {count_assymetric_edges} edges with assymetric {selected_key} in this lightning network snapshot.")

    return graph

# returns to G connected components of G in G.subgraph
def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

def strongly_connected_component_subgraphs(G):
    for c in nx.strongly_connected_components(G):
        yield G.subgraph(c)

def hitting_set(nodeSet, pathSet):         
    # simple greedy algorithm for finding a hitting set over the paths in pathSet

    # init node_weights dict
    node_weights = {}
    node_path_dict = {}  # node_path_dict[node] is a set including the paths (tuples) in which the node appears in 
    for node in nodeSet:
        node_weights[node] = 0
        node_path_dict[node] = set()  # will include paths in tuple form
                
    # compute node weights, i.e. number of paths each node hits
    for path in pathSet:
        for node in path:
            node_weights[node] += 1
            node_path_dict[node].add(path) 
            
    print('node_weights and node_path_dict are computed')

    sorted_node_weights = list(node_weights.items())  # returns list
    sorted_node_weights.sort(key=lambda tup: tup[1])  # sorts list by weight
        
    # complete hitting set computation by greedily adding the nodes that hit the remaining paths
    hitting_set = set()
    
    # filename = "../results/hitting-set-computation-output.txt"
    
    while pathSet:
        (max_hit_node, weight_of_max) = sorted_node_weights.pop()  # node with max weight at the moment  

        # tmp_len = len(pathSet)
        paths_to_remove = set(node_path_dict[max_hit_node])
        pathSet -= paths_to_remove  # remove paths hit by max_hit_node from pathSet, if any
        hitting_set.add(max_hit_node)
        del node_weights[max_hit_node]
        
        # write_and_print(filename, f"max hit node hit {weight_of_max} paths, paths before: {tmp_len}, paths after: {len(pathSet)}")    
        
        # recompute node_weights
        for node in node_weights:
            node_path_dict[node] -= paths_to_remove
            node_weights[node] = len(node_path_dict[node])
            
        # re-sort
        sorted_node_weights = list(node_weights.items())  # returns list
        sorted_node_weights.sort(key=lambda tup: tup[1])  # sorts list by weight
        
    return hitting_set