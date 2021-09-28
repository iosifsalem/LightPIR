# computes highway dimension of a given graph
# implementation of Definition 2 from the paper:
# "Highway Dimension, Shortest Paths, and Provably Efficient Algorithms"

import time
from datetime import datetime
# import pickle
# import re
import json
import numpy as np
# import random 
from random import choice
import networkx as nx
import ast
import concurrent.futures
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from lightningImpementationlUtilities import duration_since, write_and_print, write_only, strongly_connected_component_subgraphs, hitting_set

def plot_degree_distribution(node_degrees):
    # compute degree distribution
    degree_distr = {}
    for node in node_degrees:
        degree = node_degrees[node]
        if node not in degree_distr:
            degree_distr[degree] = 0
        degree_distr[degree] += 1
        
    degrees = [key for key in degree_distr]
    degrees.sort()

    plt.title(f"Degree distribution for the large component")
    plt.ylabel("number of nodes with degree x")
    plt.xlabel(f"nodes degrees")
    y_axis = [degree_distr[deg] for deg in degrees]
    plt.plot(degrees, y_axis, 'o-')
    plt.show()

def plot_component(graph, title):
    plt.title(title)
    pos = nx.spring_layout(graph)
    # if component_index == 1:
    #     pos[2604][1] = pos[5050][1]
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_edges(graph, pos)
    plt.show()

def ball_intersection_cover(args):
    # computes the intersection of Ball(u, 2*r) with cover[r]     

    [u, r] = args  # read input
    
    ball_with_core = nx.ego_graph(largest_comp, u, radius=2*r)
    ball_no_core = nx.ego_graph(largest_comp_no_core, u, radius=2*r)
    
    cover_ball_hits_no_core = cover[r].intersection(set(ball_no_core.nodes()))
    
    # check that the cover/ball intersection is non empty when inluding the core nodes 
    if len(cover_ball_hits_no_core) == 0:
        cover_ball_hits_with_core = cover[r].intersection(set(ball_with_core.nodes()))
        
        if len(cover_ball_hits_with_core) == 0:
            write_only(output_filename, f'ball_with_core({u},{r}) intersection cover[{r}] = empty. |ball_with_core.nodes| = {len(ball_with_core.nodes())}, |ball_with_core.edges| = {len(ball_with_core.edges())}, diameter(ball_with_core) = {nx.diameter(ball_with_core)}')
            # plot_component(ball_with_core, f'ball_with_core({u},{r}) intersection cover[{r}]')
        
    return (u, r, cover_ball_hits_no_core)

def plot_hub_size_distribution(hub_dict):
    hub_size_distr = {}
    
    # compute distribution 
    for node in hub_dict:
        hub_size = str(len(hub_dict[node]))
        if hub_size not in hub_size_distr:
            hub_size_distr[hub_size] = '1'
        else:
            hub_size_distr[hub_size] = str(int(hub_size_distr[hub_size]) + 1)

    hub_size_distr_file = '../results/hub-size-distr.JSON'
    with open(hub_size_distr_file, 'w') as handle:
        json.dump(hub_size_distr_file, handle)

def compute_next_core_size(core_to_HD, current_num_of_high_deg_nodes):
    # decides next core size and if the HD is minimized (according to our heuristic) 
    
    preset_sizes = [0, 50, 100, 200, 300, 400, 500, 600, 700] #700 should never be reached according to the current LN characteristics 

    if current_num_of_high_deg_nodes == 0:
        num_of_high_deg_nodes, opt_core_size_found = 50, False # set up second attempt of core computation 
        
    elif current_num_of_high_deg_nodes in preset_sizes:
        curr_HD = core_to_HD[current_num_of_high_deg_nodes]
        curr_num_of_high_deg_nodes_index = preset_sizes.index(current_num_of_high_deg_nodes)
        prev_num_of_high_deg_nodes = preset_sizes[curr_num_of_high_deg_nodes_index - 1]
        prev_HD = core_to_HD[preset_sizes[curr_num_of_high_deg_nodes_index - 1]]
        
        if curr_HD > prev_HD:
            # start binary search 
            opt_core_size_found = False
            
            core_to_HD['left margin'] = preset_sizes[max(0, curr_num_of_high_deg_nodes_index - 2)] # left margin of binary search should be the second last preset size or 0 if not possible 
            
            core_to_HD['right margin'] = current_num_of_high_deg_nodes
            
            num_of_high_deg_nodes = int((core_to_HD['left margin'] + core_to_HD['right margin'])/2)

        else:
            # continue to next preset size precomputation_done_for_largest_comp
            num_of_high_deg_nodes, opt_core_size_found = preset_sizes[curr_num_of_high_deg_nodes_index + 1], False
            
    else:
        # binary search has started already 
        
        core_size_left = core_to_HD['left margin']
        left_HD = core_to_HD[core_size_left]

        core_size_right = core_to_HD['right margin']
        right_HD = core_to_HD[core_size_right]
        
        if core_size_left == core_size_right + 1:
            # binary search has finished!
            print(f'binary search has finished!')
            
            opt_core_size_found = True
            
            if left_HD >= right_HD:
                num_of_high_deg_nodes = core_to_HD['left margin']
            else:
                num_of_high_deg_nodes = core_to_HD['right margin']
        
        else:
            opt_core_size_found = False 
            
            #compute new left/right margins 
            if left_HD <= right_HD:
                core_to_HD['right margin'] = current_num_of_high_deg_nodes
            else:
                core_to_HD['left margin'] = current_num_of_high_deg_nodes

            num_of_high_deg_nodes = int((core_to_HD['left margin'] + core_to_HD['right margin'])/2) 
            
            # break loop 
            if num_of_high_deg_nodes in core_to_HD:
                print('binary search finished! (loop detected)')
                core_to_HD['loop detected'] = True
                opt_core_size_found = True 
            
        write_and_print(output_filename, f"core_to_HD['left margin'] = {core_to_HD['left margin']}, core_to_HD['right margin'] = {core_to_HD['right margin']}")
                                    
    return num_of_high_deg_nodes, opt_core_size_found

def lightningHD(LN_snapshot_date):
    # returns HD and hub sizes, stores hubs in json, writes HD and other info in output file 
    
    ## initializing the largest connected component with unique weights
    start_time = time.time()
    last_time = time.time()
    
    # file names init 
    global output_filename 
    output_filename = f"../results/results-for-large-component-of-LNsnapshot-on-{LN_snapshot_date}.txt"
    f = open(output_filename, "w")
    f.write(f"Results for lightning's large component on {LN_snapshot_date}, version {version}\n\n")
    f.close()
    
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
    write_and_print(output_filename, f'LN snapshot has: #nodes = {graph_with_int_nodes.number_of_nodes()}, #channels = {graph_with_int_nodes.number_of_edges()}, #strongly connected components = {k}')
    components = [c for c in strongly_connected_component_subgraphs(graph_with_int_nodes)] 
    edge_component_distribution = [components[i].number_of_edges() for i in range(len(components))]
    largest_comp_index_by_edges = np.argmax(edge_component_distribution)
    global largest_comp 
    largest_comp = components[largest_comp_index_by_edges]
    write_and_print(output_filename, f'largest component (cc {largest_comp_index_by_edges}): #nodes = {largest_comp.number_of_nodes()}, #channels = {largest_comp.number_of_edges()}')
    
    # assign integer edge weights: can be based either on 'fee_base_msat' or on 'fee_proportional_millionths'
    # used_random_perturbations = set()
    deterministic_perturbation = 1  #practically equivalent to random perturbation below, but faster to compute  
    for edge in largest_comp.edges:
        (u,v) = edge
        
        # set edge (integer) weights: current weight is the base fee*10^8 plus a random perturbation in [1,1000] 
        # perturbation is for reducing the number of shortest paths with the same weight 
        # random_perturbation = choice([i for i in range(1,largest_comp.number_of_edges()+1) if i not in used_random_perturbations])
        # used_random_perturbations.add(random_perturbation)
        largest_comp[u][v]['weight'] = largest_comp[u][v]['fee_base_msat']*10**6 + deterministic_perturbation
        deterministic_perturbation += 1
                
    ## prep for selection of high degree nodes 
    # sorting nodes by degree and degree distribution plot        
    nodes_sorted_by_degree = [(node, largest_comp.degree[node]) for node in largest_comp.nodes]
    nodes_sorted_by_degree.sort(key=lambda tup: tup[1]) # sort by the second element (degree)
    
    # plot_degree_distribution(nodes_sorted_by_degree)
    
    # init high_deg_nodes 
    num_of_high_deg_nodes = 0
    high_deg_nodes = set()
    
    dataset_info_filename = 'dataset-info-batch'+str(batch)+'.json'
    
    with open(dataset_info_filename, 'r') as handle:
        dataset_info = json.load(handle)
    
    # if diameter is computed, load it from file, else compute it 
    if 'diameter' in dataset_info[str(LN_snapshot_date)]:
        diameter = dataset_info[str(LN_snapshot_date)]['diameter']
    else: 
        diameter = nx.diameter(largest_comp, e=None)       
        dataset_info[str(LN_snapshot_date)]['diameter'] = diameter
        with open(dataset_info_filename, 'w') as handle:
            json.dump(dataset_info, handle)
    
    write_and_print(output_filename, f'network diameter = {diameter}, computed in {duration_since(last_time)}')
    last_time = time.time()
       
    # radii relevant for HD computation 
    # radii = range(1,int(np.ceil(diameter/2)) + 1)
    radii = [2**i for i in range(diameter) if 2**i <= diameter]
    write_and_print(output_filename, f"radii = {radii}")
    reversed_radii = list(radii)
    reversed_radii.reverse()
    
    # plot_component(largest_comp, f"largest component of lightning network, date {date}")
    
    # compute APSP and write it to a text file
    # APSP_file = "lightning" + str(LN_snapshot_date) + "-APSP-largest-component.txt"
    APSP_file = "lightning-APSP-largest-component-batch"+str(batch)+".txt" # resuse the same file to save space 
    
    APSP_computed = False
    
    if not APSP_computed:
        print('Computing APSP...')
        # https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html 
        APSP = dict(nx.all_pairs_dijkstra_path(largest_comp))  # changed nx.all_pairs_shortest_path to Dijkstra to take into account edge weights 
        f = open(APSP_file, "w")
        for source in APSP:
            f.write(f'source:{source}\n')
            f.write(str(APSP[source])+'\n')
        f.close()
        write_and_print(output_filename, f'wrote APSP of largest component in {APSP_file}, in {duration_since(last_time)}')
        last_time = time.time()
    
    # init 
    global cover 
    cover = {} 
    nodePool = {}
    
    for r in radii:
        nodePool[r] = set() 
    
    # file name generator 
    file_radius_r = lambda r : "../prep/lightning" + str(LN_snapshot_date) + f"-APSP-largest-component-path-pool-{r}.txt"
    
    if 'precomputation_done_for_largest_comp' in dataset_info[str(LN_snapshot_date)]:
        precomputation_done_for_largest_comp = dataset_info[str(LN_snapshot_date)]['precomputation_done_for_largest_comp']    
    else:
        precomputation_done_for_largest_comp = False
    
    # read APSP from file in a memory-efficient manner, store pathpool and nodepool for hitting set computation  
    if not precomputation_done_for_largest_comp:      
        # init files to store paths according to their size 
        print("reading APSP, partitioning paths for each radius...")
        
        # init file 
        for r in radii:        
            f = open(file_radius_r(r), "w")
            f.close()
            
        # store paths of length (r, 2r] in file_radius_r(r), given that APSP file for component0 already exists.
        f = open(APSP_file, "r")
        for line in f:
            if line[0] == 's': 
                pass
            #     # in this case line == 'source:id'
            #     src = int(line[7:][:-1])  # everything after 'source:' and before '\n' 
            else:
                paths = ast.literal_eval(line)  # convert str(dict) to dict 
                for dest in paths:
                    path = paths[dest]
                    path_length = len(path) - 1  # number of nodes in path - 1 
                    for r in radii:
                        if r < path_length and path_length <= 2*r:
                            write_only(file_radius_r(r), str(path)) # add path to list of paths with lenght in (r,2r] 
                            nodePool[r] |= set(path) # add nodes of path in nodePool[r] 
        f.close()
    
        write_and_print(output_filename, f"APSP read and processed in {duration_since(start_time)}")
        last_time = time.time() 
        
        # report in dataset_info that paths are partitioned already 
        with open(dataset_info_filename, 'r') as handle:
            dataset_info = json.load(handle)
        dataset_info[str(LN_snapshot_date)]['precomputation_done_for_largest_comp'] = True
        with open(dataset_info_filename, 'w') as handle:
            json.dump(dataset_info, handle)
                        
    # binary search the core size that minimizes the HD 
    opt_core_size_found = False
    core_to_HD = {}
    last_iteration = False
    
    while not opt_core_size_found or not last_iteration:
        # if opt core size found, do one last iteration 
        if opt_core_size_found:
            last_iteration = True 
        
        write_and_print(output_filename, f"Iteration for computing min HD with core size {num_of_high_deg_nodes}")
    
        high_deg_nodes = set([nodes_sorted_by_degree[i][0] for i in range(-num_of_high_deg_nodes,0)])
    
        # init for checking how many paths are hit by high_deg_nodes for each r in radii
        paths_hit_by_high_deg_nodes = {} # num of paths that interesect with high deg nodes
        paths_not_hit_by_high_deg_nodes = {}
        
        covers_file = f"../results/covers-LNsnapshot-on-{LN_snapshot_date}.txt"
        f = open(covers_file, "w")
        f.close()
        
        # heuristic: compute hitting sets with base = high_deg_nodes
        for r in reversed_radii: 
            last_time = time.time() 
            print(f'starting computation of cover[{r}]')
            
            # inits for checking if path weights are unique 
            set_of_path_weights = set()
            exists_repetition = False
            
            paths_hit_by_high_deg_nodes[r] = 0
            paths_not_hit_by_high_deg_nodes[r] = 0     
            
            # load paths to pathPool
            pathPool = set()
            f = open(file_radius_r(r), "r")
            for line in f:
                path = ast.literal_eval(line)  # str to list 
                
                # add path in the pool if it's not hit by the high_deg_nodes set 
                if set(path).intersection(high_deg_nodes):  # exclude paths hit by the high_deg_nodes 
                    paths_hit_by_high_deg_nodes[r] += 1
                else:
                    pathPool.add(tuple(path))
                    nodePool[r] |= set(path)
                    
                    paths_not_hit_by_high_deg_nodes[r] += 1       
                    
                # check if path weight is unique 
                path_weight = sum([largest_comp[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)])
                if path_weight in set_of_path_weights:
                    exists_repetition = True
                set_of_path_weights.add(path_weight)
            write_and_print(output_filename, f"For radius {r} exists path weight repetition is {exists_repetition}")
                        
            print(f"|pathPool| = {len(pathPool)}, paths_not_hit_by_high_deg_nodes[{r}] = {paths_not_hit_by_high_deg_nodes[r]}")
            
            # compute the hitting set 
            cover[r] = hitting_set(nodePool[r], pathPool) | high_deg_nodes
            write_only(covers_file, f"len(cover[{r}]) = {len(cover[r])}, cover[{r}] = {cover[r]}")
            # high_deg_nodes |= cover[r] # new high_deg_nodes includes prev cover 
            write_and_print(output_filename, f'For radius {r}, the cover has size {len(cover[r])}. Duration for r={r}: {duration_since(last_time)}. paths_hit_by_high_deg_nodes[{r}] = {paths_hit_by_high_deg_nodes[r]}, paths_not_hit_by_high_deg_nodes[{r}] = {paths_not_hit_by_high_deg_nodes[r]}')
            
        write_and_print(output_filename, f"hitting sets computed in {duration_since(start_time)}")
        
        # Krzysztof's heuristic of deleting the high degree nodes from the graph, before taking the intersections of balls and covers
        edgeList = list(largest_comp.edges())
        global largest_comp_no_core
        largest_comp_no_core = nx.DiGraph() # largest_comp is frozen, make copy to allow node deletion 
        largest_comp_no_core.add_edges_from(edgeList)
        # if delete_base:
        largest_comp_no_core.remove_nodes_from(list(high_deg_nodes))
        
        write_and_print(output_filename, f"connected components after removing high degree nodes: {nx.number_strongly_connected_components(largest_comp_no_core)}")
        
        last_time = time.time()  
                 
        # check intersection with balls (parallel)
        print("Computing intersections of balls and covers")
        input_pairs = list(((u,r) for u in largest_comp_no_core.nodes() for r in radii))
        with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
            results = executor.map(ball_intersection_cover, input_pairs)  # results is a generator object 
        print(f"Done computing intersections of balls and covers in {duration_since(last_time)}")
        
        all_ball_intersections = [result for result in results]  # unpack generator 
        
        # compute HD 
        HD = len(high_deg_nodes) + max([len(result[2]) for result in all_ball_intersections]) # HD equals max_{u,r} size(cover[r] intersection ball(u,r))
        core_to_HD[num_of_high_deg_nodes] = HD
    
        write_and_print(output_filename, f"current HD = {HD}. core_to_HD: {core_to_HD}")
    
        if not last_iteration:
            num_of_high_deg_nodes, opt_core_size_found = compute_next_core_size(core_to_HD, num_of_high_deg_nodes)
            
    # save core_to_HD to json
    with open('../results/core-to-HD' + str(date) + '.json', 'w') as handle:
        json.dump(core_to_HD, handle)
    
    # compute hubs  
    hub_dict = {}
    for node in largest_comp.nodes():
        hub_dict[node] = set()
    
    for result in all_ball_intersections:
        (u, r, intersection) = result
        hub_dict[u] |= intersection
    
    # store hub_disct in a JSON file
    hubs_file_name = f"../results/lightning" + str(date) + "-large-component-hubs.JSON"
    
    # convert to str to store in JSON
    hub_dict_str = {}
    for node in hub_dict:
        hub_dict_str[str(node)] = str(hub_dict[node])  
    
    # add high_deg_nodes to hub_dict_str
    hub_dict_str["high_deg_nodes"] = str(high_deg_nodes)
    
    # add node_int_to_pkey to hub_dict_str
    hub_dict_str['node_int_to_pkey'] = node_int_to_pkey 
    
    # usage: all nodes share the high_deg_nodes as part of their hub set. 
    # To compute the hubs of a node u take the union (high_deg_nodes | hub_dict[u]) 
    with open(hubs_file_name, 'w') as outfile:
        json.dump(hub_dict_str, outfile)
    
    max_hub_size = len(high_deg_nodes) + max([len(hub_dict[node]) for node in hub_dict])  # hub_dict = {u:hubs(u), for nodes u}
    write_and_print(output_filename, f"max_hub_size = {max_hub_size}")
    
    # plot_hub_size_distribution(hub_dict)
       
    # export output
    write_and_print(output_filename, f"HD for the large component = {HD} (base = {num_of_high_deg_nodes}), computed in {duration_since(start_time)}")
    
    return HD, [len(hub_dict[u]) for u in hub_dict]
    
version = "version Jan-20, 22:50"
print(f"lightning-HD-factorized.py, {version}")

# dates of LN snapshots
dates_of_datasets = [13032019, 13072019, 13092019, 13112019, 13032020, 13052020, 13072020, 13092020, 13112020, 13012021]
# start_date_index = int(input('Select start_date_index: '))
# end_date_index = int(input('Select end_date_index: '))
# test_selection_of_dates = dates_of_datasets[start_date_index:end_date_index+1] 

date_batch0 = [dates_of_datasets[i] for i in range(len(dates_of_datasets)) if i%2 == 0]
date_batch1 = [dates_of_datasets[i] for i in range(len(dates_of_datasets)) if i%2 == 1]

batch = int(input('Input batch: ')) # @Michelle: please insert 0 here, as file names use this variable 
if batch == 0:
    original_dataset = date_batch0
elif batch == 1:
    original_dataset = date_batch1 

# what remains of batch1
print(f"dates in selected batch (batch {batch}): {original_dataset}")
dataset = ast.literal_eval(input('Insert a list of dates of LN snapshots (e.g. [13032019, 13072019, 13092019] ) : '))

results = {}

for date in dataset:
    print(f'Starting computation for LN snapshot on {date}')
    results[date] = {}
    results[date]['db_size'] = 0  # add manually
    results[date]['HD'], results[date]['hub_sizes'] = lightningHD(date) 

# make box plot
hub_size_data = {date:results[date]['hub_sizes'] for date in dataset}

# df = pd.DataFrame(data=hub_size_data)
df = pd.DataFrame({ key:pd.Series(value) for key, value in hub_size_data.items() })
myFig = plt.figure();
df.plot.box(grid='True')
plt.savefig("../results/boxplot-hub-distr-batch"+str(batch)+".pdf", format="pdf") 