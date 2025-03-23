#Copyright (c) 2019 Chungu Guo. All rights reserved.
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy
import random
import math
from collections import defaultdict
from random import choice
from heapq import nlargest


#--------------------------------------------------------------------------
#**************************************************************************
# Define a function to record time and write it to a file
def record_time(graphName, algorithm_name, start_time, end_time):
    elapsed_time = end_time - start_time
    with open(graphName, "a") as f:  # Append mode to avoid overwriting
        f.write(f"{algorithm_name}:\t {elapsed_time:.6f} seconds\n")

# Record the top-k nodes
def record_topk(graphName, algorithm_name, topk):
    with open(graphName, "a") as f:  # Append mode to avoid overwriting
        f.write(f"{algorithm_name}:\t {topk} \n")
#--------------------------------------------------------------------------
#************************************************************************** 
def get_topk(result, topk):
    """return the topk nodes
    # Arguments
        result: a list of result, [(node1, centrality), (node1, centrality), ...]
        topk: how much node will be returned

    Returns
        topk nodes as a list, [node1, node2, ...]
    """
    result_topk = []
    for i in range(topk):
        result_topk.append(result[i][0])
    return result_topk
#--------------------------------------
def get_beta_c(G):
    N=len(G.nodes())
    k = sum([G.degree(i) for i in G.nodes()])/N
    square_k = sum([G.degree(i)**2 for i in G.nodes()])/N

    return k/(square_k-k) 
#--------------------------------------
def get_ls(g, infeacted_set):
    """compute the average shortest path in the initial node set
     # Arguments
         g: a graph as networkx Graph
         infeacted_set: the initial node set
     Returns
         return the average shortest path
     """
    dis_sum = 0
    path_num = 0
    for u in infeacted_set:
        for v in infeacted_set:
            if u != v:
                try:
                    dis_sum += nx.shortest_path_length(g, u, v)
                    path_num += 1
                except:
                    dis_sum += 0
                    path_num -= 1
    return dis_sum / path_num
#--------------------------------------------------------------------------
#************************************************************************** 

def get_sir_result(G, rank, topk, avg, infect_prob, cover_prob, max_iter, metric='ft'):
    """perform SIR simulation
    # Arguments
        G: a graph as networkx Graph
        rank: the initial node set to simulate, [(node1, centrality), (node1, centrality), ...]
        topk: use the topk nodes in rank to simulate
        avg: simulation times, multiple simulation to averaging
        infect_prob: the infection probability
        cover_prob: the cover probability,
        max_iter: maximum number of simulation steps

    Returns
        average simulation result, a 1-D array, indicates the scale of the spread of each step
    """
    time_num_dict_list = []
    time_list = []

    for i in range(avg):
        time, time_num_dict = SIR(G, get_topk(rank, topk), infect_prob, cover_prob, max_iter, metric)
        time_num_dict_list.append((list(time_num_dict.values())))
        time_list.append(time)
            
    max_time = max(time_list) + 1
    result_matrix = np.zeros((len(time_num_dict_list), max_time))
    for index, (row, time_num_dict) in enumerate(zip(result_matrix, time_num_dict_list)):
        row[:] = time_num_dict[-1]
        row[0:len(time_num_dict)] = time_num_dict
        result_matrix[index] = row
    return np.mean(result_matrix, axis=0)

#-------------------------------------------

def SIR(g, infeacted_set, infect_prob, cover_prob, max_iter, metric):
    """Perform once simulation
    # Arguments
        g: a graph as networkx Graph
        infeacted_set: the initial node set to simulate, [node1, node2, ...]
        infect_prob: the infection probability
        cover_prob: the cover probability,
        max_iter: maximum number of simulation steps
    Returns
        time: the max time step in this simulation
        time_count_dict: record the scale of infection at each step, {1:5, 2:20, ..., time: scale, ...}
    """
    time = 0
    time_count_dict = {}
    time_count_dict[time] = len(infeacted_set)
    # infeacted_set = infeacted_set
    node_state = {}
    covered_set = set()

    for node in nx.nodes(g):
        if node in infeacted_set:
            node_state[node] = 'i'
        else:
            node_state[node] = 's'

    while len(infeacted_set) != 0 and max_iter != 0:
        ready_to_cover = []
        ready_to_infeact = []
        for node in infeacted_set:
            nbrs = list(nx.neighbors(g, node))
            nbr = np.random.choice(nbrs)
            if random.uniform(0, 1) <= infect_prob and node_state[nbr] == 's':
                node_state[nbr] = 'i'
                ready_to_infeact.append(nbr)
            if random.uniform(0, 1) <= cover_prob:
                ready_to_cover.append(node)
        for node in ready_to_cover:
            node_state[node] = 'r'
            infeacted_set.remove(node)
            covered_set.add(node)
        for node in ready_to_infeact:
            infeacted_set.append(node)
        max_iter -= 1
        time += 1
        if metric=='ft':
            time_count_dict[time] = len(covered_set) + len(infeacted_set)
        else:
            time_count_dict[time] = len(covered_set)

    return time, time_count_dict
#********************************************************************************
#================================================================================
#DegreeRank method
def degree(g, topk):
    """use the degree to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by degree, [(node1, ' '), (node2, '' ), ...]
     """
    degree_rank = nx.degree_centrality(g)
    degree_rank = sorted(degree_rank.items(), key=lambda x: x[1], reverse=True)
    rank1 = []
    for node, score in degree_rank:
        rank1.append(node)
        if len(rank1) == topk:
            for i in range(len(rank1)):
                rank1[i] = (rank1[i], ' ')
            return rank1 
    return rank1

#********************************************************************************
#================================================================================
#VoteRank method
def voterank(G, topk):
    """use the voterank to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by voterank, [(node1, score), (node2, score ), ...]
     """

    k_ = 1 / (nx.number_of_edges(G) * 2 / nx.number_of_nodes(G))
    result_rank = []
    voting_ability = {}

    node_scores = {}
    for node in nx.nodes(G):
        voting_ability[node] = 1
        node_scores[node] = G.degree(node)

    for i in range(topk):
        selected_node, score = max(node_scores.items(), key=lambda x: x[1])
        result_rank.append((selected_node, score))

        weaky = voting_ability[selected_node]
        node_scores.pop(selected_node)
        voting_ability[selected_node] = 0

        for nbr in nx.neighbors(G, selected_node):
            weaky2 = k_
            voting_ability[nbr] -= k_
            if voting_ability[nbr] < 0:
                weaky2 = abs(voting_ability[nbr])
                voting_ability[nbr] = 0
            if nbr in node_scores:
                node_scores[nbr] -= weaky
            for nbr2 in nx.neighbors(G, nbr):
                if nbr2 in node_scores:
                    node_scores[nbr2] -= weaky2
    return result_rank

#********************************************************************************
#================================================================================
#k-shell method
def kshell(G, topk):
    """use the kshell to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by kshell, [(node1, ' '), (node2, ' '), ...]
     """
    node_core = nx.core_number(G)
    core_node_list = {}
    for node in node_core:
        if node_core[node] not in core_node_list:
            core_node_list[node_core[node]] = []
        core_node_list[node_core[node]].append((node, nx.degree(G, node)))

    for core in core_node_list:
        core_node_list[core] = sorted(core_node_list[core], key=lambda x:x[1], reverse=True)
    core_node_list = sorted(core_node_list.items(), key=lambda x: x[0], reverse=True)
    kshellrank = []
    for core, node_list in core_node_list:
        kshellrank.extend([n[0] for n in node_list])

    rank = []
    for node in kshellrank:
        rank.append((node, ' '))
        if len(rank) == topk:
            return rank

#********************************************************************************
#================================================================================
#EnRenewRank method
def EnRenewRank(G, topk, order):
    """use the our method to get topk nodes
     # Arguments
         g: a graph as networkx Graph
         topk: how many nodes will be returned
     Returns
         return the topk nodes by EnRenewRank, [(node1, score), (node2, score), ...]
     """

    all_degree = nx.number_of_nodes(G) - 1
    k_ = nx.number_of_edges(G) * 2 / nx.number_of_nodes(G)
    k_entropy = - k_ * ((k_ / all_degree) * math.log((k_ / all_degree)))

    # node's information pi
    node_information = {}
    for node in nx.nodes(G):
        information = (G.degree(node) / all_degree)
        node_information[node] = - information * math.log(information)

    # node's entropy Ei
    node_entropy = {}
    for node in nx.nodes(G):
        node_entropy[node] = 0
        for nbr in nx.neighbors(G, node):
            node_entropy[node] += node_information[nbr]

    rank = []
    for i in range(topk):
        # choose the max entropy node
        max_entropy_node, entropy = max(node_entropy.items(), key=lambda x: x[1])
        rank.append((max_entropy_node, entropy))

        cur_nbrs = nx.neighbors(G, max_entropy_node)
        for o in range(order):
            for nbr in cur_nbrs:
                if nbr in node_entropy:
                        node_entropy[nbr] -= (node_information[max_entropy_node] / k_entropy) / (2**o)
            next_nbrs = []
            for node in cur_nbrs:
                nbrs = nx.neighbors(G, node)
                next_nbrs.extend(nbrs)
            cur_nbrs = next_nbrs

        # set the information quantity of selected nodes to 0
        node_information[max_entropy_node] = 0
        # delete max_entropy_node
        node_entropy.pop(max_entropy_node)
    return rank
#********************************************************************************
#================================================================================
#ClusterRank method
def clusterRank(graph, vertices=None, directed=False, verbose=False, top_k=None):
    """
    Calculates the ClusterRank of nodes in a graph using NetworkX.

    Parameters:
    - graph (networkx.Graph): The input graph (can be directed or undirected).
    - vertices (list, optional): A list of desired nodes for which ClusterRank is calculated. 
                                 If None, all nodes are considered.
    - directed (bool): Whether the graph is directed. Default is False.
    - loops (bool): Whether to include self-loops in calculations. Default is True.
    - verbose (bool): Whether to print progress. Default is False.
    - TopK (int, optional): The number of top-ranked nodes to return. If None, all nodes are returned.

    Returns:
    - list: A list of top K nodes sorted by ClusterRank.
    """
    # Handle directed/undirected graph
    #if directed and not isinstance(graph, nx.DiGraph):
    #    graph = nx.DiGraph(graph)
    #elif not directed and not isinstance(graph, nx.Graph):
    #    graph = nx.Graph(graph)

    # Use all nodes if vertices is None
    if vertices is None:
        vertices = list(graph.nodes)

    # Step 1: Calculate the transitivity (local clustering coefficient) for each node
    if verbose:
        print("Calculating local clustering coefficients.")
    transitivity = nx.clustering(graph, nodes=vertices)
    
    # Step 2: Compute first neighbors for each node
    if verbose:
        print("Calculating first neighbors for each node.")
    neighbors = {node: list(graph.neighbors(node)) for node in vertices}

    # Step 3: Calculate the initial ClusterRank value for each node
    if verbose:
        print("Calculating initial ClusterRank values.")
    initial_clusterRank = {}
    for node in vertices:
        rank = 0
        for neighbor in neighbors[node]:
            # Include the degree and add 1 for each neighbor
            degree = graph.degree(neighbor) if not directed else graph.out_degree(neighbor)
            rank += degree + 1
        initial_clusterRank[node] = rank

    # Step 4: Compute final ClusterRank values
    if verbose:
        print("Calculating final ClusterRank values.")
    final_clusterRank = {
        node: initial_clusterRank[node] * transitivity[node]
        for node in vertices
    }

    # Step 5: Sort nodes by ClusterRank value in descending order
    sorted_nodes = sorted(final_clusterRank.items(), key=lambda x: x[1], reverse=True)
    rank = []
    # Step 6: Return only the TopK nodes if specified
    if top_k:
        rank = sorted_nodes[:top_k]

    # Return the sorted nodes with their ClusterRank values
    return rank
#********************************************************************************
#================================================================================
#H-index method
def h_index(G, vertices=None, top_k=None):
    """
    Calculate the H-index for nodes in an undirected NetworkX graph.

    Parameters:
        G (networkx.Graph): The undirected graph.
        vertices (list, optional): A list of nodes for which to calculate the H-index.
                                   If None, calculates for all nodes.
        top_k (int, optional): Return only the top_k nodes based on their H-index values.
        verbose (bool): Whether to print progress messages.

    Returns:
        list: A sorted list of tuples (node, H-index) in descending order of H-index values.
    """
    if vertices is None:
        vertices = G.nodes()

    h_indices = {}

    for node in vertices:
        # Get neighbors of the node
        neighbors = G.neighbors(node)

        # Calculate degrees of neighbors
        neighbor_degrees = sorted([G.degree(n) for n in neighbors], reverse=True)

        # Calculate H-index
        h_index = 0
        for i, degree in enumerate(neighbor_degrees, start=1):
            if degree >= i:
                h_index = i
            else:
                break

        h_indices[node] = h_index

    # Sort nodes by H-index in descending order
    sorted_h_indices = sorted(h_indices.items(), key=lambda x: x[1], reverse=True)

    # Return top-k results if specified
    if top_k is not None:
        return nlargest(top_k, sorted_h_indices, key=lambda x: x[1])

    return sorted_h_indices


#********************************************************************************
#================================================================================
#VoteRank++
def get_weight(G, degree_dic, rank): 
    weight = {}
    nodes = nx.nodes(G)
    rank_list = [i[0] for i in rank]
    for node in nodes:
        sum1 = 0
        neighbors = list(nx.neighbors(G, node))
        neighbors_common_rank = list(set(neighbors) & set(rank_list))
        if len(neighbors_common_rank) != 0:  
            for nc in neighbors_common_rank:
                weight[(node, nc)] = 0
        neighbours_without_rank = list(set(neighbors) - set(rank_list))  # voting for unselected nodes
        if len(neighbours_without_rank) != 0:  # if the node has other nieghbours
            for nbr in neighbours_without_rank:
                sum1 += degree_dic[nbr]
            for neigh in neighbours_without_rank:
                weight[(node, neigh)] = degree_dic[neigh] / sum1
        else:  
            for neigh in neighbors:
                weight[(node, neigh)] = 0
    return weight
#--------------------------------------------------------------------------
def get_node_score(G, nodesNeedcalcu, node_ability, degree_dic, rank):

    weight = get_weight(G, degree_dic, rank)
    node_score = {}
    for node in nodesNeedcalcu:  # for ever node add the neighbor's weighted ability
        sum2 = 0
        neighbors = list(nx.neighbors(G, node))
        for nbr in neighbors:
            sum2 += node_ability[nbr] * weight[(nbr, node)]
        node_score[node] = math.sqrt(len(neighbors) * sum2)
    return node_score


def voterank_plus(G, l=None, lambdaa=0.1):
    '''

    :param G: use new indicator + lambda + voterank, the vote ability = log(dij)
    :param l: the number of spreaders
    :param lambdaa: retard infactor. In the paper, "Here, the suppressing factor λis set to 0.1."
    :return:
    '''

    if l is None or l > len(G):
        l = len(G)

    rank = []

    # count dict
    nodes = list(nx.nodes(G))
    degree_li = nx.degree(G)
    d_max = max([i[1] for i in degree_li])
    degree_dic = {}
    for i in degree_li:
        degree_dic[i[0]] = i[1]

    # node's vote information
    node_ability = {}
    for item in degree_li:
        degree = item[1]
        node_ability[item[0]] = math.log(1 + (degree/d_max))  # ln(x)

    # node_ability_values = node_ability.values()
    # degree_values = degree_dic.values()
    # weaky = mean_value(node_ability_values) / mean_value(degree_values)
    # node's score
    node_score = get_node_score(G, nodes, node_ability, degree_dic, rank)

    for i in range(l):
        # choose the max entropy node for the first time t aviod the error
        max_score_node, score = max(node_score.items(), key=lambda x: x[1])
        rank.append((max_score_node, score))
        # set the information quantity of selected nodes to 0
        node_ability[max_score_node] = 0
        # set entropy to 0
        node_score.pop(max_score_node)
        # for the max score node's neighbor conduct a neighbour ability surpassing
        cur_nbrs = list(nx.neighbors(G, rank[-1][0]))  # spreader's neighbour 1 th neighbors
        next_cur_neigh = []  # spreader's neighbour's neighbour 2 th neighbors
        for nbr in cur_nbrs:
            nnbr = nx.neighbors(G, nbr)
            next_cur_neigh.extend(nnbr)
            node_ability[nbr] *= lambdaa  # suppress the 1th neighbors' voting ability

        next_cur_neighs = list(set(next_cur_neigh))  # delete the spreaders and the 1th neighbors
        for ih in rank:
            if ih[0] in next_cur_neighs:
                next_cur_neighs.remove(ih[0])
        for i in cur_nbrs:
            if i in next_cur_neighs:
                next_cur_neighs.remove(i)

        for nnbr in next_cur_neighs:
            node_ability[nnbr] *= (lambdaa ** 0.5)  # suppress 2_th neighbors' voting ability
        # find the neighbor and neighbor's neighbor
        H = []
        H.extend(cur_nbrs)
        H.extend(next_cur_neighs)
        for nbr in next_cur_neighs:
            nbrs = nx.neighbors(G, nbr)
            H.extend(nbrs)

        H = list(set(H))
        for ih in rank:
            if ih[0] in H:
                H.remove(ih[0])
        new_nodeScore = get_node_score(G, H, node_ability, degree_dic, rank)
        node_score.update(new_nodeScore)

    return rank

#********************************************************************************
#================================================================================
#LCD method
def lcd_clusters(G, start_node):
    """
    Partition vertices into layers and clusters, returning a list of all clusters.
    :param G: A NetworkX graph
    :param start_node: The starting vertex for layering
    :return: List of all clusters
    """
    # Compute eccentricity and initialize layers
    eccentricity = nx.eccentricity(G, v=start_node)
    layers = {i: [] for i in range(eccentricity + 1)}

    # Calculate layers
    for node in G.nodes():
        dist = nx.shortest_path_length(G, source=start_node, target=node)
        layers[dist].append(node)

    # Clustering layers
    all_clusters = []
    for i, layer in layers.items():
        if not layer:
            continue

        # Create a subgraph containing only nodes in the current and higher layers
        valid_nodes = set(node for l in range(i, eccentricity + 1) for node in layers[l])
        subgraph = G.subgraph(valid_nodes)

        # Find clusters within the current layer using connectivity in the subgraph
        visited = set()
        for node in layer:
            if node not in visited:
                # Perform BFS restricted to valid nodes
                cluster = set()
                queue = [node]
                while queue:
                    current = queue.pop()
                    if current not in visited:
                        visited.add(current)
                        if current in layer:  # Only include nodes from the current layer
                            cluster.add(current)
                        # Add all neighbors from the subgraph to the queue
                        queue.extend(set(subgraph.neighbors(current)))
                all_clusters.append(cluster)

    return all_clusters

def find_lcd_ranking(graph, clusters, topk):
    """
    Find the top-k influential nodes using the LCD ranking process.
    :param graph: A NetworkX graph
    :param clusters: List of clusters (sets or lists of nodes)
    :param k: Number of top nodes to return
    :return: List of top-k nodes sorted by degree
    """
    # Initialize ranking list R
    ranking = []

    # Convert each cluster into a sorted list of (node, degree) pairs
    cluster_lists = []
    for cluster in clusters:
        sorted_cluster = sorted([(node, graph.degree(node)) for node in cluster], 
                                key=lambda x: x[1], 
                                reverse=True)
        cluster_lists.append(sorted_cluster)

    while len(ranking) < topk:
        # Initialize a temporary list S to hold the selected nodes in this round
        round_candidates = []

        # Pick the highest degree node from each cluster's list (if available)
        for cluster in cluster_lists:
            if cluster:  # Ensure the cluster still has nodes left
                round_candidates.append(cluster.pop(0))  # Remove the first node

        # Sort round candidates by degree in descending order
        round_candidates.sort(key=lambda x: x[1], reverse=True)

        # Add the selected nodes to the ranking
        ranking.extend([node for node, degree in round_candidates])

    # Ensure the result contains exactly k nodes
    return ranking

def lcdRank(G, topk):
    """
    Compute LCD ranking based on layering, clustering, and degree.
    :param G: A NetworkX graph
    :param topk: Number of top nodes to return
    :return: List of top-k nodes by degree, [(node1, ' '), (node2, ''), ...]
    """
    #------------------------------------------------------
    # Choose a random start node and find the farthest node
    initial_node = choice(list(G.nodes()))
    distances = nx.single_source_shortest_path_length(G, initial_node)
    start_node = max(distances.keys(), key=lambda k: distances[k])

    #------------------------------------------------------
    # Degree Computation and Ranking 
    clusters = lcd_clusters(G, start_node)
    print("No of Cluster LCDRank: ", len(clusters))

    # This needs to be update 
    RankingList = find_lcd_ranking(G, clusters, topk)

    rank1 = []
    for node in RankingList:
        rank1.append(node)
        if len(rank1) == topk:
            for i in range(len(rank1)):
                rank1[i] = (rank1[i], ' ')
            return rank1 
    return rank1



