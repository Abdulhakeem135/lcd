#********************************************************************************************************************************************************
from random import choice
import networkx as nx
#================================================================================
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
    # Create layers and  clusters
    clusters = lcd_clusters(G, start_node)
    #print("No of Cluster LCDRank: ", len(clusters))

    #------------------------------------------------------
    # Degree computation and Ranking
    RankingList = find_lcd_ranking(G, clusters, topk)


    return RankingList