import networkx as nx
from algorithms import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle


data_file = 'datasets/CEnew'
graphName="CEnew.txt"
G = nx.read_adjlist(data_file)
G.remove_edges_from(nx.selfloop_edges(G))
for node in nx.nodes(G):
    if G.degree(node) == 0:
        G.remove_node(node)

print(nx.number_of_nodes(G), nx.number_of_edges(G))
#--------------------------------------------------------------------------
#**************************************************************************
max_ = 0.03
max_topk = round(max_ * nx.number_of_nodes(G))
print("Topk Len:", str(max_topk))

#--------------------------------------------------------------------------
#**************************************************************************

degree_rank = degree(G, max_topk)
print('\n done! degree_rank')

cluster_rank = clusterRank(G, top_k=max_topk)
print('done! cluster_rank')

vote_rank = voterank(G, max_topk)
print('done! vote_rank')

vote_rank_p = voterank_plus(G, max_topk)
print('done! vote_rank_p')

EnRenew_rank = EnRenewRank(G, max_topk, 2)
print('done! EnRenew_rank' )

kshell_rank = kshell(G, max_topk)
print('done! kshell_rank')

hindex_rank = h_index(G, top_k=max_topk)
print('done! kshell_non')

lcd_rank = lcdRank(G, max_topk)
print('done! lcd_rank')
#--------------------------------------------------------------------------
#**************************************************************************

infect_prob = get_beta_c(G) * 1.5
print("infect_prob: ", infect_prob)
avg = 100
max_iter = 200000
topk = round(max_ * nx.number_of_nodes(G))

#--------------------------------------------------------------------------
#**************************************************************************
degreerank_result = []
clusterrank_result = []
voterank_result = []
voterank_p_result = []
EnRenew_result = []
kshell_result = []
hindex_result = []
lcdRank_result = []

print("SIR started ...")
for a in tqdm(range(5, 21, 3)):
    atio = a / 10
    cover_prob = infect_prob / atio
    print("\n atio: ", atio, "cover_prob: ", cover_prob)
    degreerank_result.append(get_sir_result(G, degree_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ftc')[-1] / nx.number_of_nodes(G))
    clusterrank_result.append(get_sir_result(G, cluster_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ftc')[-1] / nx.number_of_nodes(G))
    voterank_result.append(get_sir_result(G, vote_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ftc')[-1] / nx.number_of_nodes(G))
    voterank_p_result.append(get_sir_result(G, vote_rank_p, topk, avg, infect_prob, cover_prob, max_iter, 'ftc')[-1] / nx.number_of_nodes(G))
    EnRenew_result.append(get_sir_result(G, EnRenew_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ftc')[-1] / nx.number_of_nodes(G))
    kshell_result.append(get_sir_result(G, kshell_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ftc')[-1] / nx.number_of_nodes(G))
    hindex_result.append(get_sir_result(G, hindex_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ftc')[-1] / nx.number_of_nodes(G))
    lcdRank_result.append(get_sir_result(G, lcd_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ftc')[-1] / nx.number_of_nodes(G))

#--------------------------------------------------------------------------
#**************************************************************************
mark_every=1

plt.plot(np.array(range(5, 21, 3))/10, degreerank_result, 'r-o', label='DegreeRank', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(range(5, 21, 3))/10, clusterrank_result, 'k-o', label='ClusterRank', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(range(5, 21, 3))/10, voterank_result, 'c-o', label='voteRank', linewidth=1.0, markersize=4, markevery=mark_every )
plt.plot(np.array(range(5, 21, 3))/10, voterank_p_result, label='voteRank++', color='brown', linestyle='-', marker='o', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(range(5, 21, 3))/10, EnRenew_result, 'b-o', label='EnRenew', linewidth=1.0,markersize=4, markevery=mark_every )
plt.plot(np.array(range(5, 21, 3))/10, kshell_result, 'm-o', label='kshell', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(range(5, 21, 3))/10, hindex_result, 'g-o', label='h-index', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(range(5, 21, 3))/10, lcdRank_result, 'y-o', label='LCD', linewidth=1.0, markersize=4, markevery=mark_every)

plt.xlabel(r"$\lambda$", fontdict={'fontsize': 14})
plt.ylabel(r"$F(t_c)$", fontdict={'fontsize': 14})

plt.text(
    (plt.xlim()[0] + plt.xlim()[1]) / 2,  
    plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),  
    graphName, 
    fontsize=12, 
    color='black', 
    ha='center', 
    va='center', 
    bbox=dict(facecolor='white', alpha=0.6) 
)

plt.legend()

plt.savefig(graphName+"_Ft_c_Lambda_max_"+str(max_)+".jpg", dpi=600, bbox_inches='tight')
plt.show()

