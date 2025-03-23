import networkx as nx
from algorithms import *
import matplotlib.pyplot as plt
import numpy as np



data_file = 'datasets/CEnew'
graphName="CEnew.txt"

print(graphName)
G = nx.read_adjlist(data_file)
G.remove_edges_from(nx.selfloop_edges(G))
for node in nx.nodes(G):
    if G.degree(node) == 0:
        G.remove_node(node)

print(nx.number_of_nodes(G), nx.number_of_edges(G))

max_ = 0.03
max_topk = round(max_ * nx.number_of_nodes(G))
print("Topk Len:", str(max_topk))

#--------------------------------------------------------------------------
#**************************************************************************
degree_rank = degree(G, max_topk)
print('done! degree_rank')

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
print('done! hindex_rank')

lcd_rank = lcdRank(G, max_topk)
print('done! lcd_rank')

#--------------------------------------------------------------------------
#**************************************************************************
infect_prob = round(get_beta_c(G) * 1.5,4)
print("\n  infect_prob: ", infect_prob)
avg = 100
max_iter = 200000
atio = 1.5
cover_prob = infect_prob / atio
topk = max_topk
#--------------------------------------------------------------------------
#**************************************************************************

print("SIR started ...")
degreerank_result = get_sir_result(G, degree_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ft') / nx.number_of_nodes(G)
print('done!')
clusterrank_result = get_sir_result(G, cluster_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ft') / nx.number_of_nodes(G)
print('done!')
voterank_result = get_sir_result(G, vote_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ft') / nx.number_of_nodes(G)
print('done!')
voterank_p_result = get_sir_result(G, vote_rank_p, topk, avg, infect_prob, cover_prob, max_iter, 'ft') / nx.number_of_nodes(G)
print('done!')
EnRenew_result = get_sir_result(G, EnRenew_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ft') / nx.number_of_nodes(G)
print('done!')
kshell_result = get_sir_result(G, kshell_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ft') / nx.number_of_nodes(G)
print('done!')
hindex_result = get_sir_result(G, hindex_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ft') / nx.number_of_nodes(G)
print('done!')
lcd_result = get_sir_result(G, lcd_rank, topk, avg, infect_prob, cover_prob, max_iter, 'ft') / nx.number_of_nodes(G)
print('done!')

#--------------------------------------------------------------------------
#**************************************************************************

max_len = max([ len(degreerank_result),
                len(clusterrank_result),
                len(voterank_result),
                len(voterank_p_result),
                len(EnRenew_result),
                len(kshell_result),
                len(hindex_result),
                len(lcd_result)])


degreerank_array = np.ones(max_len) * degreerank_result[-1]
clusterrank_array = np.ones(max_len) * clusterrank_result[-1]
voterank_array = np.ones(max_len) * voterank_result[-1]
voterank_p_array = np.ones(max_len) * voterank_p_result[-1]
EnRenew_array = np.ones(max_len) * EnRenew_result[-1]
kshell_array = np.ones(max_len) * kshell_result[-1]
hindex_array = np.ones(max_len) * hindex_result[-1]
lcdRank_array = np.ones(max_len) * lcd_result[-1]

degreerank_array[:len(degreerank_result)] = degreerank_result
clusterrank_array[:len(clusterrank_result)] = clusterrank_result
voterank_array[:len(voterank_result)] = voterank_result
voterank_p_array[:len(voterank_p_result)] = voterank_p_result
EnRenew_array[:len(EnRenew_result)] = EnRenew_result
kshell_array[:len(kshell_result)] = kshell_result
hindex_array[:len(hindex_result)] = hindex_result
lcdRank_array[:len(lcd_result)] = lcd_result

#--------------------------------------------------------------------------
#**************************************************************************
mark_every= max(1, max_len // 20)
plt.plot(np.array(range(max_len)), degreerank_array, 'r-o', label='DegreeRank', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(range(max_len)), clusterrank_array, 'k-o', label='ClusterRank', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(range(max_len)), voterank_array, 'c-o', label='voteRank', linewidth=1.0, markersize=4, markevery=mark_every )
plt.plot(np.array(range(max_len)), voterank_p_array, label='voteRank++', color='brown', linestyle='-', marker='o', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(range(max_len)), EnRenew_array, 'b-o', label='EnRenew', linewidth=1.0,markersize=4, markevery=mark_every )
plt.plot(np.array(range(max_len)), kshell_array, 'm-o', label='kshell', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(range(max_len)), hindex_array, 'g-o', label='h-index', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(range(max_len)), lcdRank_array, 'y-o', label='LCD', linewidth=1.0, markersize=4, markevery=mark_every)

plt.xlabel('Time t', fontdict={'fontsize': 14})
plt.ylabel('F(t)', fontdict={'fontsize': 14})
plt.text(
    max_len * 0.5,  
    plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]),  
    graphName, 
    fontsize=12, 
    color='black', 
    ha='center', 
    va='center', 
    bbox=dict(facecolor='white', alpha=0.6) 
)

plt.legend()

plt.savefig(graphName+"_Ft_k"+str(max_)+"_SIR_avg"+str(avg)+".jpg", dpi=600, bbox_inches='tight')
plt.show()