import networkx as nx
from algorithms import *
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle


r = [ 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.045] # Email CEnew, Router, Hamster, blogs

#r = [0.0020, 0.0025, 0.0030, 0.0035,0.0040, 0.0045, 0.0050] #for Brightkite, Condmat, git, astroph, Facebook


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
#--------------------------------------------------------------------------
#**************************************************************************
topk_list = []
for k in r:
    topk = round(nx.number_of_nodes(G) * k)
    print(k, topk)
    topk_list.append(topk)

#--------------------------------------------------------------------------
#**************************************************************************
max_ = r[-1]
max_topk = round(max_ * nx.number_of_nodes(G))

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
#--------------------------------------------------------------------------
#**************************************************************************
degree_ls = []
cluster_ls = []
voterank_ls = []
voterank_p_ls = []
EnRenew_ls = []
kshell_ls = []
hindex_ls = []
lcd_ls=[]
    
for k in tqdm(topk_list):
    topk = k
    degree_ls.append(get_ls(G, [x[0] for x in degree_rank[:topk]]))
    cluster_ls.append(get_ls(G, [x[0] for x in cluster_rank[:topk]]))
    voterank_ls.append(get_ls(G, [x[0] for x in vote_rank[:topk]]))
    voterank_p_ls.append(get_ls(G, [x[0] for x in vote_rank_p[:topk]]))
    EnRenew_ls.append(get_ls(G, [x[0] for x in EnRenew_rank[:topk]]))
    kshell_ls.append(get_ls(G, [x[0] for x in kshell_rank[:topk]]))
    hindex_ls.append(get_ls(G, [x[0] for x in hindex_rank[:topk]]))
    lcd_ls.append(get_ls(G, [x[0] for x in lcd_rank[:topk]]))


#--------------------------------------------------------------------------
#**************************************************************************
mark_every=1

plt.plot(np.array(topk_list) / nx.number_of_nodes(G), degree_ls, 'r-o', label='DegreeRank', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(topk_list) / nx.number_of_nodes(G), cluster_ls, 'k-o', label='ClusterRank', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(topk_list) / nx.number_of_nodes(G), voterank_ls, 'c-o', label='voteRank', linewidth=1.0, markersize=4, markevery=mark_every )
plt.plot(np.array(topk_list) / nx.number_of_nodes(G), voterank_p_ls, label='voteRank++', color='brown', linestyle='-', marker='o', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(topk_list) / nx.number_of_nodes(G), EnRenew_ls, 'b-o', label='EnRenew', linewidth=1.0,markersize=4, markevery=mark_every )
plt.plot(np.array(topk_list) / nx.number_of_nodes(G), kshell_ls, 'm-o', label='kshell', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(topk_list) / nx.number_of_nodes(G), hindex_ls, 'g-o', label='h-index', linewidth=1.0, markersize=4, markevery=mark_every)
plt.plot(np.array(topk_list) / nx.number_of_nodes(G), lcd_ls, 'y-o', label='LCD', linewidth=1.0, markersize=4, markevery=mark_every)

plt.xlabel(r"$p$", fontdict={'fontsize': 14})
plt.ylabel(r"$L_s$", fontdict={'fontsize': 14})
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

plt.savefig(graphName+"_ls"+".jpg", dpi=600, bbox_inches='tight')
plt.show()

#--------------------------------------------------------------------------
#**************************************************************************
