# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 16:21:59 2017

@author: Jayant Singh
"""
import pandas as pd
import networkx as nx
import scipy as sp
import numpy as np
from sklearn.model_selection import KFold
import random
import csv

# file & algo name list

link_files = ['pp_interactions_YST.csv', 'stay_away_INF.csv',
              'nn_roundworm_CEL.csv', 'airport_network_USA.csv',
              'hamster_social_network_HMT.csv', 'copurchase_amazon_BCK.csv',
              'dblp_coauthorship_CNS.csv']
algo_list = ['CN', 'AA', 'RA', 'RA-CNI', 'PA',	'JA',	'SA',	'SO',	'HPI',	'HDP',
             'LLHN', 'IA1', 'IA2', 'MI', 'B-CN', 'LNB-AA', 'LNB-RA',	'CAR-CN',
             'R-AA', 'CAR-RA', 'FSW', 'LIT', 'NSP', 'GLHN', 'RW', 'RWR', 'FP',
             'MERW', 'SR', 'PLM', 'ACT', 'RFK', 'BI', 'LPI', 'LRW', 'SRW',
             'ORA-CNI', 'FL', 'PFP']
results_list = []
# reading links csv

'''
--- alternate method for reading link file ---
gtemp = nx.read_edgelist('pp_interactions_YST.csv',
                         create_using=nx.DiGraph(), nodetype=int)
'''
dataset = 'copurchase_amazon_BCK.csv'
links = pd.read_csv(dataset)
G = nx.from_pandas_dataframe(links, 'from_node_id', 'to_node_id')
print ''
print 'DATASET being used: \'%s\'' % dataset
print ''
# summary of the graph (before preprocessing)

'''
--- alternate method for graph summary ---
summary = nx.info(G)
print summary
'''
print 'Number of nodes (before preprocessing) = %d' % len(G.nodes())
print 'Number of edges (before preprocessing) = %d' % len(G.edges())
deg_list = nx.degree(G, G.nodes()).values()
avg_degree = sum(deg_list)/float(len(deg_list))
print 'Average degree (before preprocessing) = %f' % avg_degree

# self loops removal

print 'Number of self-loops = %d' % len(G.selfloop_edges())
G.remove_edges_from(G.selfloop_edges())
print 'Self-loops removed!'

# isolated nodes removal

print 'Number of isolated nodes = %d' % len(nx.isolates(G))
G.remove_nodes_from(nx.isolates(G))
print 'Isolated nodes removed!'

# multiple parallel edges test

no_of_nodes_with_multiP_edges = np.count_nonzero(
        nx.adjacency_matrix(G).data > 1)/2
if (no_of_nodes_with_multiP_edges == 0):
    print 'No multiple parallel edges in the graph.'
else:
    print 'Multiple parallel edges found in the graph!'
    print 'Number of nodes with multiple parallel edges = %d' % no_of_nodes_with_multiP_edges

# summary of the graph (after preprocessing)

print 'Number of nodes (after preprocessing) = %d' % len(G.nodes())
print 'Number of edges (after preprocessing) = %d' % len(G.edges())
deg_list = nx.degree(G, G.nodes()).values()
avg_degree = sum(deg_list)/float(len(deg_list))
print 'Average degree (after preprocessing) = %f' % avg_degree

# list of edges & non_edges

edges = G.edges()
non_edges = list(nx.non_edges(G))

# AUROC calculating function


def ROCevaluator(LP_function):
    secure_random = random.SystemRandom()
    iterator = int(100000)  # iterator = int(0.1 * len(edges))
    n_greater = 0  # no. of times edge score is > than non-edge score
    n_equal = 0  # no. of times edge score is = than non-edge score
    n_less = 0  # no. of times edge score is < than non-edge score
    auc = 0.0  # AUROC score value
    for i in range(iterator):
        random_edge = secure_random.choice(edges)
        random_non_edge = secure_random.choice(non_edges)
        itr1 = LP_function(G, [random_edge])
        itr2 = LP_function(G, [random_non_edge])
        for j in itr1:
            temp1 = j
        score_edge = temp1[2]
        for k in itr2:
            temp2 = k
        score_non_edge = temp2[2]
        if score_edge > score_non_edge:
            n_greater += 1
        elif score_edge == score_non_edge:
            n_equal += 1
        else:
            n_less += 1
    auc = (n_greater + 0.5 * n_equal)/iterator  # print 'AUC = %f' % auc
    return auc

# Common Neighbors (CN) implementation


def common_neighbors(G, ebunch=None):
    u, v = ebunch[0][0], ebunch[0][1]
    if u not in G:
        raise nx.NetworkXError('u is not in the graph.')
    if v not in G:
        raise nx.NetworkXError('v is not in the graph.')
    n = len(list(w for w in G[u] if w in G[v] and w not in (u, v)))  # ???
    return ((u, v, n) for u, v in ebunch)  # ???
roc_CN = ROCevaluator(common_neighbors)
results_list.append(roc_CN)
print 'AUC for Common Neighbors (CN) = %f' % roc_CN

# Adamic-Adar Index (AA) implementation

roc_AA = ROCevaluator(nx.adamic_adar_index)
results_list.append(roc_AA)
print 'AUC for Adamic-Adar Index (AA) = %f' % roc_AA

# Resource Allocation Index (RA) implementation

roc_RA = ROCevaluator(nx.resource_allocation_index)
results_list.append(roc_RA)
print 'AUC for Resource Allocation Index (RA) = %f' % roc_RA

# Resource Allocation Based on Common Neighbor Interactions (RA-CNI) 


# Preferential Attachment Index (PA) implementation

roc_PA = ROCevaluator(nx.preferential_attachment)
results_list.append(roc_PA)
print 'AUC for Preferential Attachment Index (PA) = %f' % roc_PA

# Jaccard Index (JA) implementation

roc_JA = ROCevaluator(nx.jaccard_coefficient)
results_list.append(roc_JA)
print 'AUC for Jaccard Index (JA) = %f' % roc_JA

# Salton Index (SA) implementation


def salton_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u, v = ebunch[0][0], ebunch[0][1]
    cn = common_neighbors(G, ebunch)
    for i in cn:
        temp = i
    CN = temp[2]
    si = CN/np.sqrt(len(list(
            nx.all_neighbors(G, u))) * len(list(nx.all_neighbors(G, v))))
    '''
    def predict(u, v):
        cnbors = list(common_neighbors(G, ebunch))
        union_size = len(set(G[u]) | set(G[v]))
        if union_size == 0:
            return 0
        else:
            return len(cnbors) / union_size
    '''
    return ((u, v, si) for u, v in ebunch)
roc_SA = ROCevaluator(salton_index)
results_list.append(roc_SA)
print 'AUC for Salton Index (SA) = %f' % roc_SA

# Sørensen Index (SO) implementation


def sorensen_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u, v = ebunch[0][0], ebunch[0][1]
    cn = common_neighbors(G, ebunch)
    for i in cn:
        temp = i
    CN = temp[2]
    soi = 2 * CN/(len(list(
            nx.all_neighbors(G, u))) + len(list(nx.all_neighbors(G, v))))
    return ((u, v, soi) for u, v in ebunch)
roc_SI = ROCevaluator(salton_index)
results_list.append(roc_SI)
print 'AUC for Sørensen Index (SO) = %f' % roc_SI

# Hub Promoted Index (HPI) implementation


def hub_promoted_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u, v = ebunch[0][0], ebunch[0][1]
    cn = common_neighbors(G, ebunch)
    for i in cn:
        temp = i
    CN = temp[2]
    hpi = CN/min(len(list(
            nx.all_neighbors(G, u))), len(list(nx.all_neighbors(G, v))))
    return ((u, v, hpi) for u, v in ebunch)
roc_HPI = ROCevaluator(hub_promoted_index)
results_list.append(roc_HPI)
print 'AUC for Hub Promoted Index (HPI) = %f' % roc_HPI

# Hub Depressed Index (HDI) implementation


def hub_depressed_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u, v = ebunch[0][0], ebunch[0][1]
    cn = common_neighbors(G, ebunch)
    for i in cn:
        temp = i
    CN = temp[2]
    hpi = CN/max(len(list(
            nx.all_neighbors(G, u))), len(list(nx.all_neighbors(G, v))))
    return ((u, v, hpi) for u, v in ebunch)
roc_HDI = ROCevaluator(hub_depressed_index)
results_list.append(roc_HDI)
print 'AUC for Hub Depressed Index (HDI) = %f' % roc_HDI

# Local Leicht-Holme-Newman Index (LLHN) implementation


def local_lhn_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u, v = ebunch[0][0], ebunch[0][1]
    cn = common_neighbors(G, ebunch)
    for i in cn:
        temp = i
    CN = temp[2]
    llhn = CN/(len(list(
            nx.all_neighbors(G, u))) * len(list(nx.all_neighbors(G, v))))
    return ((u, v, llhn) for u, v in ebunch)
roc_LLHN = ROCevaluator(local_lhn_index)
results_list.append(roc_LLHN)
print 'AUC for Local Leicht-Holme-Newman Index (LLHN) = %f' % roc_LLHN

with open("results.csv", 'wb') as myfile:
    out = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_ALL) 
    out.writerow(results_list)
    
