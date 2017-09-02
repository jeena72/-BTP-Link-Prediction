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
from itertools import izip as zip
from networkx.exception import NetworkXError

# file & algo name list

link_files = ['pp_interactions_YST.csv', 'stay_away_INF.csv',
              'nn_roundworm_CEL.csv', 'airport_network_USA.csv',
              'hamster_social_network_HMT.csv', 'copurchase_amazon_BCK.csv',
              'dblp_coauthorship_CNS.csv']
algo_list = ['CN',	'AA',	'RA',	'RA-CNI',	'PA',	'JA',	'SA',	'SO',	'HPI',	'HDP',
             'LLHN', 'IA1', 'IA2',	'MI', 'B-CN', 'LNB-AA', 'LNB-RA',	'CAR-CN',
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
dataset = 'pp_interactions_YST.csv'
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
adj_mat = nx.adjacency_matrix(G)
global counter
counter = 0
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
    global counter
    counter += 1
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
print 'counter value = %d' %counter
counter = 0

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
    global counter
    counter += 1
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
print 'counter value = %d' %counter
counter = 0

# Sørensen Index (SO) implementation


def sorensen_index(G, ebunch=None):
    global counter
    counter += 1
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
print 'counter value = %d' %counter
counter = 0

# Hub Promoted Index (HPI) implementation


def hub_promoted_index(G, ebunch=None):
    global counter
    counter += 1
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
print 'counter value = %d' %counter
counter = 0

# Hub Depressed Index (HDI) implementation


def hub_depressed_index(G, ebunch=None):
    global counter
    counter += 1
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
print 'counter value = %d' %counter
counter = 0

# Local Leicht-Holme-Newman Index (LLHN) implementation


def local_lhn_index(G, ebunch=None):
    global counter
    counter += 1
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
print 'counter value = %d' %counter
counter = 0

# Katz Index (KI) implementation


def katz_index(G, ebunch=None):
    global counter
    counter += 1
    key_dict = {k: v for k, v in zip(list(G.nodes()), list(range(G.order())))}
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u, v = ebunch[0][0], ebunch[0][1]
    beta = 0.001  # beta(attenuation factor) value can be changed
    identity_matrix = np.identity(nx.number_of_nodes(G))
    S = np.linalg.inv(identity_matrix -
                      beta * adj_mat.todense()) - identity_matrix
    ki = S[key_dict[u], key_dict[v]]
    return ((u, v, ki) for u, v in ebunch)

roc_KI = ROCevaluator(katz_index)
results_list.append(roc_KI)
print 'AUC for Katz Index (KI) = %f' % roc_KI
print 'counter value = %d' %counter
counter = 0

# Random Walk (RW) implementation


def random_walk(G, ebunch=None):
    global counter
    counter += 1
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u = ebunch[0][0]
    v = ebunch[0][1]
    tol = 1.0e-5  # can be modified
    M = nx.adjacency_matrix(G)

    # converting adjacency matrix to transition matrix
    for i in range(M.shape[1]):
        if(np.sum(M[i]) > 0):
            M[i] = M[i]/np.sum(M[i])

    # defining start vector
    key_dict = {k: v for k, v in zip(list(G.nodes()), list(range(G.order())))}
    p_o = [0] * G.order()
    p_o[key_dict[u]] = 1
    p_i = p_o

    # iterations to generate probability vector
    for j in range(100):  # can be modified
        p_f = M.T * p_i
        err = sum([abs(p_f[n]-p_i[n]) for n in p_i])
        if err < tol:
            break
        p_i = p_f

    # results generation
    rw_score = p_f[key_dict[v]]
    return ((u, v, rw_score) for u, v in ebunch)

roc_RW = ROCevaluator(random_walk)
results_list.append(roc_RW)
print 'AUC for Random Walks (RW) = %f' % roc_RW
print 'counter value = %d' %counter
counter = 0

# Random Walks with Restart (RWR) implementation

counter = 0
def random_walk_restart(G, ebunch=None):
    global counter
    counter += 1
    global counter 
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u = ebunch[0][0]
    v = ebunch[0][1]
    tol = 1.0e-5  # can be modified
    alpha = 0.85  # can be modified
    M = nx.adjacency_matrix(G)
    counter += 1

    # converting adjacency matrix to transition matrix
    for i in range(M.shape[1]):
        if(np.sum(M[i]) > 0):
            M[i] = M[i]/np.sum(M[i])

    # defining start vector
    key_dict = {k: v for k, v in zip(list(G.nodes()), list(range(G.order())))}
    pu_o = [0] * G.order()
    su = [0] * G.order()
    su[key_dict[u]] = 1
    # pv_o[key_dict[u]] = 1
    pu_i = pu_o
    pv_o = [0] * G.order()
    sv = [0] * G.order()
    sv[key_dict[v]] = 1
    # pv_o[key_dict[v]] = 1
    pv_i = pv_o

    # iterations to generate probability vector
    iterator = 100  # can be modified
    for j in range(iterator):
        pu_f = np.multiply(alpha, M.T * pu_i) + np.multiply((1 - alpha), su)
        err = sum([abs(pu_f[n]-pu_i[n]) for n in pu_i])
        if err < tol:
            break
        pu_i = pu_f

    for k in range(iterator):
        pv_f = np.multiply(alpha, M.T * pv_i) + np.multiply((1 - alpha), sv)
        err = sum([abs(pv_f[n]-pv_i[n]) for n in pv_i])
        if err < tol:
            break
        pv_i = pv_f

    # results generation
    rwr_score = pu_f[key_dict[v]] + pv_f[key_dict[u]]
    return ((u, v, rwr_score) for u, v in ebunch)

roc_RWR = ROCevaluator(random_walk_restart)
results_list.append(roc_RWR)
print 'AUC for Random Walks with Restart (RWR) = %f' % roc_RWR
print 'counter value = %d' % counter
counter = 0

# Local Random Walks (LRW) implementation

counter = 0
def local_random_walk(G, ebunch=None):
    global counter
    counter += 1
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u = ebunch[0][0]
    v = ebunch[0][1]
    alpha = 0.85  # can be modified
    M = nx.adjacency_matrix(G)
    counter += 1

    # converting adjacency matrix to transition matrix
    for i in range(M.shape[1]):
        if(np.sum(M[i]) > 0):
            M[i] = M[i]/np.sum(M[i])

    # defining start vector
    key_dict = {k: v for k, v in zip(list(G.nodes()), list(range(G.order())))}
    pu_o = [0] * G.order()
    su = [0] * G.order()
    su[key_dict[u]] = 1
    # pv_o[key_dict[u]] = 1
    pu_i = pu_o
    pv_o = [0] * G.order()
    sv = [0] * G.order()
    sv[key_dict[v]] = 1
    # pv_o[key_dict[v]] = 1
    pv_i = pv_o

    # iterations to generate probability vector
    iterator = 10  # can be modified IMPORTANT !!!!
    for j in range(iterator):
        pu_f = np.multiply(alpha, M.T * pu_i) + np.multiply((1 - alpha), su)
        pu_i = pu_f

    for k in range(iterator):
        pv_f = np.multiply(alpha, M.T * pv_i) + np.multiply((1 - alpha), sv)
        pv_i = pv_f

    # results generation
    lrw_score = (len(G[u]) * pu_f[key_dict[v]] + len(G[v]) * pv_f[key_dict[u]])/(2 * G.size())
    return ((u, v, lrw_score) for u, v in ebunch)

roc_LRW= ROCevaluator(local_random_walk)
results_list.append(roc_LRW)
print 'AUC for Local Random Walks (LRW) = %f' % roc_LRW
print 'counter value = %d' %counter
counter = 0

# Superposed Random Walks (SRW) implementation

counter = 0
def superposed_random_walk(G, ebunch=None):
    global counter
    counter += 1
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u = ebunch[0][0]
    v = ebunch[0][1]
    alpha = 0.85  # can be modified
    M = nx.adjacency_matrix(G)
    counter += 1
    score_collection = 0

    # converting adjacency matrix to transition matrix
    for i in range(M.shape[1]):
        if(np.sum(M[i]) > 0):
            M[i] = M[i]/np.sum(M[i])

    # defining start vector
    key_dict = {k: v for k, v in zip(list(G.nodes()), list(range(G.order())))}
    pu_o = [0] * G.order()
    su = [0] * G.order()
    su[key_dict[u]] = 1
    # pv_o[key_dict[u]] = 1
    pu_i = pu_o
    pv_o = [0] * G.order()
    sv = [0] * G.order()
    sv[key_dict[v]] = 1
    # pv_o[key_dict[v]] = 1
    pv_i = pv_o

    # iterations to generate probability vector
    iterator = 10  # can be modified IMPORTANT !!!!
    for j in range(iterator):
        pu_f = np.multiply(alpha, M.T * pu_i) + np.multiply((1 - alpha), su)
        pu_i = pu_f
        pv_f = np.multiply(alpha, M.T * pv_i) + np.multiply((1 - alpha), sv)
        pv_i = pv_f
        score_collection += (len(G[u]) * pu_f[key_dict[v]] +
                             len(G[v]) * pv_f[key_dict[u]])/(2 * G.size())
    # results generation
    lrw_score = score_collection
    return ((u, v, lrw_score) for u, v in ebunch)

roc_SRW = ROCevaluator(superposed_random_walk)
results_list.append(roc_SRW)
print 'AUC for Superposed Random Walks (SRW) = %f' % roc_SRW
print 'counter value = %d' % counter
counter = 0

# PropFlow Predictor (PFP) implementation

counter = 0
def propflow_predictor(G, ebunch=None):
    global counter
    counter += 1
    if ebunch is None:
        ebunch = nx.non_edges(G)
    u = ebunch[0][0]
    v = ebunch[0][1]
    l = 10 # can be modified
    '''
    alpha = 0.85  # can be modified
    M = nx.adjacency_matrix(G)
    counter += 1
    score_collection = 0

    # converting adjacency matrix to transition matrix
    for i in range(M.shape[1]):
        if(np.sum(M[i]) > 0):
            M[i] = M[i]/np.sum(M[i])
    '''
    # defining start vector
    key_dict = {k: v for k, v in zip(list(G.nodes()), list(range(G.order())))}
    s = [0] * G.order()
    s[key_dict[u]] = 1
    # pv_o[key_dict[u]] = 1
    pu_i = pu_o
    pv_o = [0] * G.order()
    sv = [0] * G.order()
    sv[key_dict[v]] = 1
    # pv_o[key_dict[v]] = 1
    pv_i = pv_o

    # iterations to generate probability vector
    iterator = 10  # can be modified IMPORTANT !!!!
    for j in range(iterator):
        pu_f = np.multiply(alpha, M.T * pu_i) + np.multiply((1 - alpha), su)
        pu_i = pu_f
        pv_f = np.multiply(alpha, M.T * pv_i) + np.multiply((1 - alpha), sv)
        pv_i = pv_f
        score_collection += (len(G[u]) * pu_f[key_dict[v]] +
                             len(G[v]) * pv_f[key_dict[u]])/(2 * G.size())
    # results generation
    lrw_score = score_collection
    return ((u, v, lrw_score) for u, v in ebunch)

roc_PFP = ROCevaluator(propflow_predictor)
results_list.append(roc_PFP)
print 'AUC for PropFlow Predictor (PFP) = %f' % roc_PFP
print 'counter value = %d' % counter
counter = 0

# generate result csv

with open("results.csv", 'wb') as myfile:
    out = csv.writer(myfile, delimiter=',', quoting=csv.QUOTE_ALL)
    out.writerow(results_list)
