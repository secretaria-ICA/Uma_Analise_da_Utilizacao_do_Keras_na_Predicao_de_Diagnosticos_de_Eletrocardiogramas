from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from tensorflow.keras.layers.experimental import RandomFourierFeatures
import pandas as pd
import numpy as np
import tensorflow as tf
import wfdb
import ast
import itertools
import matplotlib.pyplot as plt
import operator
import networkx as nx

def scp_graph(scp_statements):
    diagnostic_class = scp_statements.loc[scp_statements['diagnostic'] == 1.0,'diagnostic_class'].unique()

    G = nx.Graph()

    colors = [
        "gold",
        "violet",
        "darkorange",
    ]

    for k in diagnostic_class:
        G.add_node(k + ' class', layer=0)
        diagnostic_subclass = scp_statements.loc[scp_statements['diagnostic_class'] == k, 'diagnostic_subclass'].unique()
        for s in diagnostic_subclass:
            G.add_node(s + ' subclass', layer=1)
            G.add_edge(k + ' class', s + ' subclass')
            diag_codes = scp_statements.loc[scp_statements['diagnostic_subclass'] == s].index
            for c in diag_codes:
                G.add_node(c, layer=2)
                G.add_edge(s + ' subclass', c)

    pos = nx.multipartite_layout(G, subset_key="layer")

    color = [colors[data["layer"]] for v, data in G.nodes(data=True)]

    for k, v in pos.items():
        if ' subclass' in k:
            neighbors = G.neighbors(k)
            new_pos = pos[k]
            acc = []
            for n in neighbors:
                if 'class' not in n:
                    acc.append(pos[n][1])
            acc_mean = sum(acc) / len(acc)
            new_pos[1] = acc_mean
            pos[k] = new_pos

    for k, v in pos.items():
        if ' class' in k:
            neighbors = G.neighbors(k)
            new_pos = pos[k]
            acc = []
            for n in neighbors:
                acc.append(pos[n][1])
            acc_mean = sum(acc) / len(acc)
            new_pos[1] = acc_mean
            pos[k] = new_pos

    return {'G': G, 'pos': pos, 'color': color}

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)