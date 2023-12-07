import math
import operator
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from networkx import Graph
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from graphCR.active_learning import classification
from graphCR.data.cluster import Cluster
from graphCR.evaluation.quality import metrics


def get_precision_distribution(selected_graphs: List[Graph], selected_clusters: List[Cluster], gold_links: set):
    precisions = []
    graph_bins = dict()
    for index, cluster in enumerate(selected_clusters):
        tp_count = 0
        links = metrics.generate_links([cluster])
        colors = []
        for l in links:
            if l in gold_links:
                tp_count += 1
        precisions.append(tp_count / len(links))
        bucket = (math.floor(tp_count / len(links) * 10)) / 10
        if bucket not in graph_bins:
            graph_bins[bucket] = []
        bucket_graphs = graph_bins[bucket]
        bucket_graphs.append((cluster, selected_graphs[index]))
    fig, ax = plt.subplots(figsize=(10, 7))
    hist, bin_edges = np.histogram(precisions, bins=list(np.arange(0.0, 1.1, 0.1)))
    ax.hist(np.asarray(precisions), bins=list(np.arange(0.0, 1.1, 0.1)))
    binning = []
    plt.show()
    hist = hist / len(selected_clusters)
    return hist, bin_edges


def get_edge_distribution(selected_graphs, max_node_number, title='', plotting=False):
    precisions = [g.number_of_edges() for g in selected_graphs]
    hist, bin_edges = np.histogram(precisions, bins=list(np.arange(0.0, max_node_number, 1)))
    if plotting:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title(title)
        fig.set_tight_layout(False)
        ax.hist(np.asarray(precisions), bins=50)
        plt.show()
    hist = hist / len(selected_graphs)
    return hist, bin_edges

def get_node_distribution(selected_graphs, max_node_number, title='', plotting=False):
    precisions = [g.number_of_nodes() for g in selected_graphs]
    hist, bin_edges = np.histogram(precisions, bins=list(np.arange(1, max_node_number+2, 1)))
    if plotting:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_title(title)
        fig.set_tight_layout(False)
        ax.hist(np.asarray(precisions), bins=50)
        plt.show()
    hist = hist / len(selected_graphs)
    return hist, bin_edges
