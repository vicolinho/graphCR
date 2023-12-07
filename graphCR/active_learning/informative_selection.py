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
from multiprocessing import Pool


def get_informative_cluster(cluster_list: list, sample_size, cluster_graphs=None):
    for cluster in cluster_list:
        entity_resource_dict = {}
        for entity in cluster.entities.values():
            if entity.resource not in entity_resource_dict:
                entity_resource_dict[entity.resource] = 1
            else:
                entity_resource_dict[entity.resource] = entity_resource_dict[entity.resource] + 1
        una_violations = 0
        for num_res in entity_resource_dict.values():
            una_violations += num_res - 1
        cluster.una_violations = una_violations
    if cluster_graphs is None:
        cluster_list = sorted(cluster_list, key=lambda v: una_violations, reverse=True)[:sample_size]
        return cluster_list
    else:
        zipped_list = zip(cluster_list, cluster_graphs)
        sorted_pairs = sorted(zipped_list, key=lambda v: v[0].una_violations, reverse=True)[:sample_size]
        tuples = zip(*sorted_pairs)
        sample_cluster, sample_graph = [list(tuple) for tuple in tuples]
        return sample_cluster, sample_graph


def random_selection(unlabelled_graphs, unlabelled_clusters, iteration_budget, gold_links=set()):
    selected_clusters = []
    selected_graphs = []
    remaining_unlabelled_graphs = unlabelled_graphs.copy()
    remaining_unlabelled_clusters = unlabelled_clusters.copy()
    number_of_edges = 0
    while number_of_edges < iteration_budget:
        only_one_class = True
        while only_one_class:
            index = random.choices(range(len(unlabelled_graphs)), k=1)[0]
            graph = unlabelled_graphs[index]
            features, labels = classification.generate_edge_training_data([graph], gold_links)
            if 0 < labels.sum() < graph.number_of_edges():
                selected_graphs.append(unlabelled_graphs[index])
                selected_clusters.append(unlabelled_clusters[index])
                del remaining_unlabelled_graphs[index]
                del remaining_unlabelled_clusters[index]
                number_of_edges += unlabelled_graphs[index].number_of_edges()
                only_one_class = False
    return selected_graphs, selected_clusters, remaining_unlabelled_graphs, remaining_unlabelled_clusters


