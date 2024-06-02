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



def bootstrap_selection_edge_wise(model_type, training_feature_edges, labeled_vectors,
                                  unlabelled_feature_edges: np.ndarray,
                                  unlabelled_classes: np.ndarray, k,
                                  iteration_budget):
    classifiers = []
    for i in range(k):
        model = classification.get_model(model_type)
        bootstrapping_indices = np.random.choice(training_feature_edges.shape[0], training_feature_edges.shape[0],
                                                 replace=True)

        subset_vectors = training_feature_edges[bootstrapping_indices]
        subset_classes = labeled_vectors[bootstrapping_indices]
        # model = classification.train_by_numpy_edges(subset_vectors, subset_classes, model_type)
        model.fit(subset_vectors, subset_classes)
        classifiers.append(model)
    # select informative record-pairs
    distribution = np.zeros((unlabelled_feature_edges.shape[0], k))
    for i in range(len(classifiers)):
        model_i = classifiers[i]
        classes = model_i.predict(unlabelled_feature_edges)
        distribution[:, i] = classes
    agg_distribution = distribution.sum(1)

    uncertainties = {}
    for i in range(len(distribution)):
        x_u = agg_distribution[i] / float(len(classifiers))
        uncertainties[i] = x_u * (1 - x_u)
    candidate_examples = sorted(uncertainties.items(), key=operator.itemgetter(1), reverse=True)[
                         :min(iteration_budget, len(uncertainties))]
    next_batch_idxs = [val[0] for val in candidate_examples]
    new_vectors = unlabelled_feature_edges[next_batch_idxs]
    new_classes = unlabelled_classes[next_batch_idxs]
    remaining_unlabelled_feature_edges = np.delete(unlabelled_feature_edges, next_batch_idxs, axis=0)
    remaining_unlabelled_classes = np.delete(unlabelled_classes, next_batch_idxs, axis=0)
    return new_vectors, new_classes, remaining_unlabelled_feature_edges, remaining_unlabelled_classes

def fit_model (td_tuple):
    td_tuple[0].fit(td_tuple[1], td_tuple[2])
    return td_tuple[0]

# weights distance to most similar bucket difference between expected and current
def bootstrap_cluster_size_selection_edge_wise(model_type, training_feature_edges, labeled_vectors,
                                  unlabelled_feature_edges: np.ndarray,
                                  unlabelled_classes: np.ndarray, k,
                                  iteration_budget, edge_graph_index, graph_list, expected_node_dis: np.ndarray,
                                               current_node_dis: np.ndarray, bin_edges:np.ndarray):
    number_node_diff = expected_node_dis - current_node_dis
    number_node_diff = np.where(number_node_diff <= 0, 0, number_node_diff)
    scale = number_node_diff - number_node_diff.min()/float(number_node_diff.max() - number_node_diff.min())
    classifiers = []
    train_data = []
    result = []
    cos_sim_with_unlabelled = cosine_similarity(unlabelled_feature_edges, training_feature_edges)
    cos_average_distance = 1 - np.mean(cos_sim_with_unlabelled, axis=1)
    for i in range(k):
        model = classification.get_model(model_type)
        bootstrapping_indices = np.random.choice(training_feature_edges.shape[0], training_feature_edges.shape[0],
                                                 replace=True)
        subset_vectors = training_feature_edges[bootstrapping_indices]
        subset_classes = labeled_vectors[bootstrapping_indices]
        model.fit(subset_vectors, subset_classes)
        classifiers.append(model)
    distribution = np.zeros((unlabelled_feature_edges.shape[0], k))
    for i in range(len(classifiers)):
        model_i = classifiers[i]
        classes = model_i.predict(unlabelled_feature_edges)
        distribution[:, i] = classes
    agg_distribution = distribution.sum(1)

    uncertainties = {}
    node_weight = {}
    for i in range(len(distribution)):
        x_u = agg_distribution[i] / float(len(classifiers))
        graph_index_list = edge_graph_index[str(unlabelled_feature_edges[i])]
        weight = 0
        for graph_index in graph_index_list:
            node_number = graph_list[graph_index].number_of_nodes()

            bin_distances = np.abs(bin_edges - node_number)
            min_index = np.argmin(bin_distances)
            if bin_distances[min_index] > 0:
                print("number of nodes {}".format(node_number))
                print(bin_distances[min_index])
            weight += 1/float(1+bin_distances[min_index])*scale[min_index]
        node_weight[i] = weight/len(graph_index_list)
        #uncertainties[i] = 0.5 * (x_u * (1 - x_u))/0.25 + 0.5 * weight
        #uncertainties[i] = 4*(x_u * (1 - x_u)) * weight * cos_average_distance[i]
        uncertainties[i] = (4 * (x_u * (1 - x_u)) + weight + cos_average_distance[i])/3.
    candidate_examples = sorted(uncertainties.items(), key=operator.itemgetter(1), reverse=True)[
                         :min(iteration_budget, len(uncertainties))]
    # candidate_examples = sorted(uncertainties.items(), key=operator.itemgetter(1), reverse=True)[
    #                      :min(iteration_budget * 2, len(uncertainties))]
    next_batch_idxs = [val[0] for val in candidate_examples]
    # filtered_weight ={}
    # for id in next_batch_idxs:
    #     filtered_weight[id] = node_weight[id]
    # candidate_examples = sorted(filtered_weight.items(), key=operator.itemgetter(1), reverse=True)[
    #                      :min(iteration_budget, len(filtered_weight))]
    # next_batch_idxs = [val[0] for val in candidate_examples]
    new_vectors = unlabelled_feature_edges[next_batch_idxs]
    new_classes = unlabelled_classes[next_batch_idxs]

    node_numbers = {}
    # for i in range(new_vectors.shape[0]):
    #     graph_index = edge_graph_index[str(unlabelled_feature_edges[i])]
    #     node_number = graph_list[graph_index].number_of_nodes()
    #     node_numbers[i] = node_number

    remaining_unlabelled_feature_edges = np.delete(unlabelled_feature_edges, next_batch_idxs, axis=0)
    remaining_unlabelled_classes = np.delete(unlabelled_classes, next_batch_idxs, axis=0)
    return new_vectors, new_classes, remaining_unlabelled_feature_edges, remaining_unlabelled_classes, next_batch_idxs



def bootstrap_selection_cluster_wise(model_type, train_graphs_with_features, gold_links, unlabelled_graphs,
                                     unlabelled_clusters, k,
                                     iteration_budget):
    current_train_vectors, labeled_vectors = classification.generate_edge_training_data(train_graphs_with_features,
                                                                                        gold_links)

    unlabeled_vectors, _ = classification.generate_edge_training_data(unlabelled_graphs)
    classifiers = []
    for i in range(k):
        model = classification.get_model(model_type)
        bootstrapping_indices = np.random.choice(current_train_vectors.shape[0], current_train_vectors.shape[0],
                                                 replace=True)

        subset_vectors = current_train_vectors[bootstrapping_indices]
        subset_classes = labeled_vectors[bootstrapping_indices]
        model.fit(subset_vectors, subset_classes)
        classifiers.append(model)
    # select informative record-pairs
    distribution = np.zeros((unlabeled_vectors.shape[0], k))
    for i in range(len(classifiers)):
        model_i = classifiers[i]
        classes = model_i.predict(unlabeled_vectors)
        distribution[:, i] = classes
    agg_distribution = distribution.sum(1)
    uncertainties = {}
    for i in range(len(distribution)):
        x_u = agg_distribution[i] / len(classifiers)
        uncertainties[i] = x_u * (1 - x_u)
    graph_uncertainties = {}

    index = 0
    graph_index = 0
    for graph in unlabelled_graphs:
        for u, v in graph.edges():
            if graph_index not in graph_uncertainties:
                graph_uncertainties[graph_index] = uncertainties[index] / float(graph.number_of_edges())
            else:
                graph_uncertainties[graph_index] = graph_uncertainties[graph_index] + \
                                                   uncertainties[index] / float(graph.number_of_edges())
            index += 1
        graph_index += 1
    candidate_examples = sorted(graph_uncertainties.items(), key=operator.itemgetter(1), reverse=True)[
                         :min(iteration_budget, len(graph_uncertainties))]

    next_batch_idxs = [val[0] for val in candidate_examples]

    number_of_edges = 0
    selected_graphs = []
    selected_clusters = []
    remove_idx = []
    for idx in next_batch_idxs:
        if number_of_edges < iteration_budget:
            if unlabelled_graphs[idx].number_of_edges() < iteration_budget:
                selected_graphs.append(unlabelled_graphs[idx])
                selected_clusters.append(unlabelled_clusters[idx])
                remove_idx.append(idx)
                number_of_edges += unlabelled_graphs[idx].number_of_edges()
        else:
            break

    for idx in sorted(remove_idx, reverse=True):
        del unlabelled_graphs[idx]
        del unlabelled_clusters[idx]
    return selected_graphs, selected_clusters, unlabelled_graphs, unlabelled_clusters