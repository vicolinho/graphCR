
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
from graphCR.active_learning import distribution_analyis


'''
Build graphs from selected training graphs so that the distribution of precision is similar to the expected one.
The method works as follows:
1. Determine distribution for expected precisions and training clusters
2. Compute frequency difference for each bucket between expected and training precision
3. Sort clusters by difference
4. Start with the highest difference and generate clusters for bins with a negative frequency
4.1 Remove TN from graphs so that the precision increases and is similar to the bin with the negative frequency.
'''


def uniform_graph_augmentation(selected_graphs: List[Graph], selected_clusters: List[Cluster], gold_links: set,
                               features: list,
                               is_normalized=False, other_graphs=[], expected_hist=np.zeros(10), bins=[],
                               distribution_tol=0.1, max_iteration=100):
    precisions = []
    expected_hist = np.asarray([1 / float(10) for i in range(10)])
    original_graph_bins = dict()
    processed_graphs_bins = dict()
    for index, cluster in enumerate(selected_clusters):
        tp_count = 0
        links = metrics.generate_links([cluster])
        colors = []
        for l in links:
            if l in gold_links:
                tp_count += 1
        precisions.append(tp_count / len(links))
        bucket = round(math.floor(tp_count / len(links) * 10) / 10, 2)
        if bucket == 1:
            bucket = 0.9
        if bucket not in original_graph_bins:
            original_graph_bins[bucket] = []
            processed_graphs_bins[bucket] = set()
        bucket_graphs = original_graph_bins[bucket]
        if tp_count > 0:
            bucket_graphs.append((cluster, selected_graphs[index]))
    fig, ax = plt.subplots(figsize=(10, 7))
    hist, bin_edges = np.histogram(precisions, bins=list(np.arange(0.0, 1.1, 0.1)))
    ax.hist(np.asarray(precisions), bins=list(np.arange(0.0, 1.1, 0.1)))
    binning = []
    plt.show()
    print(hist)
    print(bin_edges)
    hist = hist / len(selected_clusters)
    difference = hist - expected_hist
    most_overestimated_precision_buckets = np.argsort(difference)
    most_underest_indices = np.where(difference < 0)
    # print("ratio difference: {}".format(difference))
    # print("negative indices {}".format(most_underest_indices))
    is_improved = True
    exclusive_new_graphs = []
    exclusive_new_clusters = []
    new_selected_graphs = selected_graphs.copy()
    new_selected_clusters = selected_clusters.copy()
    absolute_difference = np.abs(difference).sum()
    old_difference = 0
    i = 0
    while absolute_difference > distribution_tol and i < max_iteration and absolute_difference != old_difference:
        no_graphs = True
        index = 1
        selected_idx = None
        while no_graphs and index <= most_overestimated_precision_buckets.shape[0]:
            overest_idx = most_overestimated_precision_buckets[-index]
            if difference[overest_idx] > 0:
                overest_prec = round(math.floor(bin_edges[overest_idx] * 10) / 10, 2)
                graph_cluster_list = original_graph_bins[overest_prec]
                processed_graphs = processed_graphs_bins[overest_prec]
                if len(graph_cluster_list) > 0:
                    for i in range(len(graph_cluster_list)):
                        if i not in processed_graphs:
                            no_graphs = False
                            selected_idx = overest_idx
                            break
            index += 1
        # for overest_idx in most_overestimated_precision_buckets:
        if selected_idx is not None:
            overest_prec = round(math.floor(bin_edges[selected_idx] * 10) / 10, 2)
            # print("overestimated precision: {}".format(overest_prec))
            # graph_cluster_list = original_graph_bins[overest_prec]
            processed_graphs = processed_graphs_bins[overest_prec]
            most_underest_indices = np.where(difference < 0)
            under_est_bins = bin_edges[most_underest_indices]
            target_precisions = under_est_bins[:]
            target_precisions_upper = target_precisions + bin_edges[1]
            target_precision_set = set(target_precisions)
            # print(target_precision_set)
            # print("transform to graphs with p={}".format(target_precisions))
            if target_precisions.shape[0] > 0:
                # print("analyze {} graphs".format(len(graph_cluster_list)))
                removing_graphs = []
                graph_index = 0
                for c, g in graph_cluster_list:
                    if graph_index not in processed_graphs:
                        processed_graphs.add(graph_index)
                        links = metrics.generate_links([c])
                        tps = len(links.intersection(gold_links))
                        prec = tps / len(links)
                        prec_diff = target_precisions - prec
                        prec_diff_upper = target_precisions_upper - prec
                        target_number_matches = tps / (prec_diff + prec)
                        target_number_matches_upper = tps / (prec_diff_upper + prec)
                        print(len(links))
                        print(target_number_matches_upper)
                        # print("tps={} prec ={}".format(tps, prec))
                        # print("underrepresented current p={}".format(round(prec, 3)))
                        # print("target number of matches {}".format(target_number_matches))
                        # print("target number of matches upperboung {}".format(target_number_matches_upper))
                        current_min = np.amax(target_number_matches)
                        selected_total_weight = None
                        remove_node_options = []
                        for tnm_index in range(target_number_matches.shape[0]):
                            tnm = target_number_matches[tnm_index]
                            tnm_upper = target_number_matches_upper[tnm_index]
                            if target_precisions[tnm_index] > prec:
                                total_weight, sel_nodes = get_bin_packing_solution(len(links) - tnm,
                                                                                   len(links) - tnm_upper,
                                                                                   links, gold_links)
                            else:
                                total_weight, sel_nodes = get_bin_packing_solution(tnm - len(links),
                                                                                   tnm_upper - len(links),
                                                                                   links, gold_links)
                            if len(sel_nodes) > 0:
                                remove_node_options.append(list(sel_nodes))
                            # if abs(tnm - total_weight) < current_min:
                            #     current_min = abs(tnm - total_weight)
                            #     remove_node_options = [list(sel_nodes)]
                            #     selected_total_weight = total_weight
                            # elif abs(tnm - total_weight) == current_min:
                            #     remove_node_options.append(sel_nodes)
                        if len(remove_node_options) > 0:
                            for rem_option in remove_node_options:
                                new_graph: Graph = g.copy()
                                for rn in rem_option:
                                    new_graph.remove_node(rn)
                                new_graphs = [new_graph.subgraph(c).copy() for c in nx.connected_components(new_graph)]
                                for new_g in new_graphs:
                                    ents = []
                                    for n in new_g.nodes():
                                        ents.append(c.entities[n])
                                    cand_cluster = Cluster(ents)
                                    cand_links = metrics.generate_links([cand_cluster])
                                    if len(cand_links) != 0:
                                        cand_tps = cand_links.intersection(gold_links)
                                        cand_prec = round(math.floor(len(cand_tps) / float(len(cand_links)) * 10) / 10,
                                                          2)
                                        # if cand_prec == 1:
                                        #     cand_prec = 0.9
                                        if cand_prec > overest_prec:
                                            # print("add candidate with prec: {}".format(len(cand_tps)/float(len(cand_links))))
                                            new_selected_clusters.append(cand_cluster)
                                            exclusive_new_clusters.append(cand_cluster)
                                            exclusive_new_graphs.append(new_g)
                                            new_selected_graphs.extend(new_g)
                    graph_index += 1

        new_distribution, new_bin_edges = distribution_analyis.get_precision_distribution(new_selected_graphs, new_selected_clusters,
                                                                     gold_links)
        difference = new_distribution - expected_hist
        i = + 1
        print("ratio difference: {}".format(difference))
        most_overestimated_precision_buckets = np.argsort(difference)
        most_underest_indices = np.where(difference < 0)
        # print("negative indices {}".format(most_underest_indices))
        old_difference = absolute_difference
        absolute_difference = np.abs(difference).sum()
    return exclusive_new_graphs, exclusive_new_clusters


'''
Build graphs from selected training graphs so that the distribution of precision is similar to the expected one.
The method works as follows:
1. Determine distribution for expected precisions and training clusters
2. Compute frequency difference for each bucket between expected and training precision
3. Sort clusters by difference
4. Start with the highest difference and generate clusters for bins with a negative frequency
4.1 Remove TN from graphs so that the precision increases and is similar to the bin with the negative frequency.
'''


def graph_augmentation(selected_graphs: List[Graph], selected_clusters: List[Cluster], gold_links: set, features: list,
                       is_normalized=False,
                       other_graphs=[], expected_hist: np.ndarray = np.zeros(10), bins=[], distribution_tol=0.1,
                       max_iteration=100):
    precisions = []
    original_graph_bins = dict()
    processed_graphs_bins = dict()
    for index, cluster in enumerate(selected_clusters):
        tp_count = 0
        links = metrics.generate_links([cluster])
        colors = []
        for l in links:
            if l in gold_links:
                tp_count += 1
        precisions.append(tp_count / len(links))
        bucket = round(math.floor(tp_count / len(links) * 10) / 10, 2)
        if bucket == 1:
            bucket = 0.9
        if bucket not in original_graph_bins:
            original_graph_bins[bucket] = []
            processed_graphs_bins[bucket] = set()
        bucket_graphs = original_graph_bins[bucket]
        if tp_count > 0:
            bucket_graphs.append((cluster, selected_graphs[index]))
    fig, ax = plt.subplots(figsize=(10, 7))
    hist, bin_edges = np.histogram(precisions, bins=list(np.arange(0.0, 1.1, 0.1)))
    ax.hist(np.asarray(precisions), bins=list(np.arange(0.0, 1.1, 0.1)))
    binning = []
    plt.show()
    print(hist)
    print(bin_edges)
    hist = hist / len(selected_clusters)
    difference = hist - expected_hist
    most_overestimated_precision_buckets = np.argsort(difference)
    most_underest_indices = np.where(difference < 0)
    # print("ratio difference: {}".format(difference))
    # print("negative indices {}".format(most_underest_indices))
    is_improved = True
    exclusive_new_graphs = []
    exclusive_new_clusters = []
    new_selected_graphs = selected_graphs.copy()
    new_selected_clusters = selected_clusters.copy()
    absolute_difference = np.abs(difference).sum()
    old_difference = 0
    i = 0
    while absolute_difference > distribution_tol and i < max_iteration and absolute_difference != old_difference:
        no_graphs = True
        index = 1
        selected_idx = None
        while no_graphs and index <= most_overestimated_precision_buckets.shape[0]:
            overest_idx = most_overestimated_precision_buckets[-index]
            if difference[overest_idx] > 0:
                overest_prec = round(math.floor(bin_edges[overest_idx] * 10) / 10, 2)
                graph_cluster_list = original_graph_bins[overest_prec]
                processed_graphs = processed_graphs_bins[overest_prec]
                if len(graph_cluster_list) > 0:
                    for i in range(len(graph_cluster_list)):
                        if i not in processed_graphs:
                            no_graphs = False
                            selected_idx = overest_idx
                            break
            index += 1
        # for overest_idx in most_overestimated_precision_buckets:
        if selected_idx is not None:
            overest_prec = round(math.floor(bin_edges[selected_idx] * 10) / 10, 2)
            # print("overestimated precision: {}".format(overest_prec))
            # graph_cluster_list = original_graph_bins[overest_prec]
            processed_graphs = processed_graphs_bins[overest_prec]
            most_underest_indices = np.where(difference < 0)
            under_est_bins = bin_edges[most_underest_indices]
            target_precision_indices = np.where(under_est_bins > overest_prec)
            target_precisions = under_est_bins[target_precision_indices]
            target_precisions_upper = target_precisions + bin_edges[1]
            target_precision_set = set(target_precisions)
            # print(target_precision_set)
            # print("transform to graphs with p={}".format(target_precisions))
            if target_precisions.shape[0] > 0:
                # print("analyze {} graphs".format(len(graph_cluster_list)))
                removing_graphs = []
                graph_index = 0
                for c, g in graph_cluster_list:
                    if graph_index not in processed_graphs:
                        processed_graphs.add(graph_index)
                        links = metrics.generate_links([c])
                        tps = len(links.intersection(gold_links))
                        prec = tps / len(links)
                        prec_diff = target_precisions - prec
                        prec_diff_upper = target_precisions_upper - prec

                        target_number_matches = tps / (prec_diff + prec)
                        target_number_matches_upper = tps / (prec_diff_upper + prec)
                        # print("tps={} prec ={}".format(tps, prec))
                        # print("underrepresented current p={}".format(round(prec, 3)))
                        # print("target number of matches {}".format(target_number_matches))
                        # print("target number of matches upperboung {}".format(target_number_matches_upper))
                        current_min = np.amax(target_number_matches)
                        selected_total_weight = None
                        remove_node_options = []
                        for tnm_index in range(target_number_matches.shape[0]):
                            tnm = target_number_matches[tnm_index]
                            tnm_upper = target_number_matches_upper[tnm_index]
                            total_weight, sel_nodes = get_bin_packing_solution(len(links) - tnm, len(links) - tnm_upper,
                                                                               links, gold_links)
                            if len(sel_nodes) > 0:
                                remove_node_options.append(list(sel_nodes))
                            # if abs(tnm - total_weight) < current_min:
                            #     current_min = abs(tnm - total_weight)
                            #     remove_node_options = [list(sel_nodes)]
                            #     selected_total_weight = total_weight
                            # elif abs(tnm - total_weight) == current_min:
                            #     remove_node_options.append(sel_nodes)
                        if len(remove_node_options) > 0:
                            for rem_option in remove_node_options:
                                new_graph: Graph = g.copy()
                                for rn in rem_option:
                                    new_graph.remove_node(rn)
                                new_graphs = [new_graph.subgraph(c).copy() for c in nx.connected_components(new_graph)]
                                for new_g in new_graphs:
                                    ents = []
                                    for n in new_g.nodes():
                                        ents.append(c.entities[n])
                                    cand_cluster = Cluster(ents)
                                    cand_links = metrics.generate_links([cand_cluster])
                                    if len(cand_links) != 0:
                                        cand_tps = cand_links.intersection(gold_links)
                                        cand_prec = round(math.floor(len(cand_tps) / float(len(cand_links)) * 10) / 10,
                                                          2)
                                        if cand_prec == 1:
                                            cand_prec = 0.9
                                        if cand_prec > overest_prec:
                                            # print("add candidate with prec: {}".format(len(cand_tps)/float(len(cand_links))))
                                            new_selected_clusters.append(cand_cluster)
                                            exclusive_new_clusters.append(cand_cluster)
                                            exclusive_new_graphs.append(new_g)
                                            new_selected_graphs.extend(new_g)
                    graph_index += 1

        new_distribution, new_bin_edges = distribution_analyis.get_precision_distribution(new_selected_graphs, new_selected_clusters,
                                                                     gold_links)
        difference = new_distribution - expected_hist
        i = + 1
        print("ratio difference: {}".format(difference))
        most_overestimated_precision_buckets = np.argsort(difference)
        most_underest_indices = np.where(difference < 0)
        # print("negative indices {}".format(most_underest_indices))
        old_difference = absolute_difference
        absolute_difference = np.abs(difference).sum()
    return exclusive_new_graphs, exclusive_new_clusters


def get_bin_packing_solution_with_adding_TN(target_number, tnm_upper, links, gold_links):
    total_weight = 0
    selected_nodes = set()
    is_change = True
    # print(target_number)
    copy_links = links.copy()
    while total_weight < tnm_upper and is_change:
        tn_per_nodes = get_list_of_tn(copy_links, gold_links)
        sel_node = None
        is_change = False
        sel_tn = 0
        diff_weight = tnm_upper
        for n, tn in tn_per_nodes.items():
            if n not in selected_nodes:
                if tn <= tnm_upper:
                    if total_weight + tn <= tnm_upper:
                        sel_node = n
                        sel_tn = tn
                        diff_weight = abs((total_weight + tn) - tnm_upper)
                        break
                    elif abs((total_weight + tn) - tnm_upper) < diff_weight:
                        diff_weight = abs((total_weight + tn) - tnm_upper)
                        sel_node = n
                        sel_tn = tn
        if sel_node is not None and diff_weight < abs(total_weight - tnm_upper):
            selected_nodes.add(sel_node)
            new_links = set()
            for u, v in copy_links:
                if u != sel_node and v != sel_node:
                    new_links.add((u, v))
            copy_links = new_links
            total_weight += sel_tn
            is_change = True
    return total_weight, selected_nodes


def get_bin_packing_solution(target_number, tnm_upper, links, gold_links):
    total_weight = 0
    selected_nodes = set()
    is_change = True
    # print(target_number)
    copy_links = links.copy()
    while total_weight < tnm_upper and is_change:
        tn_per_nodes = get_list_of_tn(copy_links, gold_links)
        sel_node = None
        is_change = False
        sel_tn = 0
        diff_weight = tnm_upper
        for n, tn in tn_per_nodes.items():
            if n not in selected_nodes:
                if tn <= tnm_upper:
                    if total_weight + tn <= tnm_upper:
                        sel_node = n
                        sel_tn = tn
                        diff_weight = abs((total_weight + tn) - tnm_upper)
                        break
                    elif abs((total_weight + tn) - tnm_upper) < diff_weight:
                        diff_weight = abs((total_weight + tn) - tnm_upper)
                        sel_node = n
                        sel_tn = tn
        if sel_node is not None and diff_weight < abs(total_weight - tnm_upper):
            selected_nodes.add(sel_node)
            new_links = set()
            for u, v in copy_links:
                if u != sel_node and v != sel_node:
                    new_links.add((u, v))
            copy_links = new_links
            total_weight += sel_tn
            is_change = True
    return total_weight, selected_nodes


def get_list_of_tn(cluster_links, gold_links):
    tn_per_node = dict()
    for l in cluster_links:
        if l not in gold_links:
            if l[0] not in tn_per_node:
                tn_per_node[l[0]] = 1
            else:
                tn_per_node[l[0]] = 1 + tn_per_node[l[0]]
            if l[1] not in tn_per_node:
                tn_per_node[l[1]] = 1
            else:
                tn_per_node[l[1]] = 1 + tn_per_node[l[1]]
    return tn_per_node