import operator

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from graphCR.active_learning.selection_methods import farthest_first_selection


def informative_selection_edge_wise(training_feature_edges, labeled_vectors,
                                    unlabelled_feature_edges: np.ndarray,
                                    unlabelled_classes: np.ndarray,
                                    iteration_budget):
    pos_indices = np.where(labeled_vectors == 1)[0]
    neg_indices = np.where(labeled_vectors == 0)[0]

    # compute entropy based measure

    pos_vectors = training_feature_edges[pos_indices]
    neg_vectors = training_feature_edges[neg_indices]
    cos_diff = cosine_similarity(pos_vectors, neg_vectors)
    cos_same_pos = cosine_similarity(pos_vectors, pos_vectors)
    cos_same_neg = cosine_similarity(neg_vectors, neg_vectors)
    sum_pos_same = np.sum(cos_same_pos, axis=1)
    sum_neg_same = np.sum(cos_same_neg, axis=1)
    sum_pos_diff = np.sum(cos_diff, axis=1)
    sum_neg_diff = np.sum(cos_diff, axis=0)
    entropy_pos = - (
            sum_pos_same / training_feature_edges.shape[0] * np.log2(sum_pos_same / training_feature_edges.shape[0]) \
            + sum_pos_diff / training_feature_edges.shape[0] * np.log2(
        sum_pos_diff / training_feature_edges.shape[0]))
    assert pos_vectors.shape[0] == entropy_pos.shape[0], "wrong dimensions of pos vectors"
    entropy_neg = - (
            sum_neg_same / training_feature_edges.shape[0] * np.log2(sum_neg_same / training_feature_edges.shape[0]) \
            + sum_neg_diff / training_feature_edges.shape[0] * np.log2(
        sum_neg_diff / training_feature_edges.shape[0]))
    # compute uncertainty
    pos_max_sims = np.amax(cos_diff, axis=1)
    neg_max_sims = np.amax(cos_diff, axis=0)
    pos_max_sims = np.reshape(pos_max_sims, (pos_max_sims.shape[0], 1))
    neg_max_sims = np.reshape(neg_max_sims, (neg_max_sims.shape[0], 1))
    s_same_pos = cos_same_pos >= pos_max_sims
    s_same_neg = cos_same_neg >= neg_max_sims
    unc_pos = np.float32(s_same_pos.sum(axis=1))
    unc_neg = np.float32(s_same_neg.sum(axis=1))
    unc_pos = np.reciprocal(unc_pos)
    unc_neg = np.reciprocal(unc_neg)

    info_pos = (unc_pos + entropy_pos) / 2.
    info_neg = (unc_neg + entropy_neg) / 2.
    union = np.hstack((info_pos, info_neg))
    threshold = np.percentile(union, 75)
    pos_indices = np.where(info_pos >= threshold)
    neg_indices = np.where(info_neg >= threshold)

    # select informative training data points
    i_pos = pos_vectors[pos_indices]
    i_pos_thresh = pos_max_sims[pos_indices]
    i_neg = neg_vectors[neg_indices]
    i_neg_thresh = neg_max_sims[neg_indices]
    i_vecs = np.vstack((i_pos, i_neg))

    i_search_t = np.vstack((i_pos_thresh, i_neg_thresh))
    cos_sim_with_unlabelled = cosine_similarity(i_vecs, unlabelled_feature_edges)

    # select unlabelled candidates
    selected_unlabelled = cos_sim_with_unlabelled >= i_search_t
    next_indices = np.where(np.any(selected_unlabelled, axis=0))
    # farthest first selection
    candidate_indices = farthest_first_selection.graipher(unlabelled_feature_edges[next_indices], iteration_budget)

    new_vectors = unlabelled_feature_edges[candidate_indices]
    new_classes = unlabelled_classes[candidate_indices]
    remaining_unlabelled_feature_edges = np.delete(unlabelled_feature_edges, candidate_indices, axis=0)
    remaining_unlabelled_classes = np.delete(unlabelled_classes, candidate_indices, axis=0)
    return new_vectors, new_classes, remaining_unlabelled_feature_edges, remaining_unlabelled_classes


def informative_selection_edge_wise_opt(training_feature_edges, labeled_vectors,
                                        unlabelled_feature_edges: np.ndarray,
                                        unlabelled_classes: np.ndarray,
                                        iteration_budget,  edge_graph_index, graph_list, expected_node_dis,
                                        current_node_dis, bin_edges:np.ndarray, current_prec, expected_p,
                                        max_search_sim=0.9):

    number_node_diff = expected_node_dis - current_node_dis
    number_node_diff = np.where(number_node_diff <= 0, 0, number_node_diff)
    scale = number_node_diff - number_node_diff.min() / float(number_node_diff.max() - number_node_diff.min())
    print("exp: {} current {}".format(expected_p, current_prec))
    # p_diff = current_prec - expected_p
    # pos_p_weight = - p_diff
    # neg_p_weight = p_diff
    p_diff = current_prec - expected_p
    pos_p_weight = 0.5 - p_diff
    neg_p_weight = 0.5 + p_diff
    print(pos_p_weight)
    print("neg {}".format(neg_p_weight))
    pos_indices = np.where(labeled_vectors == 1)[0]
    neg_indices = np.where(labeled_vectors == 0)[0]

    # compute entropy based measure

    pos_vectors = training_feature_edges[pos_indices]
    neg_vectors = training_feature_edges[neg_indices]
    cos_diff = cosine_similarity(pos_vectors, neg_vectors)
    cos_same_pos = cosine_similarity(pos_vectors, pos_vectors)
    cos_same_neg = cosine_similarity(neg_vectors, neg_vectors)
    sum_pos_same = np.sum(cos_same_pos, axis=1)
    sum_neg_same = np.sum(cos_same_neg, axis=1)
    sum_pos_diff = np.sum(cos_diff, axis=1)
    sum_neg_diff = np.sum(cos_diff, axis=0)
    entropy_pos = - (
            sum_pos_same / training_feature_edges.shape[0] * np.log2(sum_pos_same / training_feature_edges.shape[0]) \
            + sum_pos_diff / training_feature_edges.shape[0] * np.log2(
        sum_pos_diff / training_feature_edges.shape[0]))
    assert pos_vectors.shape[0] == entropy_pos.shape[0], "wrong dimensions of pos vectors"
    entropy_neg = - (
            sum_neg_same / training_feature_edges.shape[0] * np.log2(sum_neg_same / training_feature_edges.shape[0]) \
            + sum_neg_diff / training_feature_edges.shape[0] * np.log2(
        sum_neg_diff / training_feature_edges.shape[0]))
    # compute uncertainty
    pos_max_sims = np.amax(cos_diff, axis=1)
    neg_max_sims = np.amax(cos_diff, axis=0)
    pos_max_sims = np.reshape(pos_max_sims, (pos_max_sims.shape[0], 1))
    neg_max_sims = np.reshape(neg_max_sims, (neg_max_sims.shape[0], 1))
    s_same_pos = cos_same_pos >= pos_max_sims
    s_same_neg = cos_same_neg >= neg_max_sims
    unc_pos = np.float32(s_same_pos.sum(axis=1))
    unc_neg = np.float32(s_same_neg.sum(axis=1))
    unc_pos = np.reciprocal(unc_pos)
    unc_neg = np.reciprocal(unc_neg)
    pos_search_space_size = 1 - pos_max_sims
    pos_search_space_size = pos_search_space_size.flatten()
    neg_search_space_size = 1 - neg_max_sims
    neg_search_space_size = neg_search_space_size.flatten()
    info_pos = (unc_pos + entropy_pos + pos_p_weight) / 3.
    info_neg = (unc_neg + entropy_neg + neg_p_weight) / 3.


    # vec_union = np.vstack((pos_vectors, neg_vectors))
    union = np.hstack((info_pos, info_neg))
    max_sims = np.vstack((pos_max_sims, neg_max_sims))
    # indices = np.argsort(union)[-iteration_budget:]
    threshold = np.percentile(union, 75)
    pos_indices = np.where(info_pos >= threshold)
    neg_indices = np.where(info_neg >= threshold)
    print("selected pos:{} neg:{}".format(len(pos_indices), len(neg_indices)))

    # select informative training data points
    i_pos = pos_vectors[pos_indices]
    i_pos_thresh = pos_max_sims[pos_indices]
    i_neg = neg_vectors[neg_indices]
    i_neg_thresh = neg_max_sims[neg_indices]
    i_vecs = np.vstack((i_pos, i_neg))
    i_search_t = np.vstack((i_pos_thresh, i_neg_thresh))
    # i_vecs = vec_union[indices]
    # print(i_vecs.shape)
    # i_search_t = max_sims[indices]


    # info_indices = graipher(i_vecs, iteration_budget)
    # i_vecs = i_vecs[info_indices]
    # i_search_t = i_search_t[info_indices]

    # remove info vectors with a small search space
    # search_indices = np.where(i_search_t.flatten() <= max_search_sim)
    # if len(search_indices) > 0:
    #     i_vecs = i_vecs[search_indices]
    #     i_search_t = i_search_t[search_indices]

    if i_vecs.shape[0] == 1:
        i_vecs = i_vecs.reshape((1, -1))

    cos_sim_with_unlabelled = cosine_similarity(i_vecs, unlabelled_feature_edges)

    # select unlabelled candidates
    selected_unlabelled = cos_sim_with_unlabelled >= i_search_t
    mean_search = (1 + i_search_t) / 2.
    mid_sims = 1 - np.abs(cos_sim_with_unlabelled - mean_search)
    next_indices = np.where(np.any(selected_unlabelled, axis=0))
    mid_sims = mid_sims.squeeze()
    new_array = np.zeros(mid_sims.shape)
    new_array[selected_unlabelled] = mid_sims[selected_unlabelled]
    assert new_array.shape == cos_sim_with_unlabelled.shape, "not same shape like sim matrix {}".format(new_array.shape)
    max_mid_sims = np.amax(new_array, axis=0)
    node_weight = np.zeros(max_mid_sims.shape[0])
    for i in range(max_mid_sims.shape[0]):
        graph_index_list = edge_graph_index[str(unlabelled_feature_edges[i])]
        weight = 0
        for graph_index in graph_index_list:
            node_number = graph_list[graph_index].number_of_nodes()
            bin_distances = np.abs(bin_edges[:-1] - node_number)
            min_index = np.argmin(bin_distances)
            weight += 1 / float(1 + bin_distances[min_index]) * scale[min_index]
        # graph_index = edge_graph_index[str(unlabelled_feature_edges[i])]
        # node_number = graph_list[graph_index].number_of_nodes()
        # bin_distances = np.abs(bin_edges[:-1] - node_number)
        # min_index = np.argmin(bin_distances)
        # weight = 1/float(1+bin_distances[min_index])*scale[min_index]
        node_weight[i] = weight/len(graph_index_list)
    max_mid_sims = max_mid_sims * node_weight
    candidate_indices = np.argsort(max_mid_sims)[-iteration_budget:]
    #
    # print(mid_sims_filtered.shape)
    # farthest first selection
    # candidate_indices = graipher(unlabelled_feature_edges[next_indices], iteration_budget)

    new_vectors = unlabelled_feature_edges[candidate_indices]

    new_classes = unlabelled_classes[candidate_indices]
    print("current: {}".format(labeled_vectors.sum()))
    print("positives: {} negatives: {}".format(new_classes.sum(), new_classes.shape[0]-new_classes.sum()))
    remaining_unlabelled_feature_edges = np.delete(unlabelled_feature_edges, candidate_indices, axis=0)
    remaining_unlabelled_classes = np.delete(unlabelled_classes, candidate_indices, axis=0)
    return new_vectors, new_classes, remaining_unlabelled_feature_edges, remaining_unlabelled_classes


def informative_selection_cluster_wise(train_graphs_with_features, gold_links, unlabelled_graphs,
                                       unlabelled_clusters, iteration_budget, diff_percentage_to_next=0.01):
    training_feature_edges, labeled_vectors = classification.generate_edge_training_data(train_graphs_with_features,
                                                                                         gold_links)
    unlabelled_feature_edges, unlabelled_classes = classification.generate_edge_training_data(unlabelled_graphs,
                                                                                              gold_links)

    # compute entropy based measure

    pos_indices = np.where(labeled_vectors == 1)[0]
    neg_indices = np.where(labeled_vectors == 0)[0]

    # compute entropy based measure

    pos_vectors = training_feature_edges[pos_indices]
    neg_vectors = training_feature_edges[neg_indices]
    cos_diff = cosine_similarity(pos_vectors, neg_vectors)
    cos_same_pos = cosine_similarity(pos_vectors, pos_vectors)
    cos_same_neg = cosine_similarity(neg_vectors, neg_vectors)
    sum_pos_same = np.sum(cos_same_pos, axis=1)
    sum_neg_same = np.sum(cos_same_neg, axis=1)
    sum_pos_diff = np.sum(cos_diff, axis=1)
    sum_neg_diff = np.sum(cos_diff, axis=0)
    entropy_pos = - (
            sum_pos_same / training_feature_edges.shape[0] * np.log2(sum_pos_same / training_feature_edges.shape[0]) \
            + sum_pos_diff / training_feature_edges.shape[0] * np.log2(
        sum_pos_diff / training_feature_edges.shape[0]))
    assert pos_vectors.shape[0] == entropy_pos.shape[0], "wrong dimensions of pos vectors"
    entropy_neg = - (
            sum_neg_same / training_feature_edges.shape[0] * np.log2(sum_neg_same / training_feature_edges.shape[0]) \
            + sum_neg_diff / training_feature_edges.shape[0] * np.log2(
        sum_neg_diff / training_feature_edges.shape[0]))
    # compute uncertainty
    pos_max_sims = np.amax(cos_diff, axis=1)
    neg_max_sims = np.amax(cos_diff, axis=0)
    pos_max_sims = np.reshape(pos_max_sims, (pos_max_sims.shape[0], 1))
    neg_max_sims = np.reshape(neg_max_sims, (neg_max_sims.shape[0], 1))
    s_same_pos = cos_same_pos >= pos_max_sims
    s_same_neg = cos_same_neg >= neg_max_sims
    unc_pos = np.float32(s_same_pos.sum(axis=1))
    unc_neg = np.float32(s_same_neg.sum(axis=1))
    unc_pos = np.reciprocal(unc_pos)
    unc_neg = np.reciprocal(unc_neg)

    info_pos = (unc_pos + entropy_pos) / 2.
    info_neg = (unc_neg + entropy_neg) / 2.
    union = np.hstack((info_pos, info_neg))
    threshold = np.percentile(union, 75)
    pos_indices = np.where(info_pos >= threshold)
    neg_indices = np.where(info_neg >= threshold)

    # select informative training data points
    i_pos = pos_vectors[pos_indices]
    i_pos_thresh = pos_max_sims[pos_indices]
    i_neg = neg_vectors[neg_indices]
    i_neg_thresh = neg_max_sims[neg_indices]
    i_vecs = np.vstack((i_pos, i_neg))

    i_search_t = np.vstack((i_pos_thresh, i_neg_thresh))
    cos_sim_with_unlabelled = cosine_similarity(i_vecs, unlabelled_feature_edges)

    # select unlabelled candidates
    selected_unlabelled = cos_sim_with_unlabelled >= i_search_t
    next_indices = np.where(np.any(selected_unlabelled, axis=0))
    # farthest first selection
    candidate_indices = farthest_first_selection.graipher(unlabelled_feature_edges[next_indices], iteration_budget)
    candidate_indices_set = set(candidate_indices.flatten())
    index = 0
    graph_index = 0
    info_count_per_graph = {}
    for graph in unlabelled_graphs:
        for u, v in graph.edges():
            if index in candidate_indices_set:
                if graph_index not in info_count_per_graph:
                    info_count_per_graph[graph_index] = 1
                else:
                    info_count_per_graph[graph_index] = info_count_per_graph[graph_index] + 1
            index += 1
        graph_index += 1
    for g_idx in info_count_per_graph.keys():
        info_count_per_graph[g_idx] = float(info_count_per_graph[g_idx]) / float(
            unlabelled_graphs[g_idx].number_of_edges())
    candidate_examples = sorted(info_count_per_graph.items(), key=operator.itemgetter(1), reverse=True)[
                         :min(iteration_budget, len(info_count_per_graph))]
    next_batch_idxs = [val[0] for val in candidate_examples]
    number_of_edges = 0
    selected_graphs = []
    selected_clusters = []
    remove_idx = []
    last_max = 0.01
    for idx in next_batch_idxs:
        if number_of_edges < iteration_budget and (info_count_per_graph[idx] / last_max) > diff_percentage_to_next:
            selected_graphs.append(unlabelled_graphs[idx])
            selected_clusters.append(unlabelled_clusters[idx])
            remove_idx.append(idx)
            last_max = info_count_per_graph[idx]
        else:
            break
        number_of_edges += unlabelled_graphs[idx].number_of_edges()
    for idx in sorted(remove_idx, reverse=True):
        del unlabelled_graphs[idx]
        del unlabelled_clusters[idx]
    return selected_graphs, selected_clusters, unlabelled_graphs, unlabelled_clusters