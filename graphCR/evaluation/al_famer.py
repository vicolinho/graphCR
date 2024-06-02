import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

import numpy
import numpy as np

import graphCR.active_learning.constant as al_const
from graphCR.active_learning import classification, informative_selection, distribution_analyis
from graphCR.active_learning.llm_labeling import LLMLabeling
from graphCR.active_learning.selection_methods import bootstrap, informativeness, farthest_first_selection
from graphCR.data.test_data import reader, famer_constant
from graphCR.evaluation.quality import metrics
from graphCR.evaluation.util import scatter, scatter_pca
from graphCR.feature_generation import graph_feature_generation, graph_construction


#edge_features=['bridges', 'betweenness', complete_ratio]
def evaluate(input_folder='E:/data/DS-C/DS-C/DS-C0/SW_0.7',
             features=['pagerank', 'closeness', 'cluster_coefficient', 'betweenness'],
             edge_features=['bridges', 'betweenness', 'complete_ratio'],
             model_type=al_const.RANDOM_FOREST,
             selection_strategy='info_opt',
             initial_training=10, total_budget=1000, min_cluster_size=2
             , increment_budget=50, is_edge_wise=True, output="result_al.csv", output_2="result_initial_al.csv",
             error_edge_ratio=0.1, use_gpt=0, considered_atts=[], api_key=None, model_name="gpt-3.5-turbo",
             cache_path='requested_pairs.csv'):
    np.random.seed(31)
    entities, cluster_graphs, cluster_list = reader.read_data(input_folder, error_edge_ratio)
    print(input_folder)
    match = re.search(r'((?<=threshold_)|(?<=SW_))[0-9]\.[0-9]{1,2}', input_folder)
    data_set = re.search(r'(?<=/)DS.*(?=/threshold|/SW_)', input_folder).group()
    threshold = float(match.group())
    print("selection strategy: {}".format(selection_strategy))
    print("is edgewise: {}".format(is_edge_wise))
    print("threshold: {}".format(threshold))
    gold_clusters = reader.generate_gold_clusters(entities)
    print("{} clusters in gold standard".format(len(gold_clusters)))
    assert (len(cluster_graphs) > 0)
    gold_links = metrics.generate_links(gold_clusters)
    print("number of clusters: {}".format(len(cluster_list)))
    graphs_with_features = []
    assert (len(cluster_graphs) > 0)
    filtered_graphs = []
    filtered_clusters = []
    all_edges = []
    memory = 0
    if use_gpt:
        llm_labeler = LLMLabeling(api_key, cache_path)

    for index, graph in enumerate(cluster_graphs):
        f_cluster, f_graphs = graph_construction.filter_links(cluster_list[index], graph)
        filtered_clusters.extend(f_cluster)
        filtered_graphs.extend(f_graphs)
        all_edges.extend([tuple(sorted((u, v))) for u, v in graph.edges()])
    for index, graph in enumerate(filtered_graphs):
        if graph.number_of_nodes() >= min_cluster_size:
            edge_mem = sum([sys.getsizeof(e) for e in graph.edges()])
            node_mem = sum([sys.getsizeof(n) for n in graph.nodes()])

            memory += edge_mem + node_mem
            feature_graph = graph_feature_generation.generate_features(graph, filtered_clusters[index], features)
            _, _, feature_graph = graph_feature_generation.link_category_feature(filtered_clusters[index],
                                                                                 feature_graph)
            feature_graph = graph_feature_generation.edge_feature_generation(graph, edge_features)
            graphs_with_features.append(feature_graph)

    hist, bin_edges = distribution_analyis.get_precision_distribution(graphs_with_features, filtered_clusters,
                                                                       gold_links)
    num_pred, num_gold, tps, p, r, f = metrics.compute_quality(cluster_list, gold_clusters)
    # graphs_with_features = graph_feature_generation.normalize_all_edges(graphs_with_features)
    numpy_edges, labels, edge_graph_dict, node_pair_ids = classification.generate_edge_training_data(graphs_with_features, gold_links)

    assert numpy_edges.shape[1] == len(features) + len(edge_features) + 2 + 1, print(numpy_edges.shape[1])
    print("size(MB):{}".format(memory/1024/1024))
    print("total number of edges:{}".format(numpy_edges.shape[0]))
    # edgehist, bin_edges_number = informative_selection.get_edge_distribution(graphs_with_features)
    distinct_numpy_edge, unique_indices = np.unique(numpy_edges, return_index=True, axis=0)
    edges_from_graphs = [graphs_with_features[graph_index] for edge_feature in
                         distinct_numpy_edge for graph_index in edge_graph_dict[str(edge_feature)]]
    max_number = max([g.number_of_nodes() for g in edges_from_graphs])
    print("max number {}".format(max_number))
    total_node_dis, total_bins = distribution_analyis.get_node_distribution(edges_from_graphs, max_number,
                                                                             "complete sel={} edge_wise={} b={} i={}"
                                                .format(selection_strategy, is_edge_wise, total_budget,
                                                        increment_budget), plotting=True)

    unlabelled_graphs = []
    # distinct_numpy_edge, edge_graph_dict = graph_feature_generation.normalize_numpy(distinct_numpy_edge, edge_graph_dict)
    unique_labels = labels[unique_indices]
    unique_node_pair_ids = np.asarray(node_pair_ids)[unique_indices]
    print("total number of unique edges:{}".format(distinct_numpy_edge.shape[0]))
    print("total number of unique labels:{}".format(unique_labels.shape[0]))
    if is_edge_wise:
        # current_train_vectors, current_train_class = informative_selection.farthest_first_selection(numpy_edges, labels,
        #                                                                          initial_training)
        only_one_class = True
        while only_one_class:
            seed_index = farthest_first_selection.graipher(distinct_numpy_edge, initial_training)
            # seed_index = np.random.choice(numpy_edges.shape[0], initial_training, replace=False)
            info_train_vectors = distinct_numpy_edge[seed_index]
            if use_gpt == 1:
                train_pair_ids = unique_node_pair_ids[seed_index]
                info_train_class = llm_labeler.prompt_new(model_name, train_pair_ids, entities, considered_atts)
                info_train_class_gt = unique_labels[seed_index]
                ground_truth = unique_labels[seed_index]
                equal_number = np.count_nonzero(info_train_class == ground_truth)
                print(info_train_class)
                print(ground_truth)
                print(equal_number/initial_training)
            elif use_gpt == 2:
                train_pair_ids = unique_node_pair_ids[seed_index]
                info_train_class = unique_labels[seed_index]
                info_train_class_gt = unique_labels[seed_index]
            elif use_gpt == 0:
                info_train_class = unique_labels[seed_index]
                info_train_class_gt = unique_labels[seed_index]
            if 0 < info_train_class.sum() < initial_training:
                only_one_class = False
        if use_gpt == 2:
            suffix = data_set + "_" + str(threshold)
            llm_labeler.fine_tune_model(model_name, train_pair_ids, entities, considered_atts, info_train_class_gt, suffix)
        scatter_pca(info_train_vectors, ['blue' if c == 1 else 'red' for c in info_train_class], 'farthest first')
        if use_gpt:
            unique_node_pair_ids = numpy.delete(unique_node_pair_ids, seed_index, axis=0)
        unlabeled_vectors = np.delete(distinct_numpy_edge, seed_index, axis=0)
        used_budget = initial_training
        unlabeled_classes = np.delete(unique_labels, seed_index, axis=0)
        print("=" * 20 + "training data initialized" + "=" * 20)
    else:
        training_graphs, train_clusters, unlabelled_graphs, unlabelled_clusters \
            = informative_selection.random_selection(graphs_with_features, filtered_clusters, initial_training,
                                                     gold_links)
        edge_counts = [g.number_of_edges() for g in training_graphs]
        used_budget = np.asarray(edge_counts).sum()
    scatter_pca(numpy_edges, labels, input_folder)

    while used_budget < total_budget:
        if is_edge_wise:
            if selection_strategy == 'info':
                new_training_features, new_training_labels, rem_unlabelled_feat, rem_unlabelled_class = \
                    informativeness.informative_selection_edge_wise(info_train_vectors, info_train_class,
                                                                          unlabeled_vectors, unlabeled_classes,
                                                                          increment_budget)
            elif selection_strategy == 'info_opt':
                edges_from_graphs = [graphs_with_features[graph_index] for edge_feature in
                                     info_train_vectors for graph_index in edge_graph_dict[str(edge_feature)]]
                is_plotting = used_budget + increment_budget >= total_budget
                train_distribution, train_bins = distribution_analyis.get_node_distribution(edges_from_graphs,
                                                                                            max_number,
                                                                                            "training sel={} edge_wise={} b={} i={} used={}"
                                                                                            .format(selection_strategy,
                                                                                                    is_edge_wise,
                                                                                                    total_budget,
                                                                                                    increment_budget,
                                                                                                    used_budget),
                                                                                            plotting=is_plotting)
                current_prec = info_train_class.sum()/info_train_class.shape[0]
                new_training_features, new_training_labels, rem_unlabelled_feat, rem_unlabelled_class = \
                    informativeness.informative_selection_edge_wise_opt(info_train_vectors, info_train_class,
                                                                              unlabeled_vectors, unlabeled_classes,
                                                                              increment_budget, edge_graph_dict,
                                                                        graphs_with_features, total_node_dis,
                                                                                     train_distribution, total_bins, current_prec, p)
            elif selection_strategy == 'bootstrap':
                new_training_features, new_training_labels_gt, rem_unlabelled_feat, rem_unlabelled_class, rec_pair_ids = \
                    bootstrap.bootstrap_selection_edge_wise(al_const.DECISION_TREE,
                                                                        info_train_vectors,
                                                                        info_train_class,
                                                                        unlabeled_vectors,
                                                                        unlabeled_classes, 100,
                                                                        increment_budget)
            elif selection_strategy == 'bootstrap_comb':
                edges_from_graphs = [graphs_with_features[graph_index] for edge_feature in
                                     info_train_vectors for graph_index in edge_graph_dict[str(edge_feature)]]
                #is_plotting = used_budget + increment_budget >= total_budget
                is_plotting = True
                train_distribution, train_bins = distribution_analyis.get_node_distribution(edges_from_graphs, max_number,
                                                             "training sel={} edge_wise={} b={} i={} used={}"
                                                            .format(selection_strategy, is_edge_wise, total_budget,
                                                                    increment_budget, used_budget), plotting=is_plotting)
                new_training_features, new_training_labels_gt, rem_unlabelled_feat, rem_unlabelled_class, rec_pair_ids = \
                    bootstrap.bootstrap_cluster_size_selection_edge_wise(al_const.DECISION_TREE,
                                                                        info_train_vectors,
                                                                        info_train_class,
                                                                        unlabeled_vectors,
                                                                        unlabeled_classes, 100,
                                                                        increment_budget, edge_graph_dict,
                                                                        graphs_with_features, total_node_dis,
                                                                                     train_distribution, total_bins)

            info_train_vectors = np.vstack((info_train_vectors, new_training_features))
            if use_gpt:
                print(unique_node_pair_ids.shape)
                print(rem_unlabelled_feat.shape)
                print(rem_unlabelled_class.shape)
                train_pair_ids = unique_node_pair_ids[rec_pair_ids]
                new_training_labels = llm_labeler.prompt_new(model_name, train_pair_ids, entities, considered_atts)
                ground_truth = unique_labels[rec_pair_ids]
                equal_number += np.count_nonzero(new_training_labels == ground_truth)
                unique_node_pair_ids = numpy.delete(unique_node_pair_ids, rec_pair_ids, axis=0)
                print(equal_number / (used_budget+increment_budget))
                info_train_class = np.hstack((info_train_class, new_training_labels))
            else:
                info_train_class = np.hstack((info_train_class, new_training_labels_gt))
            info_train_class_gt = np.hstack((info_train_class_gt, new_training_labels_gt))
            unlabeled_vectors = rem_unlabelled_feat
            unlabeled_classes = rem_unlabelled_class
            used_budget = info_train_vectors.shape[0]
            # scatter_pca(info_train_vectors, info_train_class, input_folder)

        else:
            if selection_strategy == 'info' or selection_strategy == 'info_opt':
                selected_graphs, selected_clusters, unlabelled_graphs, unlabelled_clusters = \
                    informativeness.informative_selection_cluster_wise(training_graphs,
                                                                             gold_links,
                                                                             unlabelled_graphs, unlabelled_clusters,
                                                                             increment_budget)
            elif selection_strategy == 'bootstrap':
                selected_graphs, selected_clusters, unlabelled_graphs, unlabelled_clusters = \
                    bootstrap.bootstrap_selection_cluster_wise(al_const.DECISION_TREE, training_graphs,
                                                                           gold_links,
                                                                           unlabelled_graphs, unlabelled_clusters, 100,
                                                                           increment_budget)
            training_graphs.extend(selected_graphs)
            train_clusters.extend(selected_clusters)
            edge_counts = [g.number_of_edges() for g in selected_graphs]
            used_budget += np.asarray(edge_counts).sum()
    if not is_edge_wise:
        print("training graphs before {}".format(len(training_graphs)))
        new_graphs, new_clusters = informative_selection.graph_augmentation(training_graphs, train_clusters, gold_links,
                                                                            features, is_normalized=False,
                                                                            other_graphs=unlabelled_graphs,
                                                                            expected_hist=hist, bins=bin_edges,
                                                                            distribution_tol=0.1)
        new_cid = 0
        for aug_graph in new_graphs:
            aug_feature_graph = graph_feature_generation.generate_features(aug_graph, new_clusters[new_cid],
                                                                           features)
            _, _, aug_feature_graph = graph_feature_generation.link_category_feature(new_clusters[new_cid],
                                                                                     aug_feature_graph)
            aug_feature_graph = graph_feature_generation.edge_feature_generation(aug_feature_graph, edge_features)
            training_graphs.append(aug_feature_graph)
            new_cid += 1
        print("training graphs after {}".format(len(training_graphs)))
        print("augmented graphs {}".format(len(new_graphs)))
        info_train_vectors, info_train_class = classification.generate_edge_training_data(training_graphs,
                                                                                          gold_links)
    predicted_clusters = []

    print("training data size: {}".format(info_train_vectors.shape))
    print("number of matches in training: {}".format(info_train_class.sum()))
    model = classification.train_by_numpy_edges(info_train_vectors, info_train_class, model_type, False)
    importance = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    all_features = features.copy()
    all_features.extend(['normal_link_ratio', 'strong_link_ratio', 'sim'])
    all_features.extend(edge_features)
    print(all_features)
    print(importance)
    with open(output+"_importance.csv", 'a') as importance_file:
        importance_file.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data_set, is_edge_wise, error_edge_ratio,
                                                                                  selection_strategy, threshold,
                                                                                  total_budget,
                                                                                  increment_budget,
                                                                                  info_train_class.sum() / float(
                                                                                      total_budget),
                                                                                  str(features), str(importance), str(std)))

        importance_file.close()


    # model, encoder = classification.train_by_numpy_edges_nn(info_train_vectors, info_train_class, 100,
    #                                                       dims_enc=[32, 16])
    # reduced_data = encoder.predict(current_train_vectors)
    colors = []
    for c in info_train_class:
        if c == 1:
            colors.append('blue')
        else:
            colors.append('red')
    scatter(info_train_vectors, colors, input_folder)
    for index, cluster in enumerate(filtered_clusters):
        cleaned_clusters = []
        other_graphs = list(graphs_with_features)
        ent_cluster_id_dict = {}
        for e in cluster.entities.values():
            ent_cluster_id_dict[e.iri] = e.properties[famer_constant.REC_ID]
        if gold_links is not None:
            links = metrics.generate_links([cluster])
            _, _, tps_before, _, _, _ = metrics.compute_quality_with_edge_sets(links, gold_links)

            cleaned_clusters = classification.handle_cluster_by_edges_with_support(model, cluster,
                                                                                   graphs_with_features[index])
            # cleaned_clusters = classification.handle_cluster_by_edges(model, cluster, graphs_with_features[index], features,
            #                                                       cleaned_clusters,
            #                                                       recursively=True,
            #                                                       is_normalized=False, other_graphs=other_graphs)
        pred_links = metrics.generate_links(cleaned_clusters)
        _, _, tps_after, _, _, _ = metrics.compute_quality_with_edge_sets(pred_links, gold_links)
        if tps_after < tps_before:
            # print("tps: {} after {}".format(tps_before, tps_after))
            colors = []
            styles = []
            weights = []
            for u, v in graphs_with_features[index].edges():
                weights.append(graphs_with_features[index][u][v]['sim'])
                if tuple(sorted((u, v))) in gold_links and tuple(sorted((u, v))) in pred_links:
                    colors.append('g')
                elif tuple(sorted((u, v))) in gold_links and tuple(sorted((u, v))) not in pred_links:
                    colors.append('y')
                elif tuple(sorted((u, v))) not in gold_links and tuple(sorted((u, v))) not in pred_links:
                    colors.append('b')
                else:
                    colors.append('r')
        predicted_clusters.extend(cleaned_clusters)
    print("number of predicted clusters {}".format(len(predicted_clusters)))
    frequency_dis = metrics.compute_cluster_stats(cluster_list)
    print("uncleaned clusters: " + str(frequency_dis))
    frequency_dis = metrics.compute_cluster_stats(predicted_clusters)
    print("cleaned clusters: " + str(frequency_dis))
    frequency_dis = metrics.compute_cluster_stats(gold_clusters)
    print("gold clusters: " + str(frequency_dis))
    if use_gpt:
        print(equal_number/total_budget)
    num_pred, num_gold, tps, p, r, f = metrics.compute_quality(predicted_clusters, gold_clusters)
    with open(output, 'a') as result_file:
        if not use_gpt:
            result_file.write(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data_set,
                                                                                                  is_edge_wise,
                                                                                                  error_edge_ratio,
                                                                                                  'perfect',
                                                                                                  selection_strategy,
                                                                                                  threshold,
                                                                                                  total_budget,
                                                                                                  increment_budget,
                                                                                                  1.0,
                                                                                                  info_train_class_gt.sum() / float(
                                                                                                      total_budget),
                                                                                                  str(all_features),
                                                                                                  num_gold, tps,
                                                                                                  num_pred - tps,
                                                                                                  num_gold - tps, p, r,
                                                                                                  f))
        else:
            result_file.write(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data_set, is_edge_wise,
                                                                                          error_edge_ratio,
                                                                                          model_name,
                                                                                          selection_strategy, threshold,
                                                                                          total_budget,
                                                                                          increment_budget,
                                                                                          round(equal_number / total_budget, 3),
                                                                                          info_train_class_gt.sum() / float(
                                                                                              total_budget),
                                                                                          str(all_features), num_gold,
                                                                                          tps,
                                                                                          num_pred - tps,
                                                                                          num_gold - tps, p, r, f))

        result_file.close()
    meta_data = {'ds': data_set,
                 'is_edge': is_edge_wise,
                 'error_ratio': error_edge_ratio,
                 'selection': selection_strategy,
                 'budget': total_budget,
                 'batch': increment_budget,
                 'positive': info_train_class.sum() / float(total_budget),
                 'prec': p,
                 'reca': r,
                 'f1': f}
    # training_data_comparison.save_training_data('../../selected_training_data', meta_data, info_train_vectors)
    # training_data_comparison.compare_current_training_data('../../selected_training_data', meta_data, info_train_vectors)
    print("{},{},{},{},{},{}".format(selection_strategy, total_budget, increment_budget, p, r, f))
    num_pred, num_gold, tps, p, r, f = metrics.compute_quality(cluster_list, gold_clusters)
    print("{},{},{},{},{},{}".format(selection_strategy, total_budget, increment_budget, p, r, f))
    num_pred, num_gold, tps, p, r, f = metrics.compute_quality_with_edges(all_edges, gold_clusters)
    with open(output_2, 'a') as result_file:
        result_file.write(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(data_set, threshold, num_gold, tps,
                                                          num_pred - tps,
                                                          num_gold - tps, p, r, f))
        result_file.close()
    print("{},{},{},{},{},{}".format(num_pred, num_gold, tps, p, r, f))


if __name__ == '__main__':
    evaluate()