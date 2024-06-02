from collections import defaultdict
from math import floor
from typing import List, Optional

from networkx import Graph
from sklearn import tree, ensemble, svm
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, RepeatedStratifiedKFold, RepeatedKFold

from graphCR.evaluation.quality import metrics
from graphCR.feature_generation import constant, graph_feature_generation
from graphCR.active_learning import constant as al_const
import numpy as np
import networkx as nx
from graphCR.data.cluster import Cluster
import pickle



def generate_training_data(graph_list: List[Graph], classified_nodes=None):
    labels = list()
    features = list()
    for g in graph_list:
        for n, d in g.nodes(data=True):
            features.append(d[constant.FEATURE_VECTOR])
            if classified_nodes is not None:
                labels.append(classified_nodes[n])
    features = np.asarray(features)
    labels = np.asarray(labels)
    return features, labels


def generate_edge_training_data(graph_list: List[Graph], gold_links: set = None):
    '''
    :param graph_list:
    :param gold_links:
    :return:
    '''
    labels = list()
    features = list()
    node_pair_ids = list()
    edge_feature_graph_dict = dict()
    for gid, g in enumerate(graph_list):
        for u, v, d in g.edges(data=True):
            features.append(d[constant.FEATURE_VECTOR])
            node_pair_ids.append((u, v))
            if str(d[constant.FEATURE_VECTOR]) not in edge_feature_graph_dict:
                edge_feature_graph_dict[str(d[constant.FEATURE_VECTOR])] = [gid]
            else:
                graph_ids = edge_feature_graph_dict[str(d[constant.FEATURE_VECTOR])]
                graph_ids.append(gid)
                edge_feature_graph_dict[str(d[constant.FEATURE_VECTOR])] = graph_ids
            if gold_links is not None:
                if (u, v) in gold_links or (v, u) in gold_links:
                    labels.append(1)
                else:
                    labels.append(0)
    features = np.asarray(features)
    labels = np.asarray(labels)
    return features, labels, edge_feature_graph_dict, node_pair_ids


def get_model(model_type):
    if al_const.DECISION_TREE == model_type:
        model = tree.DecisionTreeClassifier()
    elif al_const.RANDOM_FOREST == model_type:
        model = ensemble.RandomForestClassifier()
    elif al_const.SVM == model_type:
        model = svm.SVC()
    return model



def train_by_numpy_edges(features, labels, model_type, save_model=False,
                         save_model_path=None, **kwargs):
    model = None
    random_grid = dict()
    if al_const.DECISION_TREE == model_type:
        model = tree.DecisionTreeClassifier()
        criterion = ['gini', 'entropy']
        max_depth = [2, 4, 6, 8, 10, 12]
        min_samples_split = [int(x) for x in
                     np.linspace(2, 30, num=6)]  # minimum sample number to split a node
        min_samples_leaf = [2, 3, 4]  # minimum sample number that can be stored in a leaf node
        random_grid = {'criterion': criterion,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf
                       }
    elif al_const.RANDOM_FOREST == model_type:
        n_estimators = [int(x) for x in np.linspace(start=1, stop=20, num=20)]  # number of trees in the random forest
        max_features = ['sqrt']  # number of features in consideration at every split
        max_depth = [int(x) for x in
                     np.linspace(2, 20, num=10)]  # maximum number of levels allowed in each decision tree
        min_samples_split = [int(x) for x in
                     np.linspace(2, 30, num=6)]  # minimum sample number to split a node
        min_samples_leaf = [2, 3, 4]  # minimum sample number that can be stored in a leaf node
        bootstrap = [True, False]  # method used to sample data points
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        model = ensemble.RandomForestClassifier()
    elif al_const.SVM == model_type:
        model = svm.SVC()
        random_grid = {"C": np.arange(2, 10, 2),
                       "gamma": np.arange(0.1, 8, 0.4),
                       "kernel": ['rbf', 'sigmoid']}
    cv = RepeatedStratifiedKFold(n_repeats=5, random_state=35)
    rf_random = RandomizedSearchCV(model, random_grid, scoring='f1', n_iter=100, cv=cv,
                                   verbose=0, random_state=35, n_jobs=-1)
    # scores = cross_val_score(model, features, labels, cv=5, scoring='f1_macro')
    result = rf_random.fit(features, labels)
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    if save_model:
        pickle.dump(result.best_estimator_, save_model_path, compress=1)
    return result.best_estimator_


def train_by_edges(graph_list: List[Graph], gold_links, model_type, save_model=False,
                   folds=-1, save_model_path=None, **kwargs):
    features, labels = generate_edge_training_data(graph_list, gold_links)
    if folds != -1:
        features = features[:floor(features.shape[0] / folds) * (folds - 1)]
        labels = labels[:floor(labels.shape[0] / folds) * (folds - 1)]
    return train_by_numpy_edges(features, labels, model_type, save_model)


def train(graph_list: List[Graph], classified_nodes: dict, model_type, save_model=False,
          save_model_path=None, **kwargs):
    features, labels = generate_training_data(graph_list, classified_nodes)
    model = None
    random_grid = dict()
    if al_const.DECISION_TREE == model_type:
        model = tree.DecisionTreeClassifier()
        criterion = ['gini', 'entropy']
        max_depth = [2, 4, 6, 8, 10, 12]
        min_samples_split = [2, 6, 10]  # minimum sample number to split a node
        min_samples_leaf = [1, 3, 4]  # minimum sample number that can be stored in a leaf node
        random_grid = {'criterion': criterion,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf
                       }
    elif al_const.RANDOM_FOREST == model_type:
        n_estimators = [int(x) for x in np.linspace(start=1, stop=20, num=20)]  # number of trees in the random forest
        max_features = ['sqrt']  # number of features in consideration at every split
        max_depth = [int(x) for x in
                     np.linspace(2, 40, num=12)]  # maximum number of levels allowed in each decision tree
        min_samples_split = [2, 6, 10]  # minimum sample number to split a node
        min_samples_leaf = [1, 3, 4]  # minimum sample number that can be stored in a leaf node
        bootstrap = [True, False]  # method used to sample data points
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        model = ensemble.RandomForestClassifier()
    elif al_const.SVM == model_type:
        model = svm.SVC()
        random_grid = {"C": np.arange(2, 10, 2),
                       "gamma": np.arange(0.1, 8, 0.4),
                       "kernel": ['rbf', 'sigmoid']}
    cv = RepeatedStratifiedKFold(n_repeats=3, random_state=35)
    rf_random = RandomizedSearchCV(model, random_grid, n_iter=100, cv=cv,
                                   verbose=0, random_state=35, n_jobs=-1)
    # scores = cross_val_score(model, features, labels, cv=5, scoring='f1_macro')
    result = rf_random.fit(features, labels)
    print('Best Score: %s' % result.best_score_)
    # print('Best Hyperparameters: %s' % result.best_params_)
    if save_model:
        pickle.dump(result.best_estimator_, save_model_path, compress=1)
    return result.best_estimator_


def get_conflict_nodes(cluster: Cluster):
    res_entity_dict = defaultdict(list)
    for e in cluster.entities.values():
        ent_list = res_entity_dict[e.resource]
        ent_list.append(e.iri)
        res_entity_dict[e.resource] = ent_list
    conflicts = []
    for conf in res_entity_dict.values():
        if len(conf) > 1:
            conflicts.append(conf)
    return conflicts


def handle_cluster_by_edges_with_support(model, cluster: Cluster, graph: Graph, cleaned_clusters=[],
                            recursively=False, is_normalized=False, other_graphs=[]):
    confl_edges = []
    confl_nodes = set()
    clusters = {}
    new_graphs = {}
    edge_features = []
    for u, v in graph.edges():
        edge_features.append(graph[u][v][constant.FEATURE_VECTOR])
    classified = model.predict(np.asarray(edge_features))
    index = 0
    positive_edges = set()
    for u, v in graph.edges:
        label = classified[index]
        if label < 0.5:
            confl_edges.append((u, v))
            confl_nodes.add(u)
            confl_nodes.add(v)
        else:
            positive_edges.add((u, v))
        index += 1
    for r in confl_nodes:
        c = Cluster([cluster.entities[r]])
        c.id = r
        clusters[c.id] = c
        new_graph = Graph()
        new_graph.add_node(r)
        new_graphs[c.id] = new_graph
    if len(confl_nodes) == 0:
        return [cluster]
    change = True
    support = {}
    element_cluster_dict = {}
    for n in graph.nodes():
        if n in confl_nodes:
            element_cluster_dict[n] = n
            support[n] = {n: 0}
        else:
            element_cluster_dict[n] = -1
    while change:
        change = False
        for cid, c in clusters.items():
            unprocessed = [e for e in c.entities.values()]
            processed = set()
            for ent in unprocessed:
                r = ent.iri
                adjacent_edges = nx.edges(graph, r)
                for e in adjacent_edges:
                    if e not in confl_edges:
                        if e[0] == r:
                            n = e[1]
                        else:
                            n = e[0]
                        if n not in processed:
                            if cid not in support:
                                sup_c = {}
                                support[cid] = sup_c
                            sup_c = support[cid]
                            if n not in sup_c:
                                sup = compute_support(n, c, graph, positive_edges, confl_edges)
                                sup_c[n] = sup
                            if sup_c[n] > 0:
                                old_cluster = element_cluster_dict[n]
                                if old_cluster != -1:
                                    old_sup = support[old_cluster][n]
                                    if sup_c[n] > old_sup:
                                        element_cluster_dict[n] = cid
                                        clusters[old_cluster].remove(cluster.entities[n])
                                        c.entities[n] = cluster.entities[n]
                                        del support[old_cluster][n]
                                        change = True
                                else:
                                    change = True
                                    c.entities[n] = cluster.entities[n]
                                    element_cluster_dict[n] = cid
                    processed.add(n)
    return [c for c in clusters.values()]


def compute_support(n, cluster:Cluster, g:Graph, positive_edges, confl_edges):
    sup = 0
    for r in cluster.entities.keys():
        if g.has_edge(r, n) or g.has_edge(n, r):
            if (r, n) in positive_edges or (n, r) in positive_edges:
                sup += 1
            elif (r, n) in confl_edges or (n, r) in confl_edges:
                sup -= 1
    return sup


def predict_edges(model, graph: Graph):
    removed_edges = []
    edge_features = []
    for u, v in graph.edges():
        edge_features.append(graph[u][v][constant.FEATURE_VECTOR])
    classified = model.predict(np.asarray(edge_features))
    index = 0
    for u, v in graph.edges:
        if isinstance(model, ClassifierMixin):
            if classified[index] == 0:
                removed_edges.append((u, v))
        else:
            if classified[index][0] < 0.5:
                removed_edges.append((u, v))
        index += 1
    return removed_edges

