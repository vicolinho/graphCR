from networkx import Graph
import networkx
import numpy as np
from scipy import stats

from graphCR.data.cluster import Cluster
from graphCR.feature_generation import constant
from sklearn import preprocessing


def generate_features(graph: Graph, cluster, features):
    '''
    generates a feature vector for each vertex of the graph representing a cluster
    :param cluster:
    :param graph:
    :param features: list of features being computed
    :return: graph with featured nodes
    '''
    for u, v in graph.edges():
        if graph[u][v]['sim'] != 1:
            graph[u][v][constant.DISTANCE] = 1 - graph[u][v]['sim']
        else:
            graph[u][v][constant.DISTANCE] = 0.001
    for feature in features:
        node_feature = {}
        if feature == constant.PAGE_RANK:
            node_feature = networkx.pagerank(graph, weight='sim')
        elif feature == constant.BETWEENNESS:
            node_feature = networkx.betweenness_centrality(graph, weight=constant.DISTANCE)
        elif feature == constant.CLOSENESS:
            node_feature = networkx.closeness_centrality(graph, distance=constant.DISTANCE)
        elif feature == constant.CLUSTER_COEFF:
            node_feature = networkx.clustering(graph, weight='sim')
        elif feature == constant.DEGREE:
            for n in graph.nodes():
                degree_weight = networkx.degree(graph, n, weight='sim')
                node_feature[n] = degree_weight
                # degree = networkx.degree(graph, n)
                # if degree != 0:
                #     node_feature[n] = degree_weight/float(degree)
                # else:
                #     node_feature[n] = 0
        elif feature == constant.SIZE:
            for n in graph.nodes():
                node_feature[n] = graph.number_of_nodes()
        elif feature == constant.COMPLETE_RATIO:
            for n in graph.nodes():
                number_of_edges = graph.number_of_edges()
                possible_edges = graph.number_of_nodes() * (graph.number_of_nodes() - 1) / float(2)
                if possible_edges > 1:
                    complete_ratio = number_of_edges / possible_edges
                else:
                    complete_ratio = 0
                node_feature[n] = complete_ratio
        elif feature == constant.UNA_RATIO:
            node_feature = una_violation_ratio(cluster)
        networkx.set_node_attributes(graph, node_feature, feature)
    feature_vec_dict = {}
    feature_data = np.zeros((graph.number_of_nodes(), len(features)))
    for index, feature in enumerate(features):
        node_index = 0
        for n, d in graph.nodes(data=True):
            feature_data[node_index][index] = np.float32(d[feature])
            node_index += 1
    node_index = 0
    for n in graph.nodes():
        feature_vec_dict[n] = feature_data[node_index]
        node_index += 1
    networkx.set_node_attributes(graph, feature_vec_dict, constant.FEATURE_VECTOR)
    return graph


def edge_feature_generation_complete(graph: Graph, edge_features=[]) -> Graph:
    '''
    generates edge features based on the features of the adjacent nodes.
    :param edge_features: features for edges exclusive
    :param graph:
    :return: graph with edge features
    '''
    try:
        feature_vectors = networkx.get_node_attributes(graph, constant.FEATURE_VECTOR)
    except KeyError:
        print("Feature vectors for nodes are missing")
        exit(1)

    for u, v in graph.edges():
        vec_u = feature_vectors[u]
        vec_v = feature_vectors[v]
        vec_edge = vec_u - vec_v
        vec_edge = np.absolute(vec_edge)
        sim = graph[u][v]['sim']
        vec_edge = np.hstack((vec_edge, np.asarray([sim])))
        graph[u][v][constant.FEATURE_VECTOR] = vec_edge

    for f in edge_features:
        if f == constant.BRIDGES:
            bridges = set(networkx.bridges(graph))
            for u, v in graph.edges:
                vec_edge = graph[u][v][constant.FEATURE_VECTOR]
                if (u, v) in bridges:
                    sim = graph[u][v]['sim']
                    vec_edge = np.hstack((np.asarray([sim]), vec_edge))
                else:
                    vec_edge = np.hstack((np.zeros(1), vec_edge))
                graph[u][v][constant.FEATURE_VECTOR] = vec_edge
        elif f == constant.BETWEENNESS:
            edge_centrality = networkx.edge_betweenness_centrality(graph, weight=constant.DISTANCE)
            for u, v in graph.edges:
                vec_edge = graph[u][v][constant.FEATURE_VECTOR]
                if (u, v) in edge_centrality:
                    vec_edge = np.hstack((np.asarray([edge_centrality[(u, v)]]), vec_edge))
                else:
                    vec_edge = np.hstack((np.zeros(1), vec_edge))
                graph[u][v][constant.FEATURE_VECTOR] = vec_edge
        elif f == constant.COMPLETE_RATIO:
            number_of_edges = graph.number_of_edges()
            possible_edges = graph.number_of_nodes() * (graph.number_of_nodes() - 1) / float(2)
            if possible_edges >= 1:
                complete_ratio = number_of_edges / possible_edges
            else:
                complete_ratio = 0
            for u, v in graph.edges:
                vec_edge = graph[u][v][constant.FEATURE_VECTOR]
                vec_edge = np.hstack((np.asarray([complete_ratio]), vec_edge))
                graph[u][v][constant.FEATURE_VECTOR] = vec_edge
    for u in graph.nodes():
        for v in graph.nodes():
            if u != v and not graph.has_edge(u, v):
                graph.add_edge(u, v)
                vec_u = feature_vectors[u]
                vec_v = feature_vectors[v]
                vec_edge = vec_u - vec_v
                vec_edge = np.absolute(vec_edge)
                graph[u][v]['sim'] = 0.01
                vec_edge = np.hstack((vec_edge, np.zeros((1+len(edge_features)))))
                graph[u][v][constant.FEATURE_VECTOR] = vec_edge
    return graph


def edge_feature_generation(graph: Graph, edge_features=[]) -> Graph:
    '''
    generates edge features based on the features of the adjacent nodes.
    :param edge_features: features for edges exclusive
    :param graph:
    :return: graph with edge features
    '''
    try:
        feature_vectors = networkx.get_node_attributes(graph, constant.FEATURE_VECTOR)
    except KeyError:
        print("Feature vectors for nodes are missing")
        exit(1)
    for u, v in graph.edges():
        vec_u = feature_vectors[u]
        vec_v = feature_vectors[v]
        vec_edge = vec_u - vec_v
        vec_edge = np.absolute(vec_edge)
        sim = graph[u][v]['sim']
        vec_edge = np.hstack((vec_edge, np.asarray([sim])))
        graph[u][v][constant.FEATURE_VECTOR] = vec_edge

    for f in edge_features:
        if f == constant.BRIDGES:
            bridges = set(networkx.bridges(graph))
            for u, v in graph.edges:
                vec_edge = graph[u][v][constant.FEATURE_VECTOR]
                if (u, v) in bridges:
                    sim = graph[u][v]['sim']
                    vec_edge = np.hstack((np.asarray([sim]), vec_edge))
                else:
                    vec_edge = np.hstack((np.zeros(1), vec_edge))
                graph[u][v][constant.FEATURE_VECTOR] = vec_edge
        elif f == constant.BETWEENNESS:
            edge_centrality = networkx.edge_betweenness_centrality(graph, weight=constant.DISTANCE)
            for u, v in graph.edges:
                vec_edge = graph[u][v][constant.FEATURE_VECTOR]
                if (u, v) in edge_centrality:
                    vec_edge = np.hstack((np.asarray([edge_centrality[(u, v)]]), vec_edge))
                else:
                    vec_edge = np.hstack((np.zeros(1), vec_edge))
                graph[u][v][constant.FEATURE_VECTOR] = vec_edge
        elif f == constant.COMPLETE_RATIO:
            number_of_edges = graph.number_of_edges()
            possible_edges = graph.number_of_nodes() * (graph.number_of_nodes() - 1) / float(2)
            if possible_edges >= 1:
                complete_ratio = number_of_edges / possible_edges
            else:
                complete_ratio = 0
            for u, v in graph.edges:
                vec_edge = graph[u][v][constant.FEATURE_VECTOR]
                vec_edge = np.hstack((np.asarray([complete_ratio]), vec_edge))
                graph[u][v][constant.FEATURE_VECTOR] = vec_edge

    return graph


def normalize_all_edges(cluster_graphs: list):
    vectors = list()
    index = 0
    for graph in cluster_graphs:
        for u, v in graph.edges():
            vectors.append(graph[u][v][constant.FEATURE_VECTOR])
    vectors = np.asarray(vectors)
    # normalized_vectors = stats.zscore(vectors, axis=1)
    normalized_vectors = preprocessing.normalize(vectors, axis=0)
    for graph in cluster_graphs:
        for u, v in graph.edges():
            graph[u][v][constant.FEATURE_VECTOR] = normalized_vectors[index]
            index += 1
    return cluster_graphs

def normalize_numpy(edges:np.ndarray, edge_graph_index):
    normalized_numpy = edges/np.linalg.norm(edges)
    updated_dictionary = {}
    for i in range(normalized_numpy.shape[0]):
        updated_dictionary[str(normalized_numpy[i])] = edge_graph_index[str(edges[i])]
    return normalized_numpy, updated_dictionary

def normalize_all(cluster_graphs: list):
    vectors = list()
    index = 0
    for graph in cluster_graphs:
        for n, d in graph.nodes(data=True):
            vectors.append(d[constant.FEATURE_VECTOR])
    vectors = np.asarray(vectors)
    normalized_vectors = stats.zscore(vectors, axis=1)
    for graph in cluster_graphs:
        feature_vec_dict = {}
        for n in graph.nodes():
            feature_vec_dict[n] = normalized_vectors[index]
            index += 1
        networkx.set_node_attributes(graph, feature_vec_dict, constant.FEATURE_VECTOR)
    return cluster_graphs


def una_violation_ratio(cluster: Cluster):
    '''
    determines the number of records from the same source regarding a certain one.
    :param cluster:
    :return:
    '''
    node_feature = dict()
    entity_resource_dict = dict()
    for e in cluster.entities.values():
        ent_list = entity_resource_dict.get(e.resource, list())
        ent_list.append(e.iri)
        entity_resource_dict[e.resource] = ent_list
    for ent_list in entity_resource_dict.values():
        for e in ent_list:
            node_feature[e] = 1 / (len(ent_list))
    return node_feature


def link_category_feature(cluster: Cluster, graph: Graph):
    entity_resource_dict = dict()
    disjoint_resources = set()
    for e in cluster.entities.values():
        disjoint_resources.add(e.resource)
        if e.iri not in entity_resource_dict:
            entity_resource_dict[e.iri] = dict()
        resource_dict = entity_resource_dict[e.iri]
        for edge in networkx.edges(graph, e.iri):
            if e.iri == edge[0]:
                other_node = edge[1]
            else:
                other_node = edge[0]
            resource = cluster.entities[other_node].resource
            disjoint_resources.add(resource)
            if resource not in resource_dict:
                resource_dict[resource] = []
            entity_list: list = resource_dict[resource]
            sim = graph.get_edge_data(*edge)['sim']
            entity_list.append((other_node, sim))
        for res, ent_list in resource_dict.items():
            ent_list = sorted(ent_list, key=lambda v: v[1], reverse=True)
            resource_dict[res] = ent_list
    # normal links
    normal_link_count = dict()
    strong_link_count = dict()
    for iri, resource_dict in entity_resource_dict.items():
        current_resource = cluster.entities[iri].resource
        for other_res, ent_list in resource_dict.items():
            max_sim = ent_list[0][1]
            found_strong = False
            for i in range(len(ent_list)):
                if max_sim != ent_list[i][1]:
                    break
                else:
                    other_ent_list = entity_resource_dict[ent_list[i][0]][current_resource]
                    other_max_sim = other_ent_list[0][1]
                    if other_max_sim == max_sim:
                        for k in range(len(other_ent_list)):
                            if other_max_sim != other_ent_list[k][1]:
                                break
                            else:
                                if iri == other_ent_list[k][0]:
                                    found_strong = True
                                    break
                    else:
                        normal_link_count[iri] = normal_link_count.get(iri, 0) + 1
                        normal_link_count[ent_list[i][0]] = \
                            normal_link_count.get(ent_list[i][0], 0) + 1
                    if found_strong:
                        break
            if found_strong:
                strong_link_count[iri] = strong_link_count.get(iri, 0) + 1
        feature_comb = dict()
    for n, data in graph.nodes(data=True):
        feature_vec = data.get(constant.FEATURE_VECTOR, np.asarray([]))
        link_vec = np.asarray([normal_link_count.get(n, 0) / len(disjoint_resources),
                               strong_link_count.get(n, 0) / len(disjoint_resources)])
        comb_feature = np.hstack((feature_vec, link_vec))
        feature_comb[n] = comb_feature
    networkx.set_node_attributes(graph, feature_comb, constant.FEATURE_VECTOR)
    return normal_link_count, strong_link_count, graph


def matrix_root(m: np.array, inverse: bool = False) -> np.array:
    eig, base = np.linalg.eig(m)
    if not all(e > 0 for e in eig):
        raise ValueError("Matrix is not positive definite")
    if not inverse:
        root_matrix = np.matmul(np.matmul(base, np.diag(np.sqrt([e for e in eig]))), base.T)
    else:
        root_matrix = np.matmul(np.matmul(base, np.diag(np.sqrt([1 / e for e in eig]))), base.T)
    return root_matrix


def normalize(data, mean_cov=None, inverse=False, return_mean_cov=False):
    if mean_cov is not None:
        mean, cov = mean_cov
    else:
        mean, cov = (np.mean(data, axis=0, dtype=np.float32), np.cov(data, rowvar=False))
    if not inverse:
        data = data - mean
        t = matrix_root(cov, inverse=True)
        data = np.matmul(data, t)
    else:
        t = matrix_root(cov, inverse=False)
        data = np.matmul(data, t)
        data = data + mean
    if return_mean_cov:
        return data, (mean, cov)
    else:
        return data
