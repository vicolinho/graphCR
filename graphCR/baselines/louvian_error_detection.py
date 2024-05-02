from typing import List

from networkx import Graph

import networkx
from lm3kal.data.cluster import Cluster


def classify_edges_clusters(cluster_graphs: List[Graph], clusters: List[Cluster]):
    intra_edges = {}
    inter_edges = {}
    measure_values_intra = {}
    measure_values_inter = {}

    for index, cg in enumerate(cluster_graphs):
        classifiy_edges(cg)


def classifiy_edges(graph, cluster, error_threshold=0.9, weight_label='weight'):
    for u, v in graph.edges():
        if weight_label == 'weight':
            graph[u][v]['weight'] = 2
        else:
            graph[u][v]['weight'] = graph[u][v][weight_label] * 2
    resulting_clusters = networkx.community.louvain_communities(graph, weight=weight_label)
    intra_error_edges = {}
    inter_error_edges = {}
    node_id_cluster = {}
    cluster_id = 0
    cluster_graph_dict = {}
    for c in resulting_clusters:
        cluster_graph = networkx.subgraph(graph, [n for n in c])
        cluster_graph_dict[cluster_id] = cluster_graph
        agg_sum = 0
        for u, v in cluster_graph.edges():
            agg_sum += graph[u][v]['weight']
        for u, v in cluster_graph.edges():
            intra_error_edges[(u, v)] = 1.0 / cluster_graph[u][v]['weight'] * \
                                        (1.0 - agg_sum /
                                         float(cluster_graph.number_of_nodes() * (cluster_graph.number_of_nodes() - 1)))
            # print(str(agg_sum) + "--" + str(float(cluster_graph.number_of_nodes() * (cluster_graph.number_of_nodes() - 1))))
            # print(1.0 / cluster_graph[u][v]['weight'])
            #assert 0 <= intra_error_edges[(u, v)] <= 1, intra_error_edges[(u, v)]
        for n in c:
            node_id_cluster[n] = cluster_id
        cluster_id += 1
    number_cluster_inter_edges = {}
    for u, v in graph.edges():
        cluster_id_u = node_id_cluster[u]
        cluster_id_v = node_id_cluster[v]
        cluster_pair_id = tuple(sorted([cluster_id_u, cluster_id_v]))
        if cluster_id_u != cluster_id_v:
            if cluster_pair_id not in number_cluster_inter_edges:
                number_cluster_inter_edges[cluster_pair_id] = set()
            edge_set = number_cluster_inter_edges[cluster_pair_id]
            edge_set.add(tuple(sorted([u, v])))
    agg_sum = {}
    for c_pair_id, edges in number_cluster_inter_edges.items():
        sum_weight = 0
        for u_other, v_other in edges:
            sum_weight += graph[u_other][v_other]['weight']
        agg_sum[c_pair_id] = sum_weight
    for u, v in graph.edges():
        cluster_id_u = node_id_cluster[u]
        cluster_id_v = node_id_cluster[v]
        cluster_pair_id = tuple(sorted([cluster_id_u, cluster_id_v]))
        if cluster_id_u != cluster_id_v:
            u_cluster_nodes = cluster_graph_dict[cluster_id_u].number_of_nodes()
            v_cluster_nodes = cluster_graph_dict[cluster_id_v].number_of_nodes()
            sum_weight = agg_sum[cluster_pair_id]
            inter_error_edges[(u, v)] = 1.0 / graph[u][v]['weight'] * (1.0 - sum_weight /
                                                                           (2 * u_cluster_nodes * v_cluster_nodes))
    remove_links = []
    for e, error in intra_error_edges.items():
        if error > error_threshold:
            remove_links.append(e)
    for e, error in inter_error_edges.items():
        if error > error_threshold:
            remove_links.append(e)
    changed_graph = graph.copy()
    changed_graph.remove_edges_from(remove_links)
    ccs = networkx.connected_components(changed_graph)
    cid = 0
    cleaned_clusters = []
    for cc in ccs:
        entities = []
        for e_id in cc:
            entities.append(cluster.entities[e_id])
        new_cluster = Cluster(entities, str(cid))
        cid += 1
        cleaned_clusters.append(new_cluster)
    return intra_error_edges, inter_error_edges, node_id_cluster, cleaned_clusters
