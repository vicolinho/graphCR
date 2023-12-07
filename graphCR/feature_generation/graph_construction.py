from typing import List

import networkx
from networkx import Graph, DiGraph

from graphCR.data.cluster import Cluster


def build_graph(sim_edges) -> Graph:
    g = Graph()
    for u, v, sim in sim_edges:
        g.add_edge(u, v, sim=sim)
        # reverse since pagerank uses directed graphs
        g.add_edge(v, u, sim=sim)
    return g


def filter_links(cluster: Cluster, graph: Graph, types=['normal', 'strong']):
    '''
    filters all weak links in a graph and build new clusters and graphs based on the connected components of the
    resulting graph
    :param types:
    :param cluster:
    :param graph:
    :return: updated_clusters, updated_graphs
    '''
    type_set = set(types)
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
    strong_normal_edges = set()
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
                                    strong_edge = (iri, ent_list[i][0])
                                    rev_strong_edge = (ent_list[i][0], iri)
                                    strong_normal_edges.add(strong_edge)
                                    strong_normal_edges.add(rev_strong_edge)
                                    # found_strong = True
                                    #break
                    elif 'normal' in type_set:
                        strong_normal_edges.add((iri, ent_list[i][0]))
                        strong_normal_edges.add((ent_list[i][0], iri))
                    if found_strong:
                        break
    rem_edges = []
    for u, v in graph.edges():
        if (u, v) not in strong_normal_edges:
            rem_edges.append((u, v))
    rem_graph = graph.copy()
    rem_graph.remove_edges_from(rem_edges)
    subgraphs = [rem_graph.subgraph(c).copy()
                 for c in networkx.connected_components(rem_graph)]
    # print(len(removed_graph))
    # print(len(subgraphs))
    updated_clusters = []
    updated_graphs = []
    for sg in subgraphs:
        entities = [cluster.entities[n] for n in sg.nodes()]
        new_cluster = Cluster(entities)
        updated_graphs.append(sg)
        updated_clusters.append(new_cluster)
    return updated_clusters, updated_graphs