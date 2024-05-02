import os
import json
import re
from collections import defaultdict
import random
from random import sample
from typing import List

import matplotlib.pyplot as plt

from lm3kal.data.cluster import Cluster
from lm3kal.feature_generation import graph_construction

from lm3kal.data.entity import Entity
from lm3kal.data.test_data import famer_constant
import networkx as nx
import csv


def read_data(input_folder: str, error_edge_ratio=0.1):
    file_names = os.listdir(input_folder)
    file_names = sorted(file_names, reverse=True)
    if 'json' in file_names[0]:
        return read_json_famer_data(input_folder, error_edge_ratio)
    else:
        return read_csv_famer_data(input_folder, error_edge_ratio)

def read_data_for_pairs(input_folder: str, source_pair):
    file_names = os.listdir(input_folder)
    file_names = sorted(file_names, reverse=True)
    if 'json' in file_names[0]:
        return read_json_famer_data_source_pairs(input_folder, source_pair)
    else:
        return read_csv_famer_data_for_pairs(input_folder, source_pair)


def read_json_famer_data(input_folder: str, error_edge_ratio=0.1):
    '''
    Reads a gradoop folder with vertex, edge and graph head files and transform them to
    an entity dictionary and a list of graphs representing clusters. The cluster graphs consist
    of edges with similarities.
    :param error_edge_ratio:
    :param input_folder:
    :return: entities: entity dictionary, cluster_graphs: list of graphs
    '''
    file_names = os.listdir(input_folder)
    file_names = sorted(file_names, reverse=True)
    entities = dict()
    edges = list()
    for fn in file_names:
        path = os.path.join(input_folder, fn)
        if "vertices" in fn:
            if os.path.isfile(path):
                entities = read_vertex_file(path, entities)
            else:
                for vf in os.listdir(path):
                    entities = read_vertex_file(os.path.join(path, vf), entities)
        if "edges" in fn:
            if os.path.isfile(path):
                edges = read_edge_file(path, entities, edges)
            else:
                for ef in os.listdir(path):
                    edges = read_edge_file(os.path.join(path, ef), entities, edges)
            edges = pollute_edges(edges, error_edge_ratio)
            graph = graph_construction.build_graph(edges)
    cluster_graphs = [graph.subgraph(c).copy()
                      for c in nx.connected_components(graph)]
    cluster_list = []
    cluster_id = 0
    for cg in cluster_graphs:
        cc_entities = [entities[n] for n in cg.nodes()]
        cluster = Cluster(cc_entities)
        cluster.id = cluster_id
        cluster_id += 1
        cluster_list.append(cluster)
    return entities, cluster_graphs, cluster_list


def pollute_edges(edge_list:list, edge_error_ratio):
    random.shuffle(edge_list)
    dis_edges = edge_list[:round(edge_error_ratio*len(edge_list))]
    noise_edges = []
    for e in dis_edges:
        noise_edges.append((e[0], e[1], random.uniform(0.05, 1)))
    return edge_list[round(edge_error_ratio*len(edge_list)):] + noise_edges

def read_csv_famer_data(input_folder: str, error_edge_ratio=0.1):
    '''
    Reads a gradoop folder with vertex, edge and graph head files in csv format and transform them to
    an entity dictionary and a list of graphs representing clusters. The cluster graphs consist
    of edges with similarities.
    :param input_folder:
    :return: entities: entity dictionary, cluster_graphs: list of graphs
    '''
    meta_data_indices = read_meta_data(os.path.join(input_folder, "metadata.csv"))
    file_names = os.listdir(input_folder)
    file_names = sorted(file_names, reverse=True)
    entities = dict()
    edges = list()
    for fn in file_names:
        path = os.path.join(input_folder, fn)
        if "vertices" in fn:
            if os.path.isfile(path):
                entities = read_vertex_csv_file(path, entities, meta_data_indices)
            else:
                for vf in os.listdir(path):
                    entities = read_vertex_csv_file(os.path.join(path, vf), entities, meta_data_indices)
        if "edges" in fn:
            if os.path.isfile(path):
                edges = read_edge_csv_file(path, entities, edges)
            else:
                for ef in os.listdir(path):
                    edges = read_edge_csv_file(os.path.join(path, ef), entities, edges)
            edges = pollute_edges(edges, error_edge_ratio)
            graph = graph_construction.build_graph(edges)
    cluster_graphs = [graph.subgraph(c).copy()
                      for c in nx.connected_components(graph)]
    cluster_list = []
    cluster_id = 0
    for cg in cluster_graphs:
        cc_entities = [entities[n] for n in cg.nodes()]
        cluster = Cluster(cc_entities)
        cluster.id = cluster_id
        cluster_id += 1
        cluster_list.append(cluster)
    return entities, cluster_graphs, cluster_list

def read_csv_famer_data_for_pairs(input_folder: str, source_pair):
    '''
    Reads a gradoop folder with vertex, edge and graph head files in csv format and transform them to
    an entity dictionary and a list of graphs representing clusters. The cluster graphs consist
    of edges with similarities.
    :param input_folder:
    :return: entities: entity dictionary, cluster_graphs: list of graphs
    '''
    meta_data_indices = read_meta_data(os.path.join(input_folder, "metadata.csv"))
    file_names = os.listdir(input_folder)
    file_names = sorted(file_names, reverse=True)
    entities = dict()
    edges = list()
    for fn in file_names:
        path = os.path.join(input_folder, fn)
        if "vertices" in fn:
            if os.path.isfile(path):
                entities = read_vertex_csv_file(path, entities, meta_data_indices,
                                                set([source_pair[0], source_pair[1]]))
            else:
                for vf in os.listdir(path):
                    entities = read_vertex_csv_file(os.path.join(path, vf), entities, meta_data_indices, set([source_pair[0], source_pair[1]]))
        if "edges" in fn:
            if os.path.isfile(path):
                edges = read_edge_csv_file(path, entities, edges)
            else:
                for ef in os.listdir(path):
                    edges = read_edge_csv_file(os.path.join(path, ef), entities, edges)
            graph = graph_construction.build_graph(edges)
    print("#edges {}".format(graph.number_of_edges()))
    cluster_graphs = [graph.subgraph(c).copy()
                      for c in nx.connected_components(graph)]
    # cc_size = [g.number_of_nodes() for g in cluster_graphs]
    # counts, edges, bars = plt.hist(cc_size, bins='auto')
    # plt.bar_label(bars)
    # plt.title(str(source_pair))
    # plt.show()
    cluster_list = []
    cluster_id = 0
    for cg in cluster_graphs:
        cc_entities = [entities[n] for n in cg.nodes()]
        cluster = Cluster(cc_entities)
        cluster.id = cluster_id
        cluster_id += 1
        cluster_list.append(cluster)
    return entities, cluster_graphs, cluster_list


def read_json_famer_data_source_pairs(input_folder: str, source_pair):
    '''
    Reads a gradoop folder with vertex, edge and graph head files and transform them to
    an entity dictionary and a list of graphs representing clusters. The cluster graphs consist
    of edges with similarities.
    :param source_pair:
    :param input_folder:
    :return: entities: entity dictionary, cluster_graphs: list of graphs
    '''
    file_names = os.listdir(input_folder)
    file_names = sorted(file_names, reverse=True)
    print(file_names)
    entities = dict()
    edges = list()
    for fn in file_names:
        path = os.path.join(input_folder, fn)
        if "vertices" in fn:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    vertex_dict = json.loads(line)
                    if source_pair[0] == vertex_dict[famer_constant.PROPERTIES][famer_constant.RESOURCE] or \
                            source_pair[1] == vertex_dict[famer_constant.PROPERTIES][famer_constant.RESOURCE]:
                        entity = Entity(vertex_dict[famer_constant.ID],
                                        vertex_dict[famer_constant.PROPERTIES][famer_constant.RESOURCE])
                        props: dict = vertex_dict[famer_constant.PROPERTIES]
                        props.pop(famer_constant.RESOURCE)
                        entity.properties = props
                        entities[entity.iri] = entity
            print(len(entities))
        if "edges" in fn:
            edges = read_edge_file(path, entities, edges)
            print(len(edges))
    graph = graph_construction.build_graph(edges)
    cluster_graphs = [graph.subgraph(c).copy()
                      for c in nx.connected_components(graph)]
    cluster_list = []
    cluster_id = 0
    for cg in cluster_graphs:
        cc_entities = [entities[n] for n in cg.nodes()]
        cluster = Cluster(cc_entities)
        cluster.id = cluster_id
        cluster_id += 1
        cluster_list.append(cluster)
    return entities, cluster_graphs, cluster_list


def read_vertex_file(vertex_path, entity_dict):
    srcs = set()
    with open(vertex_path, encoding="utf-8") as f:
        for line in f:
            vertex_dict = json.loads(line)
            entity = Entity(vertex_dict[famer_constant.ID],
                            vertex_dict[famer_constant.PROPERTIES][famer_constant.RESOURCE])
            srcs.add(vertex_dict[famer_constant.PROPERTIES][famer_constant.RESOURCE])
            props: dict = vertex_dict[famer_constant.PROPERTIES]
            props.pop(famer_constant.RESOURCE)
            entity.properties = props
            entity_dict[entity.iri] = entity
    return entity_dict


def read_edge_file(edge_path, entities, edges):
    with open(edge_path) as f:
        for line in f:
            edge_dict = json.loads(line)
            src = edge_dict[famer_constant.SRC]
            target = edge_dict[famer_constant.TARGET]
            if src in entities and target in entities:
                edge = (edge_dict[famer_constant.SRC], edge_dict[famer_constant.TARGET],
                        float(edge_dict[famer_constant.PROPERTIES][famer_constant.SIM]))
                edges.append(edge)
        f.close()
    return edges


def read_meta_data(meta_data_path):
    meta_data_per_source = dict()
    with open(meta_data_path, newline='') as meta_file:
        meta_reader = csv.reader(meta_file, delimiter=';')
        for row in meta_reader:
            if row[0] == 'v':
                att_dict = {}
                attribute_list = row[2].split(',')
                for att_index, att in enumerate(attribute_list):
                    att_name = att.split(':')[0]
                    att_dict[att_index] = att_name
                meta_data_per_source[row[1]] = att_dict
    return meta_data_per_source


def read_vertex_csv_file(vertex_path, entity_dict, meta_dict_per_source, source_pair: set=None):
    srcs = set()
    with open(vertex_path, encoding="utf-8") as f:
        # vertex_reader = csv.reader(f, delimiter=';', escapechar='\\')
        for line in f:
            row = re.split(r'(?<!\\);', line)
            if len(row) > 0:
                att_index_dict = meta_dict_per_source[row[2]]
                if source_pair:
                    if row[2] in source_pair:
                        entity = Entity(row[0],
                                        row[2])
                        srcs.add(row[2])
                        props: dict = {}
                        att_values = re.split(r'(?<!\\)\|', row[3])
                        for index, value in enumerate(att_values):
                            props[att_index_dict[index]] = value
                        props[famer_constant.REC_ID] = props[famer_constant.DEXTER_REC_ID]
                        entity.properties = props
                        entity_dict[entity.iri] = entity
                else:
                    entity = Entity(row[0],
                                    row[2])
                    srcs.add(row[2])
                    props: dict = {}
                    att_values = re.split(r'(?<!\\)\|', row[3])
                    for index, value in enumerate(att_values):
                        props[att_index_dict[index]] = value
                    props[famer_constant.REC_ID] = props[famer_constant.DEXTER_REC_ID]
                    entity.properties = props
                    entity_dict[entity.iri] = entity
    return entity_dict


def read_edge_csv_file(edge_path, entities, edges):
    with open(edge_path, encoding="utf-8") as f:
        edge_reader = csv.reader(f, delimiter=';')
        for row in edge_reader:
            src = row[2]
            target = row[3]
            if src in entities and target in entities:
                edge = (src, target,
                        float(row[5]))
                edges.append(edge)
    return edges


def generate_gold_clusters(entities: dict) -> (List[Cluster]):
    cluster_list = []
    cluster_dict = defaultdict(list)
    for id, e in entities.items():
        entity_list = cluster_dict[e.properties[famer_constant.REC_ID]]
        entity_list.append(e)
    for rec_id, entity_list in cluster_dict.items():
        c = Cluster(entity_list, rec_id)
        cluster_list.append(c)
    return cluster_list


def generate_gold_clusters_subset(subset: List[Cluster], entities: dict) -> (List[Cluster]):
    subset_records = set()
    for c in subset:
        for e in c.entities.values():
            subset_records.add(e.properties[famer_constant.REC_ID])
    cluster_list = []
    cluster_dict = defaultdict(list)
    for id, e in entities.items():
        if e.properties[famer_constant.REC_ID] in subset_records:
            entity_list = cluster_dict[e.properties[famer_constant.REC_ID]]
            entity_list.append(e)
    for rec_id, entity_list in cluster_dict.items():
        c = Cluster(entity_list, rec_id)
        cluster_list.append(c)
    return cluster_list


def determine_class_labels(entities: dict, cluster_graphs: List[nx.Graph]):
    entity_labels = dict()
    for g in cluster_graphs:
        records = defaultdict(list)
        for n in g.nodes():
            e: Entity = entities[n]
            rec_list = records[e.properties[famer_constant.REC_ID]]
            rec_list.append(e)
        top_rec = sorted(records, key=lambda k: len(records[k]), reverse=True)[0]
        for cl_label, rec_list in records.items():
            for e in rec_list:
                if cl_label == top_rec:
                    entity_labels[e.iri] = 1
                else:
                    entity_labels[e.iri] = 0
    return entity_labels


