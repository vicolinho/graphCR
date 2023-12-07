import collections
from typing import List

from graphCR.data.cluster import Cluster

def generate_links(cluster_list:List[Cluster]):
    mappings = set()
    for c in cluster_list:
        ents = list(c.entities.keys())
        ents = sorted(ents)
        for i in range(len(ents)):
            for k in range(i+1, len(ents)):
                mappings.add(tuple(sorted((ents[i], ents[k]))))
    return mappings

def compute_quality(cluster_list:List[Cluster], reference_cluster_list:List[Cluster]):
    predicted_mappings = generate_links(cluster_list)
    gold_mappings = generate_links(reference_cluster_list)
    tps = predicted_mappings.intersection(gold_mappings)
    prec = len(tps)/float(len(predicted_mappings))
    rec = len(tps)/float(len(gold_mappings))
    f1 = 2*prec*rec/(prec+rec)
    return len(predicted_mappings), len(gold_mappings), len(tps), prec, rec, f1


def compute_quality_with_edges(all_links: List[tuple], reference_cluster_list: List[Cluster]):
    predicted_mappings = set(all_links)
    gold_mappings = generate_links(reference_cluster_list)
    tps = predicted_mappings.intersection(gold_mappings)
    print(len(tps))
    prec = len(tps)/float(len(predicted_mappings))
    rec = len(tps)/float(len(gold_mappings))
    f1 = 0
    if prec + rec > 0:
        f1 = 2 * prec * rec / (prec + rec)
    return len(predicted_mappings), len(gold_mappings), len(tps), prec, rec, f1

def compute_quality_with_edge_sets(all_links, gold_mappings):
    predicted_mappings = set(all_links)
    tps = predicted_mappings.intersection(gold_mappings)
    prec = 0
    if len(predicted_mappings) > 0:
        prec = len(tps)/float(len(predicted_mappings))
    rec = len(tps)/float(len(gold_mappings))
    f1 = 0
    if prec + rec > 0:
        f1 = 2*prec*rec/(prec+rec)
    return len(predicted_mappings), len(gold_mappings), len(tps), prec, rec, f1


def compute_cluster_stats(cluster_list:List[Cluster]):
    counts = [len(c.entities) for c in cluster_list]
    counter = collections.Counter(counts)
    return counter



