from typing import List

from graphCR.data.entity import Entity


class Cluster:


    def __init__(self, entities: List[Entity]=[], cluster_iri=None):
        self.entities = dict()
        for e in entities:
            self.entities[e.iri] = e
        self.cluster_iri = cluster_iri
        self.id = 0
        self.una_violations = -1
        self.number_of_records = len(self.entities)

    def remove(self, entity: Entity):
        self.entities.pop(entity.iri)

    def remove_by_iri(self, iri: str):
        self.entities.pop(iri)

    def __repr__(self):
        return str(self.id)+": "+str([str(k) for k in self.entities.keys()])

    def copy(self):
        c = Cluster(list(self.entities.values()), self.cluster_iri)
        c.id = self.id
        c.number_of_records = self.number_of_records
        c.una_violations = self.una_violations
        return c

