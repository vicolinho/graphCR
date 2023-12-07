

class Entity(object):

    def __init__(self, iri, resource, cluster_id: str=None, properties: dict=None):
        self.iri = iri
        self.resource = resource
        self.cluster_id = cluster_id
        self.properties = properties

    def __str__(self):
        return '{}, cluster {}, props: {}'.format(self.iri, self.cluster_id, self.properties.keys())

    def __repr__(self):
        return '{}, cluster {}, props: {}'.format(self.iri, self.cluster_id, len(self.properties.keys()))