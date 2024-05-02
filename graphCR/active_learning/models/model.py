
import os
from abc import ABC

class Model(ABC):

    def __init__(self, config):
        pass

    def generate(self, prompt, **kwargs):
        pass

    def unwrap_response(self, response):
        pass

    def unwrap_usage(self, response):
        pass


class TestModel(Model):

    def __init__(self, config):
        self.config = config
        pass

    def generate(self, prompt, **kwargs):
        return kwargs.get("default", '')

    def unwrap_response(self, response):
        return response

    def unwrap_usage(self, response):
        return {'total_tokens': 0, 'prompt_tokens': 0, 'completion_tokens': 0}

