import os

from lm3kal.active_learning.models.anthropic import AnthropicModel
from lm3kal.active_learning.models.model import Model, TestModel
from lm3kal.active_learning.models.openai import OpenAIModel

open_ai_key = os.getenv('OPENAI_API_KEY')
anthropic_key = os.getenv('ANTHROPIC_API_KEY')

def getModel(model_name) -> Model:
    if 'gpt' in model_name:
        print(open_ai_key)
        return OpenAIModel(open_ai_key, model_name)
    elif model_name.startswith('claude'):
        return AnthropicModel(anthropic_key, model_name)
    elif model_name == 'test':
        return TestModel
    else:
        None
