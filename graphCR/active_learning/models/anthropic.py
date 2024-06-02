###Anthropic Model###
from anthropic import Anthropic
import anthropic
import itertools

# https://docs.anthropic.com/claude/reference/selecting-a-model
from graphCR.active_learning.models.model import Model

DEFAULT_PARAMETERS = {
    'max_tokens_to_sample': 4096,
    'stop_sequences': [anthropic.HUMAN_PROMPT]
}


class AnthropicModel(Model):
    def __init__(self, api_key, model_name):
        self.model_name = model_name
        self.client = Anthropic(auth_token=api_key)

    def generate(self, prompt, **kwargs):

        # messages = "".join([entry[0]+entry[1] for entry in zip(itertools.cycle([anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT]), prompt)]) + anthropic.AI_PROMPT
        # message= f'{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}'
        messages = [{"role": "user", "content": prompt}]
        # print("MESSAGES: "+message)
        response = self.client.messages.create(
            model=self.model_name,
            # prompt=message
            messages=messages,
            # max_tokens_to_sample=4096,
            max_tokens=4096,
            **kwargs
        )
        return response

    def unwrap_response(self, response) -> str:
        content = response.content
        if len(content) == 0:
            return ""
        else:
            return response.content[0].text

    def unwrap_usage(self, response):
        return {}
