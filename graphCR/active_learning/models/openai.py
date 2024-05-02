###OpenAI Model###
import os
import time

from lm3kal.active_learning.models.model import Model
import openai
from openai import OpenAI

CHAT_COMPLETION_TYPE = 0
COMPLETION_TYPE = 1

model_map = {
    "gpt-3.5-turbo-instruct": COMPLETION_TYPE,
    "gpt-3.5-turbo": CHAT_COMPLETION_TYPE,
    "gpt-4": CHAT_COMPLETION_TYPE,
    "gpt-4-turbo": CHAT_COMPLETION_TYPE,
    "gpt-4-0125-preview": CHAT_COMPLETION_TYPE
}


class OpenAIModel(Model):

    def __init__(self, api_key, model_name):
        self.api_key = api_key
        self.model_name = model_name
        # self.client = OpenAI(api_key=api_key)
        self.fine_tuned_model_id = None

    def generate(self, prompt, **kwargs):
        # if (model_map[self.model_name] == CHAT_COMPLETION_TYPE):
            openai.api_key = self.api_key
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            return response
        # else:
        #     openai.api_key = self.api_key
        #     response = openai.completions.create(
        #         model=self.model_name,
        #         prompt=prompt,
        #         **kwargs
        #     )
        #     return response

    def unwrap_response(self, response) -> str:
        return response.choices[0].message.content

    def unwrap_usage(self, response):
        return {}

    def fine_tune(self, training_file, validation_file, suffix):
        fine_tuning_job = self.client.fine_tuning.jobs.create(model=self.model_name,
                                                              training_file=training_file,
                                                              validation_file=validation_file,
                                                              hyperparameters={'n_epochs': 5,
                                                              'batch_size': 4}, suffix= suffix

                                                              )
        # openai.FineTune.create(
        #     model_engine=self.model_name,
        #     n_epochs=20,
        #     batch_size=4,
        #     learning_rate=1e-05,
        #     max_tokens=2048,
        #     training_file=os.path.abspath(training_file),
        #     validation_file=os.path.abspath(validation_file),
        # )
        job_id = fine_tuning_job["id"]
        print(f"Fine-tuning job created with ID: {job_id}")
        while True:
            fine_tuning_status = openai.FineTune.get_status(job_id)
            status = fine_tuning_status["status"]
            print(f"Fine-tuning job status: {status}")

            if status in ["completed", "failed"]:
                break
            time.sleep(60)
        fine_tuned_model_id = fine_tuning_status["fine_tuned_model_id"]
        self.fine_tuned_model_id = fine_tuned_model_id

