import json
import sys
import re
import time

import numpy
from openai import OpenAI
from sklearn.model_selection import train_test_split
from graphCR.active_learning.models.model_factory import getModel

from graphCR.data.test_data import reader


class LLMLabeling:

    def __init__(self, api_key, cache_file=None):
        self.cache_file = cache_file
        if cache_file is not None:
            self.cache = self.read_cache_file(cache_file)
        else:
            self.cache = {}
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

    def read_cache_file(self, cache_file):
        cache = {}
        try:
            with open(cache_file, encoding="utf-8") as f:
                for l in f:
                    values = re.split(',', l)
                    cache[(values[0], values[1])] = int(values[2])
                f.close()
        except FileNotFoundError as e:
            pass
        return cache

    def write_cache_file(self, cache_file):
        old_cache = self.read_cache_file(cache_file)
        with open(cache_file, mode='a', encoding="utf-8") as f:
            for p, label in self.cache.items():
                if (p[0], p[1]) not in old_cache:
                    f.write('{},{},{}\n'.format(p[0], p[1], label))
            f.close()



    def handle_response(self, response):
        if "yes" in response.lower():
            return 1
        else:
            return 0

    def fine_tune_model(self, model_name, rec_pairs, entities, considered_atts, ground_truth_labels, suffix):
        '''
        generate training and validation json files to fine tune an openAI llm model. The generated files
        can be used in the web application.
        :param model_name:
        :param rec_pairs: sample of record pairs represented with the ids
        :param entities: dictionary of records
        :param considered_atts: attributes being used to specify the records
        :param ground_truth_labels: labels given by a user for the certain record pairs
        :param suffix: suffix of the model name
        :return:
        '''
        model = getModel(model_name)
        json_list = []
        print(len(rec_pairs))
        for index, p in enumerate(rec_pairs):
            t = tuple((p[0], p[1]))
            properties1 = {}
            for k, v in entities[p[0]].properties.items():
                if len(considered_atts) == 0 and len(v.strip()) > 0:
                    properties1[k] = v
                elif len(v.strip()) > 0 and k in considered_atts:
                    properties1[k] = v
            properties2 = {}
            for k, v in entities[p[1]].properties.items():
                if len(considered_atts) == 0 and len(v.strip()) > 0:
                    properties2[k] = v
                elif len(v.strip()) > 0 and k in considered_atts:
                    properties2[k] = v
            rec1 = json.dumps(properties1)
            rec2 = json.dumps(properties2)
            print(rec1)
            print(rec2)
            answer = ground_truth_labels[index]
            if answer == 1:
                answer_text = "yes"
            else:
                answer_text = "no"
            msg = [{"role": "system", "content": "You are a helpful assistant for classifying record pairs as the same entity or"
                                                 " not. Therefore, you consider the attributes of both records "
                                                 "represented as JSON objects and decide on the content if they are the "
                                                 "same entity or not. Be aware that the content of a record can slightly "
                                                 "differ due to quality issues such as typos or missing values. \n "
                                                 "Output: yes/no\n"
                                                 "### Entity A: \n"
                                                 "### Entity B:"}, {"role": "user", "content": "Entity A: {} \n Entity B: {}".format(rec1, rec2)},
                       {"role": "assistant", "content": answer_text}]
            json_train_data = json.dumps({"messages": msg})
            json_list.append(json_train_data)
        both_classes = False
        iters = 0
        if len(ground_truth_labels) > ground_truth_labels.sum() > 0:
            while not both_classes or iters < 10:
                X_train, X_test, y_train, y_test = train_test_split(json_list,
                                                                    ground_truth_labels,
                                                                    test_size=0.5, random_state=42)
                if 0 < y_train.sum() < len(y_train) and 0 < y_test.sum() < len(y_test):
                    both_classes = True
                iters+=1

        with open("train.jsonl", 'w') as f:
            for json_msg in X_train:
                f.write(json_msg +"\n")
            f.close()

        with open("validation.jsonl", 'w') as f:
            for json_msg in X_test:
                f.write(json_msg + "\n")
            f.close()




    def prompt_new(self, model_name, rec_pairs, entities, considered_atts):
        train_classes = []

        model = getModel(model_name)

        for index, p in enumerate(rec_pairs):
            t = tuple((p[0], p[1]))
            if t not in self.cache:
                properties1 = {}
                for k, v in entities[p[0]].properties.items():
                    if len(considered_atts) == 0 and len(v.strip()) > 0:
                        properties1[k] = v
                    elif len(v.strip()) > 0 and k in considered_atts:
                        properties1[k] = v
                properties2 = {}
                for k, v in entities[p[1]].properties.items():
                    if len(considered_atts) == 0 and len(v.strip()) > 0:
                        properties2[k] = v
                    elif len(v.strip()) > 0 and k in considered_atts:
                        properties2[k] = v
                rec1 = json.dumps(properties1)
                rec2 = json.dumps(properties2)
                print(rec1)
                print(rec2)
                prompt = self.build_prompt(rec1, rec2)
                completion = model.generate(prompt) #, max_tokens=50)
                is_match = self.handle_response(model.unwrap_response(completion))
                self.cache[t] = str(is_match)
                train_classes.append(is_match)
            else:
                train_classes.append(int(self.cache[t]))
        self.write_cache_file(self.cache_file)
        return numpy.asarray(train_classes)

    def build_prompt(self, entity_a, entity_b):
        return """You are a helpful assistant for classifying record pairs as the same entity or not. 
        Therefore, you consider the attributes of both records represented as JSON objects and decide on the content 
        if they are the same entity or not. 
        Be aware that the content of a record can slightly differ due to quality issues such as typos or missing values.
Output: yes/no

### Entity A:
{}

### Entity B:
{}""".format(entity_a, entity_b)
        
if __name__ == '__main__':
    args = sys.argv[1:]
    dexter_considered_atts = {"famer_product_name"
                              "famer_model_list", "famer_model_no_list",
                              "famer_brand_list", "famer_keys", "<page title>", "optical zoom", "digital zoom",
                              "resolution",
                              "camera dimension"}
    entities, cluster_graphs, cluster_list = reader.read_data('E:/data/DS-C/DS-C/DS-C0/SW_0.7', 0)
    llm_labeler = LLMLabeling(args[0], 'requested_pairs.csv')
    llm_labeler.prompt_new('claude-3-opus-20240229', [('5f3ba6203fd5b671b22037e3', '5f3ba6203fd5b671b2202582')], entities,
                           dexter_considered_atts)

