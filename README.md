<h2 align="center"> Graph-based Active Learning for Entity Cluster
Repair</h2>



Description
========
This project consists of the source code to repair clusters for multi-source entity resolution. The approach utilizes graph metrics for creating a classification model predicting if a link between two records is correct or not. Due to the requirement of training data, an active learning step is integrated that can use the ground truth as an oracle or large language models connected via an API



Usage
=====
To run our approach, execute the method `evaluate` of the `graphCR.evaluation.al_famer` module passing the following parameters. The 
`graphCR.evaluation.al_experiments` shows an exemplary call.

Paramter | Description
---------|-------------
input_folder | folder consisting of a similarity graph
is_edge_wise | selected samples are edges(true)/(selected samples are cluster(false)) not used
use_gpt | 0=not used, 1=used, 2= generates the training/validation files to fine-tune OpenAI models
model_name | name of the llm
api_key | api key of the of the llm provider 
initial_training | initial number of training edges(20)
increment_budget | number of records per iteration in the active learning step(20)
selection_strategy | {bootstrap, bootstrap_comp = "bootstrap ext"}
output | output file for the quality evaluation 
output_2 | output file for the initial quality evaluation 
error_edge_ratio | ratio of edges where noise is added
