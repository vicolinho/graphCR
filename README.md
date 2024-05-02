<h2 align="center"> Graph-based Active Learning for Entity Cluster
Repair</h2>



Description
========
This project consists of the source code for the submitted paper with the title "Graph-based Active Learning for Entity Cluster
Repair".



Usage
=====
To run our approach, execute the method `evaluate` of the `graphCR.evaluation.al_famer` module passing the following parameters. The 
`graphCR.evaluation.al_experiments` shows an exemplary call.

Paramter | Description
---------|-------------
input_folder | folder consisting of a similarity graph
is_edge_wise | selected samples are edges(true)/(selected samples are cluster(false)) not used
use_gpt | 0=not used, 1=used, 2= generates the training/validation files to fine tune open ai models
model_name | name of the llm
api_key | api key of the of the llm provider 
initial_training | initial number of training edges(20)
increment_budget | number of records per iteration in the active learning step(20)
selection_strategy | {bootstrap, bootstrap_comp = "bootstrap ext"}
output | output file for the quality evaluation 
output_2 | output file for the initial quality evaluation 
error_edge_ratio | ratio of edges where noise is added
