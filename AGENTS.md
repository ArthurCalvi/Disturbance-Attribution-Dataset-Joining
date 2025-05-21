# Context 

This repository is used for my PhD research about Forest Disturbances. 

It has two main objectives : 
- Preprocess datasets
- Join datasets by using Louvain Communities and HDBSCAN in order to have better information on disturbance events in France

It has been coded while ago. It's very messy. I want to refactor eveything to have a more clean and lean code that respects PEP8 and best practices. 

# Repository structure 

Old codes can be found in the following folders : 
- annotation/
- join_datasets/
- process_datasets/
- results/
- sampling/
- visualisation/

In join_datasets, experiences.ipynb defines the latest experiments conducted to build this pipeline : Once de the datasets are preprocessed we can join them using Louvain Communities and then apply HDBSCAN on those communities to get better information on each disturbance event. 

# Aim 

New code, simplified, relying on PEP8 and OOP. 

**Structure :** 
excerpts/ 
src/
- __init__.py 
- preprocessing/
- join/
- utils.py
- constants.py
tests/
results/

  
