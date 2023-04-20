# Fine-grained Expressivity of Graph Neural Networks

Source code for paper "Fine-grained Expressivity of Graph Neural Networks" that answers the following research questions:
- Q1: (distance_preservation) To what extent do our graph metrics act as a proxy for distances between MPNN's learned vectorial representations?
- Q2: (GNN_untrained) How do untrained MPNNs compared to their trained counterparts in terms of predictive performance?

## Requirements

- Python 3.7+
- Pytorch 1.10+
- [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

## Q1: 
- ```distance_preservation/Finegrain_MPNN_evaluation.ipynb```: experiments comparing graph distance using our metrics and MPNN embedding distances, using simulated SBM graphs and real-world benchmark graphs from TUDataset
- ```distance_preservation/distance.py```: code to compute Prokhorov distance/Wasserstein distance using our metrics on Iterated Degree Measures

## Q2:
- ```GNN_untrained/```: experiments demonstrate the surprising effectiveness of untrained MPNNs compared to their trained counterparts
