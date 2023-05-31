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
  - Reproduce Figure 2, 3, 4, 5
- ```distance_preservation/Prokhorov.py```: code to compute Prokhorov distance/Wasserstein distance using our metrics on Iterated Degree Measures
- ```distance_preservation/compute_dist_TUD.py```: code to pre-compute pairwise graph distance using Prokhorov distance/Wasserstein distance for certain TUDatasets

## Q2:
- ```GNN_untrained/```: experiments demonstrate the surprising effectiveness of untrained MPNNs compared to their trained counterparts
  - To reproduce results of MPNNs with graph size normalization (e.g., Table 1 - 3), run ```python GNN_untrained/main_gnn_mean.py --layer 3 --hid_dim 512``` (```--hid_dim 128``` for Table 3)
  - To reproduce results of MPNNs without graph size normalization (e.g., Table 4), run ```python GNN_untrained/main_gnn_run.py --layer 3 --hid_dim 128```
