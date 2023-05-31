import numpy as np
import prokhorov as distance #credit to Jan - Updated script
import time
import random
import torch
import pickle

import argparse

from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
import os.path as osp
from torch_geometric.utils import to_networkx

seed = 406
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(args):
    path = osp.join(args.data_path, args.data)
    dataset = TUDataset(path, name=args.data) #.shuffle()
    print(f"dataset has {len(dataset)} graphs")

    num_G = len(dataset)
    D = np.zeros((num_G, num_G)) #upper triangular matrix
    for i in range(num_G):
        for j in range(i+1,num_G):
            D[i,j] = distance.demoRun(to_networkx(dataset[i], to_undirected=True), to_networkx(dataset[j], to_undirected=True),
                                    "Wasserstein distance", "flow, at most 3 iterations after refinement", 
                                    distance.wassersteinMinCostFlow, convIterationBound = 3, verbose=False)
            pickle.dump(D, open(f"{args.data}_pairwise_dist.pkl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute distance of graphs in TUDataset")
    parser.add_argument("--data", type=str, default="MUTAG", help="dataset name")
    parser.add_argument("--data_path", type=str, default='./datasets', help="dataset path")

    args = parser.parse_args()

    print(args)
    main(args)
