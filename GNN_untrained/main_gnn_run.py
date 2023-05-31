import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN, GINE, GC
import os
import argparse
import numpy as np
import pickle

def main(args):
    ### Smaller datasets.
    dataset = [["MUTAG", False], ["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
                ["REDDIT-BINARY", False]]


    results = []
    for d, use_labels in dataset:
        # Download dataset.
        dp.get_dataset(d)

        acc, s_1, s_2, time, t_std = gnn_evaluation(GC, d, [args.layer], [args.hid_dim], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=args.num_reps, all_std=True, untrain=False)

        print(d + " " + "GraphConv " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(time) + " " + str(t_std))
        results.append(d + " " + "GraphConv " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(time) + " " + str(t_std))

        acc, s_1, s_2, time, t_std = gnn_evaluation(GC, d, [args.layer], [args.hid_dim], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=args.num_reps, all_std=True, untrain=True)

        print(d + " " + "GraphConv " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(time) + " " + str(t_std))
        results.append(d + " " + "GraphConv " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(time) + " " + str(t_std))



        # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        acc, s_1, s_2, time, t_std = gnn_evaluation(GIN, d, [args.layer], [args.hid_dim], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=args.num_reps, all_std=True, untrain=False)

        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(time) + " " + str(t_std))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(time) + " " + str(t_std))

        acc, s_1, s_2, time, t_std = gnn_evaluation(GIN, d, [args.layer], [args.hid_dim], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=args.num_reps, all_std=True, untrain=True)

        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2)+ " " + str(time) + " " + str(t_std)) 
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2) + " " + str(time) + " " + str(t_std))
    
        file_name = os.path.join(args.result_path, d + "_" + str(args.hid_dim) 
                                 + "_" + str(args.layer) + "_timed_constant.pkl")
        pickle.dump(results, open(file_name, "wb" ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Untrain MPNN")
    parser.add_argument("--num_reps", type=int, default=10, help="number of experiment runs")
    parser.add_argument("--hid_dim", type=int, default=64, help="hidden dimension (number of functions)")
    parser.add_argument("--layer", type=int, default=3, help="hidden layer (number of WL iterations)")    
    parser.add_argument("--result_path", type=str, default="./results/", help="dataset folder path")

    args = parser.parse_args()

    print(args)
    main(args)
