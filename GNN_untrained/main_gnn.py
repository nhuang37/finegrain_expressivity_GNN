import auxiliarymethods.datasets as dp
from auxiliarymethods.gnn_evaluation import gnn_evaluation
from gnn_baselines.gnn_architectures import GIN, GINE, GC

def main():
    num_reps = 1

    ### Smaller datasets.
    dataset = [["IMDB-BINARY", False], ["IMDB-MULTI", False], ["NCI1", True], ["PROTEINS", True],
               ["REDDIT-BINARY", False], ["ENZYMES", True]]

    results = []
    for d, use_labels in dataset:
        # Download dataset.
        dp.get_dataset(d)

        acc, s_1, s_2 = gnn_evaluation(GC, d, [3], [64], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True, untrain=False)

        print(d + " " + "GraphConv " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GraphConv " + str(acc) + " " + str(s_1) + " " + str(s_2))

        acc, s_1, s_2 = gnn_evaluation(GC, d, [3], [64], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True, untrain=True)

        print(d + " " + "GraphConv " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GraphConv " + str(acc) + " " + str(s_1) + " " + str(s_2))



        # GIN, dataset d, layers in [1:6], hidden dimension in {32,64,128}.
        acc, s_1, s_2 = gnn_evaluation(GIN, d, [3], [64], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True, untrain=False)

        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))

        acc, s_1, s_2 = gnn_evaluation(GIN, d, [3], [64], max_num_epochs=200, batch_size=64,
                                       start_lr=0.01, num_repetitions=num_reps, all_std=True, untrain=True)

        print(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))
        results.append(d + " " + "GIN " + str(acc) + " " + str(s_1) + " " + str(s_2))




    for r in results:
        print(r)


if __name__ == "__main__":
    main()
