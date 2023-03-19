# Untrained MPNNs for Graph Classification

1. Simulation
- Clustering of graphs sampled from a mixture of Erdos-Renyi graphs ER(p) and Stochastic Block Models SBM(p,q)
- Fix $n=50, p=0.5$, vary $q \in [0.1, 0.3, 0.4, 0.45 ]$
- Untrained MPNNs yield perfect clustering for $q \in [0.1, 0.3]$, almost perfect clustering for $q = 0.4$, and fail for $q = 0.45$ 

2. Real-world graphs (TUDataset)
- Binary graph classification (without node features): 81/9/10 train/validation/test splits (across 10 runs)
- Untrain MPNNs achieve competitive performance compared to trained MPNNs (based on GIN architecture that simulates WL test)

| MUTAG | #params (K) |Accuracy ($\pm$ std) 
| --- | --- | --- |
| Trained | 16.898 | 0.88 (0.12)
| Untrained | 4.290 | 0.87 (0.11)


---

| PROTEINS | #params (K) |Accuracy ($\pm$ std) 
| --- | --- | --- |
| Trained | 16.898 | 0.72 (0.04)
| Untrained | 4.290 | 0.70 (0.03)

---



| IMDB-BINARY | #params (K) |Accuracy ($\pm$ std) 
| --- | --- | --- |
| Trained | 16.898 | 0.70 (0.05)
| Untrained | 4.290 | 0.69 (0.03)
