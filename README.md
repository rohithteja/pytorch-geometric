# Pytorch geometric example for node classification using cora dataset

This repo contains the code for graph neural network implementation using pytorch geometric on the cora dataset.

## Install pytorch geometric
Check pytorch version
`python -c "import torch; print(torch.__version__)"`

Install pytorch geometric with the same pytorch version

`pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html`

`pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html`

`pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html`

`pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu101.html`

`pip install torch-geometric`


## To run the experiments:
First load data using:
`python load_data.py`

Run the GCN model using:
`python node-classification-gcn.py --lr 0.01 --decay 5e-4 --hc 16 --epochs 1000`

Run the GAT model using:
`python node-classification-gat.py --lr 0.01 --decay 5e-4 --hc 16 --epochs 1000`

lr = learning rate  
hc = number of neurons in the hidden layer
