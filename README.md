# NCPNET

## Overview

This repo is the code for NCPNET based on the open-sourced code for *TGL: A General Framework for Temporal Graph Training on Billion-Scale Graphs*.

## Requirements
- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511

The temporal sampler is implemented using C++, please compile the sampler first with the following command
> python setup.py build_ext --inplace

## Dataset
To use your own dataset, you need to put the following files in the folder `\DATA\\<NameOfYourDataset>\`

1. `edges.csv`: The file that stores temporal edge informations. The csv should have the following columns with the header as `,src,dst,time,ext_roll` where each of the column refers to edge index (start with zero), source node index (start with zero), destination node index, time stamp, extrapolation roll (0 for training edges, 1 for validation edges, 2 for test edges). The CSV should be sorted by time ascendingly.
2. `ext_full.npz`: The T-CSR representation of the temporal graph. We provide a script to generate this file from `edges.csv`. You can use the following command to use the script 
    >python gen_graph.py --data \<NameOfYourDataset>
3. `edge_features.pt` (optional): The torch tensor that stores the edge featrues row-wise with shape (num edges, dim edge features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*
4. `node_features.pt` (optional): The torch tensor that stores the node featrues row-wise with shape (num nodes, dim node features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*
5. `labels.csv` (optional): The file contains node labels for dynamic node classification task. The csv should have the following columns with the header as `,node,time,label,ext_roll` where each of the column refers to node label index (start with zero), node index (start with zero), time stamp, node label, extrapolation roll (0 for training node labels, 1 for validation node labels, 2 for test node labels). The CSV should be sorted by time ascendingly.

## Run

> python train_node_cp.py  --data REDDIT --config ./config/TGAT.yml --model ./models/REDDIT_TGAT.pkl --exp 10 --gpu 1 --tau 0.5 --calib_epochs 1500 --lr_calib 0.001 --sigma 0.0001 --cov_weight 0.0 --epsilon 0.01

> python train_node_cp.py  --data REDDIT --config ./config/JODIE.yml --model ./models/REDDIT_JODIE.pkl --exp 10 --gpu 1 --tau 0.5 --calib_epochs 2000 --lr_calib 0.005 --sigma 0.00001 --cov_weight 0.0 

> python train_node_cp.py  --data REDDIT --config ./config/TGN.yml --model ./models/REDDIT_TGN.pkl --exp 10 --gpu 2 --tau 0.5 --calib_epochs 1500 --lr_calib 0.01 --sigma 0.0001 --cov_weight 0.0

> python train_node_cp.py --data WIKI --config ./config/TGAT.yml --model ./models/WIKI_TGAT.pkl --exp 10 --gpu 0 --tau 0.3 --calib_epochs 1500 --lr_calib 0.001 --sigma 0.0001 --cov_weight 0.0

> python train_node_cp.py --data WIKI --config ./config/JODIE.yml --model ./models/WIKI_JODIE.pkl --exp 10 --gpu 0 --tau 0.8 --calib_epochs 1500 --lr_calib 0.001 --sigma 0.0001 --cov_weight 0.0 --epsilon 0.01

> python train_node_cp.py --data WIKI --config ./config/TGN.yml --model ./models/WIKI_TGN.pkl --exp 10 --gpu 1 --tau 0.3 --calib_epochs 1500 --lr_calib 0.001 --sigma 0.0001 --cov_weight 0.0


## License

This project is licensed under the Apache-2.0 License.