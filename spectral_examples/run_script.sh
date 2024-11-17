#!/bin/bash
python sparse_inv.py
python graph_partitioning.py
python exp_design.py
python robust_pca.py 1
python robust_pca.py 2
python robust_pca.py 5

