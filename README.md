# ART 

# Attractive & Repulsive Trainig 
Code implementation of paper, ART: Class-Incremental learning using Class-wise Flags

# Requirements
1. pytorch, python : pytorch 1.7, python 3.8

# For reproducibility
1. For experimental result : run main.py file
2. For visualization : run figure_cifar10.ipynb and figure_cifar100.ipynb

# Detailed Experimental Results
1. CIFAR 100 / 5 incrmental task

|         | 20    | 40     | 60      | 80      | 100   | Average |
|---------|-------|--------|---------|---------|-------|---------|
| r+kd    | 88.4  | 74.125 | 61.4    | 52.1375 | 43.24 | 63.86   |
| icarl   | 88.35 | 74.1   | 60.85   | 51.4125 | 44.16 | 63.77   |
| eeil    | 88.4  | 76.275 | 66.3333 | 58.8375 | 51.88 | 68.35   |
| bic     | 88.4  | 77.625 | 68.2667 | 61.6375 | 53.22 | 69.83   |
| ART+bic | 88.4  | 77.25  | 68.15   | 60.6625 | 54.25 | 69.74   |
| wa      | 89.85 | 78.125 | 68.9667 | 60.7375 | 53.23 | 70.18   |
| Art+wa  | 89.85 | 78.425 | 68.6833 | 61.35   | 54.41 | 70.54   |
| CTL+wa  | 89.85 | 71.2   | 62.3    | 52.6875 | 46.11 | 64.43   |


3. For visualization : run figure_cifar10.ipynb and figure_cifar100.ipynb
