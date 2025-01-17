# Attractive & Repulsive Trainig 
Code implementation for the following article:

Attractive and Repulsive Training to Address Inter-Task Forgetting Issues in Continual Learning, Hong-Jun Choi and Dong-Wan Choi, Neurocomputing, 2022

# Requirements
1. pytorch, python : pytorch 1.7, python 3.8

# For reproducibility
1. For experimental result : run main.py file
2. For visualization : run figure_cifar10.ipynb and figure_cifar100.ipynb

# Detailed Experimental Results
## 1. CIFAR 100 Accuracy

### 1.1 CIFAR 100 / 5 incrmental task

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


### 1.2 CIFAR 100 / 10 incremental task

|         | 10   | 20    | 30      | 40     | 50    | 60      | 70      | 80      | 90      | 100   | Average |
|---------|------|-------|---------|--------|-------|---------|---------|---------|---------|-------|---------|
| r+kd    | 91.3 | 85.25 | 75.2    | 66.675 | 58.3  | 54.55   | 49.4429 | 45.1125 | 40.1222 | 38.32 | 57.00   |
| icarl   | 91.8 | 85.95 | 75.5    | 67.475 | 59.16 | 55.25   | 50.6857 | 44.7875 | 41.3333 | 38.24 | 57.60   |
| eeil    | 91.3 | 86.25 | 78.1667 | 71.75  | 63.84 | 60.1667 | 56.1429 | 51.55   | 47.7667 | 45.36 | 62.33   |
| bic     | 91.3 | 85.75 | 79.0333 | 71.35  | 63.9  | 59.65   | 54.1714 | 49.7125 | 46.6556 | 44.08 | 61.59   |
| ART+bic | 91.3 | 85.6  | 79.6667 | 72     | 64.44 | 61.35   | 56.5286 | 52.8875 | 48.2222 | 45.91 | 62.96   |
| wa      | 92.5 | 85.8  | 79.2333 | 72.925 | 65.66 | 61.4    | 57.2714 | 53.8375 | 50.7889 | 48.27 | 63.91   |
| Art+wa  | 92.5 | 87.5  | 80.7    | 75.2   | 67.82 | 63.8667 | 59.9    | 56.3    | 52.1667 | 48.77 | 65.80   |
| CTL+wa  | 92.5 | 85.05 | 73.9333 | 68.4   | 60.56 | 54.3333 | 50.1286 | 44.325  | 40.4667 | 36.4  | 57.07   |


### 1.3 CIFAR 100 / 20 incremental task

|         | 5    | 10   | 15      | 20    | 25    | 30      | 35      | 40     | 45      | 50    | 55      | 60      | 65      | 70      | 75      | 80      | 85      | 90      | 95      | 100   | Average  |
|---------|------|------|---------|-------|-------|---------|---------|--------|---------|-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-------|----------|
| r+kd    | 92   | 89.7 | 85.0667 | 80.7  | 76.2  | 70.7333 | 66.8286 | 60.675 | 55.6222 | 52.42 | 50.5091 | 49.7167 | 46.1077 | 45.6    | 43.1067 | 40.7375 | 39.1882 | 35.3889 | 34.2105 | 34.43 | 57.44705 |
| icarl   | 90.8 | 87.7 | 84.8    | 80.55 | 74.6  | 69.8667 | 66.6    | 61.175 | 56.5333 | 53.36 | 52.4    | 49.7167 | 45.6    | 45.8857 | 44.2133 | 40.8875 | 39.3765 | 35.8111 | 34.3053 | 34.48 | 57.43305 |
| eeil    | 92   | 89.5 | 86.4667 | 82.8  | 77    | 73      | 70.5429 | 65.025 | 60.5778 | 56.18 | 54.3091 | 52.3167 | 49.0923 | 48.9429 | 47.68   | 43.9    | 42.3529 | 39.5667 | 37.2737 | 36.79 | 60.26583 |
| bic     | 92   | 87.8 | 84.3333 | 81.6  | 76.76 | 71.0667 | 68.2    | 63.8   | 59.2222 | 55.08 | 52.7455 | 50.9333 | 48.6462 | 47.0429 | 43.96   | 40.5875 | 38.2824 | 36.9778 | 34.6316 | 33.27 | 58.34696 |
| ART+bic | 92   | 87.8 | 85.8667 | 83.45 | 79.48 | 73.2333 | 69.6571 | 67.525 | 62.2444 | 57.52 | 56.0545 | 53.45   | 50.4    | 49.1    | 46.84   | 44.6375 | 42.7882 | 41.2889 | 38.4105 | 38.03 | 60.98881 |
| wa      | 92   | 89.7 | 86.3333 | 82.35 | 78.16 | 74.4    | 71.3714 | 67.7   | 63.3556 | 60.16 | 59.2    | 56.55   | 54.1385 | 52.9    | 50.96   | 49.425  | 47.3765 | 45.1333 | 43.1579 | 42.31 | 63.33407 |
| Art+wa  | 92   | 90.6 | 87.6667 | 84.9  | 80.28 | 77.4333 | 73.9143 | 70.325 | 66.1778 | 62.68 | 59.2    | 58.35   | 54.5231 | 53.8857 | 51.8    | 49.9375 | 47.9882 | 45.7889 | 44.1474 | 42.85 | 64.72239 |
| CTL+wa  | 92   | 91.9 | 85.4667 | 81.75 | 76.4  | 69.9333 | 67.9714 | 60.625 | 56.3778 | 53.76 | 50.3455 | 47.5333 | 44.9385 | 43.6857 | 42.08   | 38.4    | 37.1412 | 33.7556 | 32.7895 | 32.48 | 56.96667 |

### 1.4 CIFAR 100 / 50 incremental task

|         | 2    | 4     | 6       | 8      | 10   | 12      | 14      | 16      | 18      | 20    | 22      | 24      | 26      | 28      | 30      | 32      | 34      | 36      | 38      | 40     | 42      | 44      | 46      | 48      | 50    | 52      | 54      | 56      | 58      | 60      | 62      | 64      | 66      | 68      | 70      | 72      | 74      | 76      | 78      | 80      | 82      | 84      | 86      | 88      | 90      | 92      | 94      | 96      | 98      | 100   | Average |
|---------|------|-------|---------|--------|------|---------|---------|---------|---------|-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|--------|---------|---------|---------|---------|-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-------|---------|
| r+kd    | 94.5 | 91.75 | 95.1667 | 92.25  | 91   | 90.6667 | 88.2143 | 86.9375 | 85.3333 | 83.15 | 80.5    | 78.25   | 77.3462 | 75.3929 | 71.2    | 70.8125 | 68.2941 | 67.9444 | 67.1579 | 62.025 | 59      | 57.4545 | 56.913  | 56.3125 | 53.28 | 53.3846 | 52.4074 | 50.3393 | 49.931  | 49.5833 | 47.7097 | 48.7969 | 47.8333 | 47.5441 | 46.3    | 46.3056 | 46.4054 | 44.3553 | 42.2436 | 41.5875 | 41.8293 | 39.3333 | 39.314  | 36.8409 | 36.6444 | 37.5217 | 35.7872 | 35.8854 | 35.5306 | 35.06 | 59.08   |
| icarl   | 95   | 91.25 | 93.8333 | 92.125 | 91.4 | 89      | 87.9286 | 85.125  | 83.4444 | 82.25 | 80.0909 | 77.5833 | 77.1923 | 75.75   | 71.7    | 70.3438 | 68.0588 | 67.25   | 66.4211 | 62.275 | 59.619  | 58.7045 | 56.9565 | 56.875  | 54.6  | 54.1538 | 53.4259 | 51.1786 | 51.069  | 50.3    | 48.0161 | 48.7031 | 47.9242 | 47.7206 | 46      | 45.6667 | 45.3108 | 44.1711 | 42.6026 | 41.3    | 39.8902 | 38.75   | 37.6512 | 37.3068 | 36.6222 | 36.9239 | 35.0638 | 34.0938 | 35.2755 | 35.74 | 58.87   |
| eeil    | 94.5 | 92.25 | 94.8333 | 94.375 | 93.1 | 90.5833 | 88.0714 | 86.75   | 85.3333 | 83.35 | 80.5455 | 78.9583 | 77.7308 | 76.4643 | 72.9333 | 71.6875 | 68.8235 | 68.4444 | 68.3158 | 64.95  | 60.5952 | 58.7727 | 57.413  | 56.4583 | 55.26 | 55.0577 | 53.8889 | 51.1607 | 51.0172 | 50.55   | 49      | 50.0469 | 48.0758 | 48.7059 | 47.3    | 46.9861 | 46.2432 | 44.9474 | 43.5897 | 41.625  | 41.061  | 39.75   | 40.1628 | 38.4773 | 38.1111 | 37.913  | 35.9574 | 34.6354 | 35.4898 | 35.5  | 59.82   |
| bic     | 94.5 | 89.75 | 93.1667 | 90.875 | 89.8 | 88.9167 | 87.7143 | 84.0625 | 82.8333 | 81.65 | 78.9091 | 77.7083 | 75      | 73      | 71.1333 | 70.375  | 68.5882 | 65.6389 | 64.9737 | 62.75  | 58.9048 | 56.4545 | 54.8261 | 54.2708 | 52.72 | 52.5385 | 51.6667 | 50.1071 | 49.8276 | 46.95   | 46.5323 | 47.1094 | 44.6364 | 45.0147 | 44.0429 | 43.2361 | 43.0946 | 41.1447 | 40.6795 | 38.4    | 38.6585 | 35.6905 | 36.6279 | 35.0568 | 34.5778 | 35.413  | 33.2553 | 33.8333 | 32.5612 | 31.8  | 57.28   |
| ART+bic | 94.5 | 93    | 93.8333 | 92.875 | 91.1 | 89.6667 | 89.2857 | 86.3125 | 85.1667 | 83.9  | 81.2727 | 79.375  | 77.6538 | 76.6786 | 72.4667 | 71.1563 | 68.7353 | 68.4167 | 68.1053 | 64.425 | 62.3095 | 59.75   | 57.7174 | 57.5417 | 56.66 | 55.4423 | 54.5556 | 52.5893 | 52.3276 | 51.8667 | 51.2581 | 50.2188 | 49.5455 | 48.5882 | 48.1714 | 48.1389 | 48.3514 | 45.4474 | 45.859  | 43.4125 | 43.4878 | 40.0357 | 40.1512 | 39.5227 | 39.5889 | 40.3152 | 38.4149 | 36.6667 | 36.3367 | 37.22 | 60.51   |
| wa      | 95.5 | 89    | 92.6667 | 89.625 | 88   | 87.8333 | 86      | 84.4375 | 83.2222 | 81.25 | 80.0909 | 78.0833 | 76.8462 | 76.8214 | 73.4    | 73      | 70.9706 | 69.1944 | 68.6053 | 65.4   | 62.7619 | 62.1136 | 60      | 58.8542 | 57.72 | 57.4808 | 56.5    | 56.0714 | 54.5345 | 53.65   | 52.2258 | 50.6719 | 50.9242 | 50      | 49.6429 | 47.9444 | 47.8784 | 46.2368 | 45.7949 | 44.2    | 43.3659 | 42.5119 | 42.7209 | 40.9205 | 40.5667 | 39.0217 | 38.7128 | 37.2396 | 37.4286 | 35.38 | 60.77   |
| Art+wa  | 95.5 | 92.25 | 94.6667 | 93     | 91.7 | 89.8333 | 88.9286 | 86.9375 | 85.6667 | 84.7  | 82.8636 | 81.125  | 80.6923 | 78.8571 | 75.7333 | 73.8125 | 72.0588 | 71.4722 | 70.5789 | 68.2   | 64.9762 | 62.3864 | 60.2174 | 60.0625 | 58.16 | 57.75   | 57.4815 | 55.2679 | 55.1552 | 53.35   | 52.6452 | 50.7031 | 51.4242 | 50.4853 | 51.2429 | 48.875  | 49.8514 | 47.5921 | 47.5128 | 45.775  | 45.1951 | 43.869  | 44.0814 | 42.7727 | 42.5111 | 40.4783 | 40.5957 | 40.8646 | 39.9796 | 39.36 | 62.52   |
| CTL+wa  | 95.5 | 92.25 | 96.3333 | 93.875 | 91.8 | 88.8333 | 86.7143 | 85      | 82.9444 | 82.3  | 76.5    | 75.75   | 74.5    | 73.3214 | 66.8667 | 67.3438 | 65.1176 | 62.9444 | 61.8421 | 58.3   | 54.7381 | 54.4318 | 53.3696 | 50.2292 | 50.24 | 48.7308 | 48.1111 | 45.2857 | 45.0172 | 43.6833 | 42.5484 | 45.3906 | 42.6515 | 44.1765 | 40.6286 | 39.4167 | 40.6622 | 39.5132 | 36.2436 | 34.8125 | 34.7805 | 33.9643 | 33.2674 | 31.4205 | 31.8    | 32.837  | 28.9043 | 29.3229 | 30.2041 | 29.62 | 55.07   |



## 2. TinyImageNet Accuracy

### 2.1 TinyImageNet / 5 incrmental task

|         | 40     | 80     | 120    | 160     | 200    | Average  |
|---------|--------|--------|--------|---------|--------|----------|
| r+kd    | 70.250 | 59.825 | 46.867 | 37.300  | 30.990 | 43.75    |
| icarl   | 70.450 | 59.775 | 47.717 | 37.863  | 31.590 | 44.24    |
| eeil    | 70.250 | 58.375 | 49.883 | 41.475  | 36.420 | 46.54    |
| bic     | 70.250 | 62.250 | 53.717 | 45.038  | 37.720 | 49.68    |
| ART+bic | 70.25  | 62.625 | 53.7   | 44.4625 | 37.97  | 49.68938 |
| wa      | 69.500 | 62.175 | 54.767 | 45.063  | 36.930 | 49.73    |
| Art+wa  | 69.500 | 62.750 | 54.317 | 44.650  | 37.650 | 49.84    |
| CTL+wa  | 69.500 | 56.075 | 49.433 | 36.625  | 25.510 | 41.91    |

### 2.2 TinyImageNet / 10 incremnetal task

|         | 20    | 40    | 60    | 80    | 100   | 120     | 140     | 160     | 180     | 200   | Average |
|---------|-------|-------|-------|-------|-------|---------|---------|---------|---------|-------|---------|
| r+kd    | 74.20 | 65.95 | 60.47 | 52.35 | 47.78 | 41.88   | 36.61   | 33.70   | 30.53   | 27.40 | 44.08   |
| icarl   | 74.40 | 65.95 | 60.93 | 52.25 | 47.34 | 41.85   | 36.14   | 33.33   | 30.46   | 27.02 | 43.92   |
| eeil    | 74.20 | 64.50 | 61.13 | 53.95 | 50.38 | 45.57   | 40.67   | 37.53   | 34.90   | 32.14 | 46.75   |
| bic     | 74.20 | 65.45 | 61.77 | 55.18 | 50.76 | 45.75   | 40.21   | 36.96   | 33.40   | 27.95 | 46.38   |
| ART+bic | 74.2  | 65.8  | 62.6  | 55.25 | 51.76 | 46.5167 | 41.2714 | 37.5875 | 34.9444 | 31.68 | 47.49   |
| wa      | 74.40 | 67.30 | 64.57 | 57.18 | 54.12 | 48.83   | 44.31   | 39.85   | 36.24   | 31.14 | 49.28   |
| Art+wa  | 74.40 | 66.25 | 64.13 | 56.55 | 52.98 | 48.12   | 44.41   | 40.43   | 37.16   | 34.26 | 49.36   |
| CTL+wa  | 74.40 | 62.80 | 58.53 | 53.23 | 48.34 | 46.73   | 41.61   | 36.94   | 33.88   | 29.74 | 45.76   |

### 2.3 TinyImageNet / 20 incremental task

|         | 10   | 20   | 30      | 40    | 50    | 60      | 70      | 80     | 90      | 100   | 110     | 120     | 130     | 140     | 150     | 160     | 170     | 180     | 190     | 200   | Average  |
|---------|------|------|---------|-------|-------|---------|---------|--------|---------|-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-------|----------|
| r+kd    | 78.8 | 73   | 67.6667 | 62.95 | 59.56 | 56.1    | 51.1429 | 46.35  | 44.9778 | 42.98 | 38.5091 | 37.0833 | 34.0923 | 32.5    | 30.3733 | 29.1625 | 28.2    | 26.2222 | 23.8947 | 23.79 | 42.56    |
| icarl   | 77.4 | 73.7 | 67.2    | 62.55 | 60.16 | 55.5333 | 51.9714 | 46.9   | 45.3111 | 43.18 | 38.7636 | 37.9333 | 34.5385 | 33.0286 | 30.3333 | 29.3875 | 28.8706 | 26.3444 | 24.4526 | 23.8  | 42.84    |
| eeil    | 78.8 | 73.8 | 67.8667 | 63.65 | 61.8  | 58.2    | 55.0286 | 50.575 | 46.6    | 45.52 | 43.4545 | 41.5333 | 39.0615 | 36.6    | 35.2133 | 33.6125 | 32.3647 | 30.2667 | 28.5368 | 27.56 | 45.85    |
| bic     | 78.8 | 73.7 | 67      | 63.65 | 61.44 | 57.5    | 53.2857 | 47.625 | 45.2889 | 42.76 | 37.6909 | 34.7    | 32.7385 | 30.9    | 28.68   | 27.175  | 25.8588 | 22.1333 | 20.1053 | 19.36 | 41.66    |
| ART+bic | 78.8 | 74.8 | 69.6    | 64.4  | 61.24 | 57.9667 | 54.7429 | 49.5   | 47.4444 | 45.44 | 42.2727 | 40.3833 | 38.4462 | 35.6714 | 33.8133 | 31.35   | 31.3765 | 28.5444 | 27.4316 | 26.96 | 45.33597 |
| wa      | 74.8 | 72.3 | 68.2    | 64.45 | 62.76 | 60.0333 | 56.6857 | 52.075 | 48.8667 | 47.58 | 44.2727 | 41.4333 | 38.1231 | 35.9429 | 32.8    | 30.775  | 28.5059 | 26.6667 | 24.3789 | 22.9  | 45.20    |
| Art+wa  | 74.8 | 75   | 70.1333 | 66.2  | 63.68 | 59.5667 | 57.5714 | 51.925 | 49      | 47.94 | 45.0909 | 43.15   | 41.1077 | 39.1143 | 37.24   | 35.25   | 33.9294 | 32.2222 | 31.1579 | 30.2  | 47.87    |
| CTL+wa  | 74.8 | 75   | 65.8667 | 65.15 | 58.72 | 55.5333 | 54.6571 | 46.925 | 46.6222 | 41.36 | 41.1818 | 41.9333 | 38.8154 | 36.1429 | 35.2933 | 33.35   | 33.9765 | 31.3667 | 30.1368 | 29.03 | 45.32    |


### 2.4 TinyImageNet / 50 incremental task

|         | 4    | 8     | 12      | 16     | 20   | 24      | 28      | 32      | 36      | 40    | 44      | 48      | 52      | 56      | 60      | 64      | 68      | 72      | 76      | 80     | 84      | 88      | 92      | 96      | 100   | 104     | 108     | 112     | 116     | 120     | 124     | 128     | 132     | 136     | 140     | 144     | 148     | 152     | 156     | 160     | 164     | 168     | 172     | 176     | 180     | 184     | 188     | 192     | 196     | 200   | Average  |
|---------|------|-------|---------|--------|------|---------|---------|---------|---------|-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|--------|---------|---------|---------|---------|-------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-------|----------|
| r+kd    | 69.5 | 77    | 79      | 77.875 | 74.5 | 71.1667 | 65.7143 | 63.5    | 61.0556 | 58.15 | 59.3636 | 57.4167 | 54.6923 | 52.25   | 52.1    | 49.375  | 48      | 46.7778 | 44.5263 | 43.45  | 40.8571 | 40.6136 | 40.3043 | 40.7708 | 39.4  | 37.4808 | 36.4259 | 35.8929 | 34.3103 | 33.6167 | 32.0161 | 31.3125 | 29.9697 | 30.6618 | 29.2714 | 27.8889 | 27.6486 | 28.7763 | 27.7308 | 26.425  | 26.0976 | 26.5    | 24.3953 | 23.9318 | 25.1111 | 23.413  | 22.2021 | 22.5833 | 22.3571 | 22.2  | 41.76    |
| icarl   | 71.5 | 76.5  | 78      | 75.625 | 72.2 | 69.75   | 65.5    | 62.5    | 61.7222 | 59    | 58.5    | 55.875  | 55.4615 | 51.8929 | 53.2667 | 49.6875 | 48.0588 | 45.6944 | 42.7368 | 42.425 | 40.4048 | 39.6364 | 40.3261 | 40.1458 | 38.6  | 37.0192 | 35.3148 | 35.3036 | 33.8103 | 33.0333 | 31.0484 | 30.2188 | 29.6061 | 29.2794 | 28.2714 | 27.4861 | 27.2432 | 28.0526 | 26.7564 | 25.75   | 25.8537 | 25.3095 | 24.5581 | 24.2273 | 24.5778 | 22.4348 | 21.2553 | 21.3333 | 21.8469 | 21.55 | 41.12    |
| eeil    | 69.5 | 77.25 | 78.8333 | 76     | 73   | 71.0833 | 67.2143 | 65.125  | 63.7778 | 59.45 | 59.1364 | 57.5417 | 56.2308 | 54.2143 | 53.3667 | 51.6563 | 49.2353 | 48.8889 | 46.8421 | 45.5   | 43.5    | 43.0909 | 42.8478 | 42.5208 | 41.42 | 40.1346 | 38.1852 | 38.1964 | 37.1724 | 36.0333 | 34.9516 | 33.4375 | 32.9394 | 32.2353 | 32.1571 | 30.4306 | 30.2703 | 29.9211 | 28.8333 | 28.2125 | 28.122  | 27.8095 | 27.2093 | 26.1364 | 25.8333 | 25.3804 | 24.2021 | 24.1667 | 23.7857 | 23.34 | 43.40    |
| bic     | 69.5 | 70.75 | 75      | 73.125 | 69.4 | 66.6667 | 61      | 60.5625 | 58      | 54.4  | 52.6818 | 51.2917 | 48      | 47.2857 | 45.5667 | 42.375  | 41.1471 | 39.25   | 35.3421 | 35.825 | 34.1905 | 32.3636 | 31.587  | 31.5208 | 30.36 | 26.6923 | 24.6111 | 23.75   | 23.1207 | 22.0333 | 20.6774 | 19.8594 | 19.1515 | 18.4118 | 17.7143 | 16.75   | 16.0541 | 15.8684 | 15.5769 | 14.9    | 14.6341 | 14.5476 | 13.4884 | 12.0227 | 12.3556 | 10.8043 | 10.5532 | 9.85417 | 10.3469 | 9.89  | 32.68    |
| ART+bic | 69.5 | 74    | 77.5    | 76.375 | 72.7 | 70.9167 | 66      | 63      | 61.8889 | 58.7  | 57.4091 | 55.0833 | 54      | 52.5    | 51.6    | 49.25   | 48.6176 | 46.7222 | 43.4211 | 42.875 | 41.5714 | 40.6364 | 41.4565 | 41.25   | 40.4  | 38.3654 | 36.5556 | 36.0357 | 34.9828 | 33.7167 | 32.0484 | 30.4375 | 31.1212 | 30.2059 | 30.1286 | 29.5694 | 28.7838 | 29.6053 | 27.8846 | 26.575  | 25.8171 | 26.3571 | 25.1047 | 24.125  | 24.5556 | 23.2391 | 22.4149 | 22.2813 | 22.3265 | 22.1  | 41.67776 |
| wa      | 67.5 | 76.5  | 76.5    | 75.75  | 73   | 70.9167 | 65.9286 | 63.4375 | 61.5    | 60.1  | 58.7273 | 56.1667 | 54.3846 | 52.75   | 52.8    | 49.0313 | 47.1176 | 46.6389 | 41.5789 | 41.7   | 39.6667 | 39.4545 | 39.0652 | 38.4167 | 36.92 | 35.5192 | 33.037  | 32.75   | 29.7931 | 29.0167 | 28.0806 | 26.9375 | 25.7727 | 25.1176 | 24.0429 | 22.2361 | 21.5135 | 21.1316 | 20.1282 | 19.05   | 18.4268 | 18.0119 | 17.2442 | 16.2955 | 15.4889 | 15.087  | 14.5851 | 14.2708 | 13.7245 | 13.2  | 38.13    |
| Art+wa  | 67.5 | 76    | 78.5    | 79.125 | 76.3 | 74.5    | 68.5714 | 65.5    | 64.8333 | 61.25 | 60.2273 | 59.125  | 58      | 56.2857 | 54.9333 | 52.5313 | 50.9412 | 50.7778 | 47.5    | 46.55  | 45.9762 | 44.2955 | 43.8478 | 43.5417 | 42.86 | 42.0192 | 40.1296 | 40.2321 | 38.4655 | 37.5833 | 37.5484 | 36.1875 | 35.2879 | 34.7941 | 34.6714 | 32.875  | 31.4459 | 31.9474 | 31.3718 | 30.2125 | 30.0732 | 29.1548 | 28.5698 | 27.9318 | 27.8556 | 26.9565 | 26.3936 | 25.9375 | 25.3265 | 24.77 | 45.10    |
| CTL+wa  | 67.5 | 78.5  | 80.3333 | 79.875 | 75.7 | 70.4167 | 66.7857 | 62.375  | 62.2778 | 57.7  | 56.4545 | 54.625  | 52.1923 | 53.1786 | 49.2667 | 49.8125 | 50.4706 | 45.2778 | 40.4474 | 42.9   | 40.3571 | 40.7955 | 39.6522 | 36.7083 | 37.14 | 34.0385 | 36.6111 | 35.6071 | 34.3793 | 34.2667 | 29.8226 | 30.1875 | 28.9394 | 29.4412 | 30.7571 | 26.7361 | 25.0135 | 30.3421 | 26.7436 | 26.4625 | 28.0122 | 22.381  | 25.7558 | 24.1591 | 26.2444 | 21.7391 | 22.8723 | 23.2292 | 21.6327 | 23.14 | 41.26    |
