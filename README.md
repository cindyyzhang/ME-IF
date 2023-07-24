# Matrix Estimation for Individual Fairness

## How this library is structured

-   ME-IF
    -   data
        -   gen_data
        -   run_knn
        -   run_mlp
        -   run_usvt
        -   samples
    -   plot
        -   ind_fairness
    -   utils
        -   DataMatrix
        -   KNN
        -   MLP
        -   USVT
        -   utils

The data folder contains the scripts for generating synthetic datasets, which are stored in `samples`. Once the datasets are generated, use `run_knn`, `run_mlp`, and `run_usvt` to run the corresponding algorithms on the dataset. The utils folder contains functions and classes that assist in processing and analyzing the data. The plot folder contains the script for generating plots comparing the individual fairness ratios with and without USVT pre-processing.

## Requirements

To install requirements:

```
pip install -r requirements.txt
```

## Usage

To reproduce the results of the paper, first generate the corresponding dataset using the following command:

```
python -m data.gen_data
```

Note that you can adjust the hyperparameters in `data/gen_data.py` to modify the number of individuals, number of clusters to sample from, the length of the feature vector, and the proportion of observed entries.

Once the data has been generated, run the following command to reproduce Figure 1 of the paper:

```
python -m plot.ind_fairness
```
