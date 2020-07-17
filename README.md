# paramsearch
A commandline program to generate tables of hyperparameters.

# install
```bash
git clone https://github.com/parenthetical-e/paramsearch.git
cd paramsearch
pip install .
```

# dependencies
- python >= 3.6
- [anaconda](https://docs.anaconda.com/anaconda/install/)


# usage
`paramsearch.py` has four sampling options -- grid, normal, uniform, and loguniform. To see the each of their docs, run:
```bash
paramsearch.py [OPTION] --help
```

To see how each option works in practice, let's start off in assuming we have two hyperparameters.
1. --alpha (0, 1)
2. --beta (1, 1000)

## Grid search
To search over a 10 by 10 grid, run:

```bash
paramsearch.py grid example.csv --alpha='(0, 1, 10)' --beta='(1, 1000, 10)'
```

To search over a 10 by 100 grid, while adding an index to distribute over 2 GPUs, run:

```bash
paramsearch.py grid example.csv --num_gpu=2 --gpu_prefix='cuda:' --alpha='(0, 1, 10)' --beta='(1, 1000, 100)'
```

## Uniform search
To search over 100 uniform random samples, run:

```bash
paramsearch.py uniform example.csv --num_samples=100 --alpha='(0, 1)' --beta='(1, 1000)'
```

## Loguniform search
To search over 5000 loguniform random samples, run:

```bash
paramsearch.py loguniform example.csv --num_samples=5000 --alpha='(0, 1)' --beta='(1, 1000)'
```

Note: A range of (0-1) is probably not what you want for a loguniform search.

## Normal search
To search over 20 normal samples run: 

```bash
paramsearch.py normal example.csv --num_samples=20 --alpha='(0, 1)' --beta='(1, 1000)'
```

Note: the interface is different. The first entry is the mean, and the second is the standard deviation (M, SD).
