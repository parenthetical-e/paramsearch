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
`paramsearch.py` has four sampling options -- grid, normal, uniform, and loguniform. For complete documenation of each run 
```bash
paramsearch.py [OPTION] --help
```

To see each option works in practive let's start off in assuming we have two hyperparameters we want to explore, over fixed range.
1. --alpha (0, 1)
2. --beta (1, 100)

## Grid search
To search these over a 10 by 10 grid, run:

```bash
paramsearch.py grid example.csv --alpha='(0, 1, 10)' --beta='(1, 1000, 10)'
```

To search these over a 10 by 100 grid while adding an index to distribute over 2 GPUs, run:

```bash
paramsearch.py grid example.csv --num_gpu=2 --gpu_prefix='cuda' --alpha='(0, 1, 10)' --beta='(1, 1000, 100)'
```

## Uniform random search
To search these 100 uniform random samples, run:

```bash
paramsearch.py uniform example.csv --num_samples=100 --alpha='(0, 1)' --beta='(1, 100)'
```

## Loguniform random search
To search these 5000 loguniform random samples, run:

```bash
paramsearch.py loguniform example.csv --num_samples=5000 --alpha='(0, 1)' --beta='(1, 100)'
```

## Normal random search
To search these 20 normal samples where the first entry is the mean and second is the standard deviation (M, SD), run:

```bash
paramsearch.py uniform example.csv --num_samples=20 --alpha='(0, 1)' --beta='(1, 100)'
```
