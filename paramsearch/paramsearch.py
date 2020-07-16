import fire
import numpy as np
from scipy.stats import loguniform as sci_loguniform
from itertools import product
from itertools import cycle


def _gpu_check(num_gpu):
    num_gpu = int(num_gpu)
    if num_gpu < 0:
        raise ValueError("num_gpu must be positive")

    return num_gpu


def _build_table(values, num_gpu):
    # then rearrange it into a nice csv/table
    table = product(*values)
    i_table = []

    # The table depends on GPU use, so...
    # No GPU
    if num_gpu == 0:
        for i, t in enumerate(table):
            i_table.append((i, *t))

        head = "row_code," + ",".join(keys)
        if fmt is None:
            fmt = '%i,' + '%.6f,' * (len(keys) - 1) + '%.6f'

    # Use GPU(s). Dividing the work equally between 'em.
    else:
        device_count = cycle(range(num_gpu))
        for i, t in enumerate(table):
            i_table.append((i, next(device_count), *t))

        head = "row_code,device_code," + ",".join(keys)
        if fmt is None:
            fmt = '%i,%i,' + '%.6f,' * (len(keys) - 1) + '%.6f'

    # Form final table and save it.
    return np.vstack(i_table), head, fmt


def _save(name, table, head, fmt):
    np.savetxt(name, table, delimiter=",", header=head, fmt=fmt, comments="")


def grid(name, fmt=None, num_gpu=0, **kwargs):
    """Grid parameter search."""

    # Sanity
    num_gpu = _gpu_check(num_gpu)

    # Create the grid
    keys = sorted(list(kwargs.keys()))
    values = []
    for k in keys:
        start, stop, n = kwargs[k]
        v = np.linspace(start, stop, n)
        values.append(v)

    # Build the table and save it
    table, head, fmt = _build_table(values, num_gpu)
    _save(name, table, head, fmt)


def normal(name, fmt=None, num_gpu=0, seed_value=None, **kwargs):
    """Gaussian parameter search."""

    # Init
    prng = np.random.RandomState(seed_value)

    # Generate random HPs
    keys = sorted(list(kwargs.keys()))
    values = []
    for k in keys:
        loc, scale, n = kwargs[k]
        v = prng.normal(loc=loc, scale=scale, size=n)
        values.append(v)

    # Build the table and save it
    table, head, fmt = _build_table(values, num_gpu)
    _save(name, table, head, fmt)


def uniform(name, fmt=None, num_gpu=0, seed_value=None, **kwargs):
    """Uniform parameter search."""

    # Init
    prng = np.random.RandomState(seed_value)

    keys = sorted(list(kwargs.keys()))
    values = []
    for k in keys:
        start, stop, n = kwargs[k]
        v = prng.uniform(start, stop, n)
        values.append(v)

    # Build the table and save it
    table, head, fmt = _build_table(values, num_gpu)
    _save(name, table, head, fmt)


def loguniform(name, fmt=None, num_gpu=0, seed_value=None, **kwargs):
    """Loguniform parameter search."""

    keys = sorted(list(kwargs.keys()))
    values = []
    for k in keys:
        start, stop, n = kwargs[k]
        v = sci_loguniform.rvs(start, stop, size=n,
                               random_state=seed_value).tolist()
        values.append(v)

    # Build the table and save it
    table, head, fmt = _build_table(values, num_gpu)
    _save(name, table, head, fmt)


if __name__ == "__main__":
    # Define subprograms for 'paramsearch.py`
    cl = {
        "grid": grid,
        "normal": normal,
        "loguniform": loguniform,
        "uniform": uniform
    }

    # Define the full CL, automagically.
    fire.Fire(cl)
