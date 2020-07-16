#!/usr/bin/env python

import fire
import csv
import numpy as np
from scipy.stats import loguniform as sci_loguniform
from itertools import product
from itertools import cycle


def _gpu_check(num_gpu):
    num_gpu = int(num_gpu)
    if num_gpu < 0:
        raise ValueError("num_gpu must be positive")

    return num_gpu


def _build_table(keys, values, num_gpu, gpu_prefix, grid=False):
    # Create the first table
    if grid:
        table = product(*values)
    else:
        table = zip(*values)

    # Init the final table, which will hold metadata in addition
    # to all the contents of the first table
    i_table = []

    # No GPU
    if num_gpu == 0:
        head = ["row_code"] + keys
        for i, t in enumerate(table):
            i_table.append((i, *t))

    # Yes GPU(s).
    # !! Dividing the work equally between 'em. !!
    else:
        head = ["row_code", "device_code"] + keys
        device_count = cycle(range(num_gpu))
        for i, t in enumerate(table):

            # Name the GPU device get a name?
            if gpu_prefix is None:
                device_code = next(device_count)
            else:
                device_code = gpu_prefix + str(next(device_count))

            i_table.append((i, device_code, *t))

    # The final table parts are:
    return i_table, head


def _save(name, table, head):
    # np.savetxt(name, table, delimiter=",", header=head, fmt=fmt, comments="")
    with open(name, mode='a+') as handle:
        writer = csv.writer(handle)
        writer.writerow(head)
        writer.writerows(table)


def grid(name, num_gpu=0, gpu_prefix=None, **kwargs):
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
    table, head = _build_table(keys, values, num_gpu, gpu_prefix, grid=True)
    _save(name, table, head)


def normal(name,
           num_sample=1,
           num_gpu=0,
           gpu_prefix=None,
           seed_value=None,
           **kwargs):
    """Gaussian parameter search."""

    # Init
    prng = np.random.RandomState(seed_value)

    # Generate random HPs
    keys = sorted(list(kwargs.keys()))
    values = []
    for k in keys:
        loc, scale = kwargs[k]
        v = prng.normal(loc=loc, scale=scale, size=num_sample)
        values.append(v)

    # Build the table and save it
    table, head = _build_table(keys, values, num_gpu, gpu_prefix)
    _save(name, table, head)


def uniform(name,
            num_sample=1,
            num_gpu=0,
            gpu_prefix=None,
            seed_value=None,
            **kwargs):
    """Uniform parameter search."""

    # Init
    prng = np.random.RandomState(seed_value)

    keys = sorted(list(kwargs.keys()))
    values = []
    for k in keys:
        start, stop = kwargs[k]
        v = prng.uniform(start, stop, num_sample)
        values.append(v)

    # Build the table and save it
    table, head = _build_table(keys, values, num_gpu, gpu_prefix)
    _save(name, table, head)


def loguniform(name,
               num_sample=1,
               num_gpu=0,
               gpu_prefix=None,
               seed_value=None,
               **kwargs):
    """Loguniform parameter search."""

    keys = sorted(list(kwargs.keys()))
    values = []
    for k in keys:
        # Get range
        start, stop = kwargs[k]

        # lognormal is not def for exactly zero.
        if np.isclose(start, 0.0):
            start += np.finfo(np.float).eps

        # Sample
        v = sci_loguniform.rvs(start,
                               stop,
                               size=num_sample,
                               random_state=seed_value).tolist()
        values.append(v)

    # Build the table and save it
    table, head = _build_table(keys, values, num_gpu, gpu_prefix)
    _save(name, table, head)


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
