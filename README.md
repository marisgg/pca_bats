# PCA Bats
Bat Algorithm with Principal Component Analysis for multi-dimensional optimisation problems

## Requirements
Install the requirements with pip:
```
pip install -r requirements.txt
```

## Run
Define a run of the bat algorithm in the main method of the main file, you need to define a X-dimensional fitness function e.g.
```
run(
    True,               # Whether to use PCA analysis for principal individuals
    functions.rosen,    # Function to benchmark
    -100,               # Lower bound of problem
    100,                # Upper bound
    200                 # Generations
    )
```

## Reproduce results

To reproduce the results of the paper, run the `benchmark.py` script twice. Once with `levy=True` and once with `levy=False`, indicating the use of levy flight over global search, in the run method of the `main.py` script. Then execute
```
python3 benchmark.py
```
for each option for `levy`. For reproducibility, all results were produced with numpy's `default_rng()` initialised with seed 0.