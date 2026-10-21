# Low-Rank Quantile Surface Causal Discovery

This repository contains the implementation and core reproduction scripts for
Low-Rank Quantile Surface Causal Discovery (LRQS).  LRQS compares low-rank
reconstruction scores of empirical conditional quantile surfaces in the two
candidate causal directions.

## Installation

Create a Python environment and install the project requirements:

```shell
python -m pip install -r requirements.txt
```

## Using LRQS

```python
import numpy as np
from lrqs import predict_causal_direction

direction = predict_causal_direction(
    X=np.asarray(x1).ravel(),
    Y=np.asarray(x2).ravel(),
)
```

The return value is `1` for `X -> Y` and `-1` for `Y -> X`.  The released
implementation uses `K=1` by default, which gives an intercept-inclusive rank
cap of `K+1=2`.  The paper's main evaluation protocol uses `K_fit=2`; the
reproduction scripts below therefore pass `K=2` explicitly.

The main keyword arguments are:

- `K`: fitted surface complexity (`1` by default)
- `G`: number of conditioning bins (`10` by default)
- `B`: number of response quantiles (`10` by default)
- `n_iter`: outer iterations (`10` by default)
- `n_inner`: inner iterations (`5` by default)
- `n_init`: random initializations (`5` by default)
- `fix_g`: use the identity response transformation when `True`

## Core reproduction commands

Run the commands from the repository root:

```shell
python -m reproduce.run_lrqs_est_g
python -m reproduce.run_lrqs_fix_g
python -m reproduce.run_anm
python -m reproduce.run_divot
```

Each command runs the **42 custom benchmark settings**: 30 PNL settings and 12
MultiRank settings, with 100 pairs per setting (4,200 pair-level rows).  Fresh
outputs are written under `reproduce/results/rerun/`; existing result files
under `reproduce/results/baselines/` and `reproduce/results/proposed/` are not
overwritten.

The LRQS variants are:

- `est-g`: estimate a monotone response transformation by isotonic regression
- `fix-g`: fix the response transformation to the identity

ANM is called through the
[`causal-learn`](https://github.com/py-why/causal-learn) package.  The included
DIVOT wrapper is adapted from the
[official DIVOT implementation](https://github.com/TURuibo/DIVOT).

### Running the standard benchmarks

The 24 standard benchmark settings are present but disabled by default.  To
include them, uncomment the clearly marked `Optional standard benchmarks`
block in `reproduce/bivariate.py`.  Uncommenting that block adds the 24
settings to the default 42.  To run only the standard settings, also comment
out the 42 custom entries.

## Data

The datasets used by the reproduction scripts and reported results are stored
under `data/bivariate/`.

- `PNL/`: 30 custom PNL settings
- `MultiRank/`: 12 custom structural-complexity settings
- `ANLSMN_pairs/`, `Benchmark_simulated/`, `Dataverse_pairs/`,
  `DREAM_challenges/`, `Flow_simulated/`, `QPE_constrained/`, and
  `Tuebingen/`: the 24 standard benchmark settings

The standard benchmark organization follows the evaluation suite released
with [QPE](https://github.com/cyisk/qpe_cd).  Dataset-specific terms and
attribution from the original sources continue to apply.

## Results

Result CSV files are stored under `reproduce/results/`:

```text
reproduce/results/
├── baselines/
│   ├── t1/        # 42 custom settings (4,200 pair rows per complete method)
│   └── t2/        # 24 standard settings (3,859 evaluated pair rows)
└── proposed/      # LRQS results across all 66 settings (8,059 pair rows)
```

Pair-level result files use the columns
`dataset_id,pair_id,direction,delta_time`.  Metrics files use
`dataset_id,prec,audrc,dt_mean,dt_std`.

The standard-benchmark row counts reflect the exclusion of Tuebingen pairs
47, 52, 53, 54, 55, 70, 71, 105, and 107, following the benchmark protocol.

The QPE and CVEL source code is not included.  The provided CSVs were produced
with the corresponding official implementations:

- QPE: <https://github.com/cyisk/qpe_cd>
- CVEL: <https://github.com/xijohnny/causal-velocity>

## Third-party software

- ANM is provided by `causal-learn` and used as an installed dependency.
- `divot.py` is adapted from DIVOT and remains subject to its MIT license; the
  upstream license text is included in `licenses/DIVOT-LICENSE`.
- QPE and CVEL are represented only by links, settings documentation, and
  result CSVs; their source code is not included.

## License

LRQS code in this repository is released under the MIT License; see `LICENSE`.
Third-party code and datasets are governed by their respective terms.  The
DIVOT notice is preserved separately as described above.
