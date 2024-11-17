# SpectralSCSExperiments
This repository contains code to recreate the results in our [paper]() on projections for spectral matrix cones.  
If you wish to use SCS as a solver, you should *NOT* use this repository but instead use the official [repository](https://github.com/cvxgrp/scs).

## Recreating the figures in the paper
1. **Compiling the code**: Place yourself in the root folder and run `pip install .`
2. **Running experiments**: Place yourself in the folder `spectral_examples` and run the script `run_script.sh`.
3. **Plotting**: Place yourself in the folder `spectral_examples/plotting` and run the script `run_plots.sh`.

The second step takes about 48 hours to run sequentially (data files containing the results can be found in the folder `spectral_examples/plotting/data`). 
After the third step, several figures will appear in the folder `spectral_examples/plotting/figures`.


## Citing
If you wish to cite this work, you may use the following BibTex:

```bibtex
@article{Cederberg24,
title = {{P}rojections onto {S}pectral {M}atrix {C}ones},
journal = {},
volume = {},
pages = {},
year = {2024},
issn = {},
doi = {},
author = {Daniel Cederberg and Stephen Boyd},
}
