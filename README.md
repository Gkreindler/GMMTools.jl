# GMMTools

[![Build Status](https://github.com/Gkreindler/GMMTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Gkreindler/GMMTools.jl/actions/workflows/CI.yml?query=branch%3Amain)


# Summary 
*Preliminary/In progress.*

A toolbox for generalized method of moments (GMM) and classical minimum distance (CMD) estimation, with functionality aimed at streamlining estimating models that have long runtime and estimation launched on a computer cluster.

For broadly related projects, see [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl), [DrWatsonSim.jl](https://github.com/sebastianpech/DrWatsonSim.jl), [GMMInference.jl](https://github.com/schrimpf/GMMInference.jl). [SMM.jl](https://github.com/floswald/SMM.jl)

# Features and use-cases
## Who could this package be useful for?
The idea behind this package is that rapid iteration is useful when doing science. When working with large or complicated economic models and estimating them using GMM-type methods, valuable research time can be wasted collecting results, re-running partial estimation cycles, or bootstrap runs, after a bug causes an error in one run, etc. This package aims to automate and speed up some common steps in such workflows.

What does this package add above and beyond just coding `g'Wg` directly? The ultimate aim is to offer a similar set of features that a typical OLS package adds above and beyond coding directly `(X'X)-1X'Y`.

If you find yourself estimating GMM/CMD models and spend significant time on routine operations, read on.

## Features
Core estimation features:
1. run CMD estimation or GMM estimation, either one-step or two-step with optimal weight matrix
1. multiple initial conditions
1. parameter box constraints
1. estimate asymptotic variance-covariance matrix
1. “slow” bootstrap

Convenience features:
1. efficiently resume estimation based on incomplete results (e.g. when bootstrap run #63, or initial condition #35, fails or runs out of time after many hours)
1. parallel initial conditions (embarrassingly parallel using Distributed.jl)
1. parallel bootstrap (embarrassingly parallel using Distributed.jl)
1. suitable for running on computer clusters (e.g. using slurm)
1. include limits for time or number of iterations 
1. easily select subset of parameters to estimate
1. easily select subset of moments used in estimation

### Dev to-do list:
1. flags for (1) optimum from run that did not converge, (2) 1st stage optimum from run that did not converge
1. double-check how bootstrap works for CMD_optimal
1. accept user-provided bootstrap sampling function
1. test time limit hit
1. output estimation results text
1. test package install on new computer
1. test the existing “quick” bootstrap option
1. Decide whether to check or reuse existing initial conditions, bootstrap samples, when resuming estimation with incomplete results
1. (lower priority) implement "proper" package tests

easy:
1. boot cleanup: compile boot misc files into single large files, delete folders

### Wish-list
1. integrate optimization backends other than `curve_fit` from `LsqFit.jl`, e.g. `Optim.jl`, [`GalacticOptim.jl`](`https://github.com/SciML/GalacticOptim.jl`), etc.
1. automatic differentiation 
1. more general estimation of the covariance of the moments, e.g. Newey-West, etc.
1. integrate with RegressionTables.jl
1. compute sensitivity measure (Andrews et al 2017)
1. (using user-provided function to generate data from model) Monte Carlo simulation to compute size and power.
1. (using user-provided function to generate data from model) Monte Carlo simulation of estimation finite sample properties (simulate data for random parameter values ⇒ run GMM ⇒ compare estimated parameters with underlying true parameters)

# Install
To install this package:
`] add https://github.com/Gkreindler/GMMTools.jl`

Note: as of January 2023, this packages requires `LsqFit.jl#master` (for the `maxTime` option). After the step above, run:
1. `] remove LsqFit`
1. `] add LsqFit#master  # or ] https://github.com/JuliaNLSolvers/LsqFit.jl#master`
1. `] update # this should be optional`

# Basic Usage
The user must provide two objects:
1. A function `moments(theta,data)` that returns an NxM matrix, where `theta` is the parameter vector, `N` is the number of observations, and `M` is the number of moments. `N=1` for CMD.
1. An object `data`. (Can be anything. By default `Dict{String, Any}` with values tha are vectors or matrices with 1st dimension of size `N`. In this format, sampling for slow bootstrap is done automatically.)

# Examples
See examples with toy models in 
```
examples/example_cmd.jl
examples/example_gmm2step.jl
examples/example_parallel.jl # parallel computation for (i) multiple initial conditons, and (ii) boostrap
```

# Notes
- the optimizer is LsqFit.jl, because the GMM/CMD objective is a sum of squares (using the Cholesky decomposition of the weighting matrix). In principle, other optimizers can be used. The Cholesky decomposition `Whalf` of a positive definite matrix `W` is a matrix that satisfies `Whalf' * Whalf = W`. This means that we can re-write the objective `g'Wg` as `(Whalf*g)'*(Whalf*g)`, which is a sum of squares.
- in optimization, the gradient is currently computed using finite differences
- rules for combining estimation results with multiple initial conditions: 
    - runs that produce errors are ignored
    - the optimum `theta_hat` corresponds to the run with the minimum objective value
    - if the run with minimum objective value did not converge (hit `maxTime` or `maxIter`) a flag is returned to signal this issue. <span style="color:red"> TODO: what flag?. </span>


# Resuming estimation after crash, errors, or time limit
If during an estimation cycle some runs are completed successfully while others raise errors or hit the time limit, re-running the entire estimation cycle is efficient, that is, it does not repeat those runs that finished successfully. When it takes a long time to complete an entire estimation cycle (e.g. because of many initial conditions, bootstrap, or both), this can be helpful. There are three possible types of issues that may arise:
1) The entire estimation cycle crashes, e.g. out of memory error, or cluster time limit reached, etc.
2) An individual run leads to an error.
3) An individual run does not converge because it exceeds `maxIter` or `maxTime`.

All three issues can be handled in `GMMTools.jl`.

Set the options `"main_overwrite_runs"` and `"boot_overwrite_runs"`  in `gmm_options` to one of the following
```
0 = do not overwrite anything, but launch the runs that are missing # this will save the most time
1 = overwrite runs that hit the time or iterations limit
2 = overwrite runs that errored
3 = overwrite both 1 and 2
10 = overwrite all existing files
```

It is the user's responsibility to ensure that the existing results and the new runs are compatible. TODO: decide if/what checks to do, such as whether the initial conditions are the same (or load them from existing files).

### Understanding the file output from incomplete cycles
We save the results from each initial condition run in a separate file (one-row dataframe) in `"SUBFOLDER/results_df_run_<iter_n>.csv"` where `SUBFOLDER` is one of `"results", "step1", "step2"`. After all runs are finished, we combine all results into a single dataframe in `"estimation_SUBFOLDER_df.csv"`. To avoid a large number of files (thousands in the case of bootstrap with multiple initial conditions), we clean up and delete the individual run output files (the entire `"SUBFOLDER"` subfolder) after the combined dataframe is generated.

## Acknowledgements
For useful suggestions: Peter Deffebach, Michael Creel.