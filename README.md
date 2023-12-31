# GMMTools

[![Build Status](https://github.com/Gkreindler/GMMTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Gkreindler/GMMTools.jl/actions/workflows/CI.yml?query=branch%3Amain)


# Summary 
*Preliminary/In progress.*

A toolbox for generalized method of moments (GMM) with functionality aimed at streamlining estimating models that have long runtime, using parallel computing.

This package takes care of several `menial' tasks: saving and loading estimation results from files, seamlessly continuing long-running jobs that failed mid-estimation, creating publication-ready tables using RegressionTables.jl. The package gives special attention to the optimizer "backend" used to minimize the GMM objective function, and several options are available.

## Key Features
Core estimation features:
1. one- and two-step GMM estimation
1. nonlinear optimizer: support for Optim.jl and LsqFit.jl backends, multiple initial conditions, box constraints, automatic differentiation (AD)
1. inference: (1) asymptotic i.i.d. and (2) Bayesian (weighted) bootstrap

Convenience features:
1. integrated with RegressionTables.jl for publication-quality tables
1. efficiently resume estimation based on incomplete results (e.g. when bootstrap run #63, or initial condition #35, fails or runs out of time or out of memory after many hours)
1. parallel over initial conditions (embarrassingly parallel using `Distributed.jl`)
1. parallel bootstrap
1. suitable for running on computer clusters (e.g. using slurm)
1. simple option to normalize parameters before they are fed to the optimizer
1. simple option to estimate a subset of parameters and use fixed values for the others (also works with AD)

# Example
See a fully worked out example in
```
examples/example_ols.jl
```

The following tutorial covers how to install Julia and this package and explains the above example script line by line: [docs/src/tutorials/gmm.md](https://github.com/Gkreindler/GMMTools.jl/blob/main/docs/src/tutorials/gmm.md). It aims to be accessible to first-time Julia users who are familiar with the econometrics of GMM.

# Basic usage
The user must provide a moment function `mom_fn(data, theta)` that takes a `data` object (it can be a DataFrame or any other object) and a parameter vector `theta`. The function must returns an $N\times M$ matrix, where $N$ is the number of observations and $M$ is the number of moments.

To estimate a model using two-step optimal GMM, compute the asymptotic variance-covariance matrix, and display a table with the results, run
```julia
myfit = GMMTools.fit(MYDATA, mom_fn, theta0, mode=:twostep)
GMMTools.vcov_simple(MYDATA, mom_fn, myfit)
regtable(myfit)
```

The parameter `theta0` is either a vector of size $P$, or a $K\times P$ matrix with $K$ sets of initial conditions. The convenience function `GMMTools.random_initial_conditions(theta0::Vector, K::Int; theta_lower=nothing, theta_upper=nothing)` generates `K` random initial conditions around `theta0` (and between `theta_lower` and `theta_upper`, if provided).

For inference using Bayesian (weighted) bootstrap, replace the second line by 
```julia
# generate bootstrap weights first and separately. In case the estimation is interrupted, running this code again generates exactly the same weights, so we can continue where we left off.
bweights_matrix = boot_weights(MYDATA, myfit, nboot=500, method=:simple, rng_initial_seed=1234) 
vcov_bboot(MYDATA, mom_fn, theta0, myfit, bweights_matrix, opts=myopts)
```


# Options
`fit()` accepts detailed options that control (1) whether and how results are saved to file, and (2) the optimization backend and options.

```julia
# estimation options
myopts = GMMTools.GMMOptions(
                path="C:/temp/",          # path where to save estimation results (path = "" by default, in which case no files are created)
                write_iter=false,         # write results for each individual run (corresponding to initial conditions)
                clean_iter=true,          # delete individual run results after estimation is finished
                overwrite=false,          # if false, read existing results (or individual runs) and do not re-estimate existing results. It is the user's responsibility to ensure that the model and theta0 have not changed since the last run
                optimizer=:lsqfit,        # optimization backend to use. LsqFit.jl uses the Levenberg-Marquardt algorithm (see discussion below)
                optim_autodiff=:forward,  # use automatic differentiation (AD) to compute gradients. Currently, only forward AD using ForwardDiff.jl is supported
                lower_bound=[0.0, -Inf],  # box constraints
                upper_bound=[Inf,  10.0],
                optim_opts=(show_trace=true,), # additional options for curve_fit() from LsqFit.jl in a NamedTuple. (For Optim.jl, this should be an Optim.options() object)
                theta_factors::Union{Vector{Float64}, Nothing} = nothing, # options are nothing or a vector of length P with factors for each parameter. Parameter theta[i] will be replaced by theta[i] * theta_factors[i] before optimization. Rule of thumb: if theta[i] is on the order of 10^M, pick theta_factor[i] = 10^(-M).
                trace=1)                  # 0, 1, or 2

myfit = GMMTools.fit(MYDATA, mom_fn, theta0, mode=:twostep, opts=myopts)
```

## Writing and reading results
`GMMTools.fit(...)` saves the estimation results in two files: `fit.csv` contains a table with one row per set of initial conditions, and `fit.json` contains several estimation parameters and results. `GMMTools.write(myfit::GMMFit, opts::GMMOptions; subpath="fit")` is the lower level function to write `GMMFit` objects to files.

`GMMTools.vcov_simple(...)` saves a `vcov.json` file that includes, among other objects, the variance-covariance matrix `myfit.vcov.V`. `GMMTools.vcov_bboot(...)` also saves two files `vcov_boot_fits_df.csv` (all individual runs for all bootstrap runs) and `vcov_boot_weights.csv` (rows are bootstrap runs, columns are data observations). The lower level function is `GMMTools.write(myvcov::GMMvcov, opts::GMMOptions; subpath="vcov")`.

`GMMTools.read_fit(full_path; subpath="fit", show_trace=false)` reads estimation results and loads them into a `GMMFit` object. `GMMTools.read_vcov(full_path; subpath="vcov", show_trace=false)` reads vcov results and loads them into a `GMMvcov` object. Note that `GMMTools.read_fit()` attempts to also read the vcov from the same folder. Otherwise, read the vcov separately and attach it using 
```julia
myfit = GMMTools.read_fit(mypath1)
myfit.vcov = GMMTools.read_vcov(mypath2)
```

## Capturing errors
By default, any error during optimization stops the entire estimation (or inference) command.

Set the `GMMOptions` field `throw_errors=false` to capture these errors and write them to file, but not interrupt the rest of the estimation procedure.
- when using multiple initial conditions, all iterations that error are recorded in `myfit.fits_df` with `errored=true` and `obj_value=Inf`. If all iterations error, we have `myfit.errored=true` and several other fields are recorded as `missing`
- for bootstrap results, similar rules apply. Note that inference objects (SEs, vcov, etc.) are computed dropping the bootstrap runs that errored. `@warn` messages should be displayed to remind the user that this is happenening. It is the user's responsibility to ensure this behavior is ok for their use case.

## Misc convenience options

__Scaling paarmeters__ The `theta_factors` option in `GMMOptions()` requests that the parameters be scaled before feeding them to the optimizer. Parameter theta[i] will be replaced by theta[i] * theta_factors[i] before optimization. Rule of thumb: if `theta[i]` is on the order of `10^M`, pick `theta_factor[i] = 10^(-M)`.

Example. Suppose `theta = [alpha, beta]` and we expect `alpha` to be between 0 and 1 while `beta` to be on the order of 0 to 1000. In general (not always) it's a good idea to scale `beta` down to approximately 0 to 1, which will ensure that the optimizer "cares" similarly about `alpha` and `beta`. (This also helps to make sense of magnitude of the default optimizer tolerance for theta, typically called `x_tol`.) We achieve this simply by selecting `theta_factors=[1.0, 0.001]`. All other inputs and outputs should have the original magnitudes: initial conditions `theta0` and final estimates `myfit.theta_hat`.

# Package To-do list

## Dev to-do list
1. compute sensitivity measure (Andrews et al 2017)
1. classical minimum distance (CMD), CUE
1. more general estimation of the covariance of the moments, cluster, HAC, Newey-West, etc.
1. other optimization backends
1. tests
1. (using user-provided function to generate data from model) Monte Carlo simulation to compute size and power.
1. (using user-provided function to generate data from model) Monte Carlo simulation of estimation finite sample properties (simulate data for random parameter values ⇒ run GMM ⇒ compare estimated parameters with underlying true parameters)

## Documentation to-do list
1. docs
1. Bootstrap options, including custom weihts function and an example
1. walkthrough for re-running failed estimation, stability of the random initial condition and bootstrap weights
1. (easy) example with subset parameters
1. Optimization discussion: finite vs AD, discontinuities in the objective function (e.g. due to iterated value function), (anecdotal) advantages of using the Levenberg-Marquardt algorithm over Optiml.jl
1. Non-linear estimation example
1. Example with AD including cache data and implicit function differentiation

For related projects, see 
- [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl) and [DrWatsonSim.jl](https://github.com/sebastianpech/DrWatsonSim.jl) 
- [GMMInference.jl](https://github.com/schrimpf/GMMInference.jl) 
- [SMM.jl](https://github.com/floswald/SMM.jl)


<!---
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


### Wish-list
1. integrate optimization backends other than `curve_fit` from `LsqFit.jl`, e.g. `Optim.jl`, [`GalacticOptim.jl`](`https://github.com/SciML/GalacticOptim.jl`), etc.
1. more general estimation of the covariance of the moments, e.g. Newey-West, etc.
1. compute sensitivity measure (Andrews et al 2017)
1. (using user-provided function to generate data from model) Monte Carlo simulation to compute size and power.
1. (using user-provided function to generate data from model) Monte Carlo simulation of estimation finite sample properties (simulate data for random parameter values ⇒ run GMM ⇒ compare estimated parameters with underlying true parameters)

# Install
To install this package:
`] add https://github.com/Gkreindler/GMMTools.jl`

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
-->

# Acknowledgements
Contributor: Peter Deffebach.
For useful suggestions and conversations: Michael Creel, Jeff Gortmaker.