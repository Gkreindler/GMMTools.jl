# GMM with GMMTools.jl

This tutorial shows how to use GMMTools.jl to estimate models using the generalized method of moments. If you have questions, open an issue or get in touch (gkreindler@g.harvard.edu).

## Getting started
Install [Julia](https://julialang.org/downloads/) and pick an IDE (I use [VSCode](https://code.visualstudio.com/)). Julia has an interactive mode (called REPL) and you can run entire scripts. To use a package [environment](https://jkrumbiegel.com/pages/2022-08-26-pkg-introduction/), start your Julia script with
```julia
using Pkg
Pkg.activate(".")
```
to use an environment in the current folder. The environment will be stored in two files (`Project.toml` and `Manifest.toml`). (You can also use a specific path `Pkg.activate("C:/my/favorite/path/")`). You can access the package manager from REPL by typing `]`. See [this](https://jkrumbiegel.com/pages/2022-08-26-pkg-introduction/) or other tutorials on environments and packages.

You can install packages at the REPL by typing `] add MyPackage`. For exmple, `add CSV` adds the package [`CSV.jl`](https://github.com/JuliaData/CSV.jl). Common useful packages in economics are [`DataFrames.jl`](https://github.com/JuliaData/DataFrames.jl), [`FixedEffectModels.jl`](https://github.com/FixedEffects/FixedEffectModels.jl), [`RegressionTables.jl`](https://github.com/jmboehm/RegressionTables.jl), and [`Optim.jl`](https://julianlsolvers.github.io/Optim.jl/stable/). Install all of them with `] add CSV, DataFrames, FixedEffectModels, RegressionTables, Optim`.

To install `GMMTools.jl`, you need to get it directly from github like this
```
] add https://github.com/Gkreindler/GMMTools.jl#redesign-methods
```

## Example 1: Estimate OLS using GMM

### Theory
We start with a simple example: estimate a bivariate OLS regression. We will estimate $\alpha$ and $\beta$ from a linear regression
```math
y_i = \alpha + \beta x_i + \epsilon_i
```
Given residuals $\widehat\epsilon_i = y_i - \widehat\alpha + \widehat\beta x_i$, the two moment conditions are 
```math
\mathbb E \widehat\epsilon_i = 0
```
and 
```math
\mathbb E \widehat\epsilon_i x_i = 0.
```

### Setup
To run this example, copy to your folder of choice the file [`auto-mpg.csv`](https://github.com/Gkreindler/GMMTools.jl/blob/redesign-methods/examples/auto-mpg.csv) and the script [`example_ols.jl`](https://github.com/Gkreindler/GMMTools.jl/blob/redesign-methods/examples/example_ols.jl).

### Understanding the example script
The example script starts by initializing the environment and using the required packages 
```julia
using Pkg
Pkg.activate(".")

using LinearAlgebra # for identity matrix "I"
using CSV
using DataFrames
using FixedEffectModels # for benchmarking
using RegressionTables

using GMMTools
```
The first time you run the script, you will need to install these packages to your new environment. See above.

Next, we load the sample data and add a constant column
```julia
# load data, originally from: https://www.kaggle.com/datasets/uciml/autompg-dataset/?select=auto-mpg.csv 
    df = CSV.read("examples/auto-mpg.csv", DataFrame)
    df[!, :constant] .= 1.0
```

### Setting up a model for estimation
In our example, $y_i$ measures `mpg` and $x_i$ measures `acceleration`. We will estimate two parameters $\theta = (\theta_1, \theta_1) = (\alpha, \beta)$.

We start by defining a `GMMProblem`, a type of object specific to this package. It holds the dataframe with data `df`, a weighting matrix (here we use the identity matrix `I`, which requires the package `LinearAlgebra`), and a vector of initial conditions for $(\alpha, \beta)$.
```julia
# initial parameter guess
    theta0 = [0.0, 0.0]

# create GMM problem (data + weighting matrix + initial parameter guess)
    myprob = create_GMMProblem(data=df, W=I, theta0=theta0)
```

### Moment function
The moment function is a key part of our model. For the package `GMMTools.jl`, the moment function must take as argument a `GMMProblem` object called `prob` and a vector of parameters `theta`. It must return a matrix with rows corresponding to data observations and columns corresponding to moments. We will get an error is the moment function returns a vector instead of a matrix. As mentioned above, the function accesses the dataframe with data using `prob.data`. It then further selects dataframe columns using `prob.data.mpg` and `prob.data.acceleration`. (Passing the entire object `prob` allows more advanced functionalities, for example, storing pre-allocated matrices in `prob.cache` for doing model computation.)
```julia
function ols_moments(prob::GMMProblem, theta)
    
    # residuals
    resids = @. prob.data.mpg - theta[1] - theta[2] * prob.data.acceleration
    
    # n observations x 2 moments
    moms = hcat(resids, resids .* prob.data.acceleration)
    
    return moms
end
```

### Estimation and Asymptotic Variance-Covariance
We are ready to estimate this model:
```julia
    myfit = fit(myprob, ols_moments)
```
And compute the asymptotic variance covariance matrix:
```julia
    vcov_simple(myprob, ols_moments, myfit)
```
This command modifies the `myfit` object by adding the `myfit.vcov` object.

We can display results using a modified version of `regtable` from the great `RegressionTables.jl` package:
```julia
    GMMTools.regtable(myfit)
```
which yields the following in REPL:
```
--------------------

            --------
                 (1)
--------------------
θ_1           4.972*
             (1.974)
θ_2         1.191***
             (0.128)
--------------------
Estimator        GMM
--------------------
N                398
R2
--------------------
```

This is quite similar to the result from using the `reg` function from `FixedEffectsModel.jl` (there are no fixed effects here)
```julia
    r = reg(df, term(:mpg) ~ term(:acceleration))
    RegressionTables.regtable(r)
```
namely
```
-----------------------
                  mpg  
               --------
                    (1)
-----------------------
(Intercept)      4.970*
                (2.043)
acceleration   1.191***
                (0.129)
-----------------------
Estimator           OLS
-----------------------
N                   398
R2                0.177
-----------------------
```

### Bayesian Bootstrap
The example script also shows how to compute [Bayesian (weighted) bootstrap](https://matteocourthoud.github.io/post/bayes_boot/) inference. More detailed notes to be added.

### More details
For most up-to-date details, check the source files in `src/functions_estimation.jl`, etc.

The `GMMProblem` object has the following fields
- `data`: can be accessed with `myprob.data` (any type, by default a DataFrame)
- `theta0`: vector with starting guesses of `theta`, or a $K\times P$ matrix where $K$ is the number of initial conditions to try, and $P$ the number of parameters (vector)
- (TO BE ADDED) `theta_lower`: Vector of lower bounds (default is -Inf)
- (TO BE ADDED) `theta_upper`: vector of upper bounds (default is +Inf)
- `theta_names`: parameter names (vector of strings, optional)
- `W`: weighting matrix for GMM
- `weights`: observation weights (vector, optional)


The function `fit` takes the following argments
- a `GMMProblem`
- a moment function `mom_fn`, here `ols_moments` defined above
- (optional) a Boolean `run_parallel` which is `True` by default. (See more on parallel computing below.) 
- (optional) a parameter `opts=default_gmm_opts()` which contains the following estimation options:
    - `path = ""` - a file path for writing result to file. This will not happen if the path is the empty string (default)
    - `write_iter=false` - whether to write results from each separate run (when running with multiple initial conditions)
    - `clean_iter=false` - whether to delete iteration files
    - `overwrite=false`  - whether to overwrite existing results. If this is false, `fit` will aim to read results from file. (Requires `opts.path != ""`)
    - `trace=0`          - whether to print details during estimation, `=0,1,2`
    - `optim_opts=default_optim_opts()` - options for the optimizer function in `Optim.jl`

The object `myfit` is of the type `GMMResult` with fields
- `theta0` - initial conditions
- `theta_hat` - estimated parameters
- `theta_names`
- `obj_value` - GMM objective value at `theta_hat`
- `converged` - dummy
- `iterations` - number of iterations
- `iteration_limit_reached` - dummy
- `time_it_took` - in seconds
- `all_results` - DataFrame with results from all initial conditions
- `N` - number of observations
- `vcov` - variance covariance object (a Dictionary) or nothing



