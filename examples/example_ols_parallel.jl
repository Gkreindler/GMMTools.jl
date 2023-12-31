using Distributed

using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

rmprocs(workers())
display(workers())
addprocs(2)
display(workers())

@everywhere using Pkg
@everywhere Pkg.activate(".")

@everywhere begin

    using Revise
    using LinearAlgebra # for identity matrix "I"
    using CSV
    using DataFrames
    using FixedEffectModels # for benchmarking
    using RegressionTables

    using GMMTools
    using Optim # need for NewtonTrustRegion()

    using Random
end

# load data, originally from: https://www.kaggle.com/datasets/uciml/autompg-dataset/?select=auto-mpg.csv 
    df = CSV.read("examples/auto-mpg.csv", DataFrame)
    df[!, :constant] .= 1.0

# Run plain OLS for comparison
    r = reg(df, term(:mpg) ~ term(:acceleration))
    regtable(r)

# define moments for OLS regression
# residuals orthogonal to the constant and to the variable (acceleration)
# this must always return a Matrix (rows = observations, columns = moments)
@everywhere function ols_moments_fn(data, theta)
    resids = @. data.mpg - theta[1] - theta[2] * data.acceleration
    return hcat(resids, resids .* data.acceleration)
end

# initial parameter guess
    Random.seed!(123)
    theta0 = random_initial_conditions([10.0, 0.0], 20)

### using Optim.jl
    # estimation options
    myopts = GMMTools.GMMOptions(
                    path="C:/git-repos/GMMTools.jl/examples/temp/", 
                    optimizer=:lsqfit,
                    theta_lower=[-Inf, -Inf],
                    theta_upper=[Inf, Inf],
                    optim_autodiff=:forward,
                    write_iter=true,
                    clean_iter=true,
                    overwrite=true,
                    trace=1)

    # estimate model
    myfit = GMMTools.fit(df, ols_moments_fn, theta0, opts=myopts, run_parallel=true)
