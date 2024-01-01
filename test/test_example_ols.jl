
using LinearAlgebra # for identity matrix "I"
using CSV
using Random

using DataFrames
using FixedEffectModels # for benchmarking
using RegressionTables


using GMMTools
display(GMMTools.fit)


# load data, originally from: https://www.kaggle.com/datasets/uciml/autompg-dataset/?select=auto-mpg.csv 
    df = CSV.read("auto-mpg.csv", DataFrame)
    df[!, :constant] .= 1.0


# Run plain OLS for comparison
    r = reg(df, term(:mpg) ~ term(:acceleration))
    RegressionTables.regtable(r)

    # store results here but this is not tracked by git
    isdir("results") || mkdir("results")

# define moments for OLS regression
# residuals orthogonal to the constant and to the variable (acceleration)
# this must always return a Matrix (rows = observations, columns = moments)
function ols_moments_fn(data, theta)
    resids = @. data.mpg - theta[1] - theta[2] * data.acceleration
    return hcat(resids, resids .* data.acceleration)
end

# initial parameter guess
    Random.seed!(123)
    theta0 = GMMTools.random_initial_conditions([10.0, 0.0], 20)


### Most parsimonius usage
    myfit = GMMTools.fit(df, ols_moments_fn, theta0)
    GMMTools.vcov_simple(df, ols_moments_fn, myfit)
    GMMTools.regtable(myfit)

### See `examples/example_ols_2step.jl` for more options and functionalities

    true