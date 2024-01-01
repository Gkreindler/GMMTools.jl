using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using Revise
using LinearAlgebra # for identity matrix "I"
using CSV
using DataFrames
using FixedEffectModels # for benchmarking
using RegressionTables

using GMMTools
using Optim # need for NewtonTrustRegion()

using Random

# load data, originally from: https://www.kaggle.com/datasets/uciml/autompg-dataset/?select=auto-mpg.csv 
    df = CSV.read("examples/auto-mpg.csv", DataFrame)
    df[!, :constant] .= 1.0


# Run plain OLS for comparison
    reg_ols = reg(df, term(:mpg) ~ term(:acceleration))
    regtable(reg_ols)

# define moments for OLS regression
# residuals orthogonal to the constant and to the variable (acceleration)
# this must always return a Matrix (rows = observations, columns = moments)
function ols_moments_fn(data, theta)
    resids = @. data.mpg - theta[1] - theta[2] * data.acceleration
    return hcat(resids, resids .* data.acceleration)
end

# initial parameter guess
    Random.seed!(123)
    theta0 = random_initial_conditions([10.0, 0.0], 20)

    ### important to use paramter names, so we can keep track of which parameters are fixed
    theta_names = ["constant", "acceleration"] 
    
    ### Alternative 
    theta_names = default_theta_names(size(theta0, 2))

### Plain
    myopts = GMMOptions(theta_names=theta_names)
    myfit1 = GMMTools.fit(df, ols_moments_fn, theta0, opts=myopts)
    GMMTools.vcov_simple( df, ols_moments_fn, myfit1, opts=myopts)
    GMMTools.regtable(myfit1)

### Fix constant in optimization
    theta_fixed = [3.970, missing]

    # load fixed values into moment function
    ols_moments_fn_fixed = (data, theta_small) -> ols_moments_fn(data, theta_add_fixed_values(theta_small, theta_fixed))

    # subset initial conditions & parameter names
    theta0_fixed = theta0[:, ismissing.(theta_fixed)]
    theta_names_fixed = theta_names[ismissing.(theta_fixed)]

    # estimate model
    myopts = GMMOptions(theta_names=theta_names_fixed)
    myfit2 = GMMTools.fit(df, ols_moments_fn_fixed, theta0_fixed, mode=:twostep, opts=myopts)

    ### Simple inference
    vcov_simple(df, ols_moments_fn_fixed, myfit2, opts=myopts)
    println("Asymptotic iid errors:")
    regtable(myfit1, myfit2) |> display

    ### Bootstrap
    bweights_matrix = boot_weights(df, myfit2, nboot=100, method=:simple, rng_initial_seed=1234) 
    vcov_bboot(df, ols_moments_fn_fixed, theta0_fixed, myfit2, bweights_matrix, opts=myopts)

    println("Bootstrap standard errors:")
    regtable(myfit1, myfit2) |> display