using Pkg
Pkg.activate(".")

using Revise
using GMMTools
using LinearAlgebra
using CSV
using DataFrames
using FixedEffectModels
using StatsBase

using RegressionTables

# for inference
using FiniteDiff
using ForwardDiff

# load data
# https://www.kaggle.com/datasets/uciml/autompg-dataset/?select=auto-mpg.csv
df = CSV.read("examples/auto-mpg.csv", DataFrame)
df[!, :constant] .= 1.0

# regression of interest
r = reg(df, term(:mpg) ~ term(:acceleration))

# define moments for OLS regression
function ols_moments(prob, theta)
    
    resids = @. prob.data.mpg - theta[1] - theta[2] * prob.data.acceleration
    
    # n by 2 matrix of moments
    moms = hcat(resids, resids .* prob.data.acceleration)
    
    return moms
end

    theta0 = [0.0, 0.0]

    myprob = create_GMMProblem(data=df, W=I, theta0=theta0)
                
    myfit = GMMTools.fit(myprob, ols_moments)

    # Asymptotic SEs
    vcov_simple(myprob, ols_moments, myfit)

    # print results
    GMMTools.regtable(myfit)

    # myboots = bboot(mymodel, setup_fn=setup_RD, nboot=100, cluster_var=:city_id, opts=myopts);
