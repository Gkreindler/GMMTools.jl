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


# load data, originally from: https://www.kaggle.com/datasets/uciml/autompg-dataset/?select=auto-mpg.csv 
    df = CSV.read("examples/auto-mpg.csv", DataFrame)
    df[!, :constant] .= 1.0


# Run plain OLS for comparison
    r = reg(df, term(:mpg) ~ term(:acceleration))
    RegressionTables.regtable(r)

# define moments for OLS regression
# residuals orthogonal to the constant and to the variable (acceleration)
# this must always return a Matrix (rows = observations, columns = moments)
function ols_moments_fn(data, theta)
    
    resids = @. data.mpg - theta[1] - theta[2] * data.acceleration
    
    # n by 2 matrix of moments
    moms = hcat(resids, resids .* data.acceleration)
    
    return moms
end

# initial parameter guess
    # theta0 = [0.0, 0.0]
    theta0 = randn(20,2)

# create GMM problem (data + weighting matrix + initial parameter guess)
    # myprob = create_GMMProblem(
    #             data=df, 
    #             W=I,    
    #             theta0=theta0)
                
# estimation options
myopts = GMMTools.GMMOptions(
                path="C:/git-repos/GMMTools.jl/examples/temp/", 
                optim_algo=NewtonTrustRegion(), 
                optim_autodiff=:forward,
                write_iter=true,
                clean_iter=true,
                overwrite=true,
                trace=1)

# estimate model
myfit = fit(df, ols_moments_fn, theta0, mode=:twostep, opts=myopts)

# compute asymptotic variance-covariance matrix and save in myfit.vcov
    vcov_simple(df, ols_moments_fn, myfit)

# print table with results
    GMMTools.regtable(myfit)








# compute Bayesian (weighted) bootstrap inference and save in myfit.vcov
    vcov_bboot(df, ols_moments_fn, theta0, myfit, nboot=500)
    GMMTools.regtable(myfit) # print table with new bootstrap SEs


fsdfds

# bootstrap with weightes drawn at the level of clusters defined by the variable df.cylinders
    vcov_bboot(myprob, ols_moments, myfit, cluster_var=:cylinders, nboot=500)
    GMMTools.regtable(myfit)

### Multiple initial conditions and save to file
    theta0 = [0.3, 0.5]

    theta0_matrix = random_theta0(theta0, 100) # generate 100 random initial conditions "around" theta0

    myprob = create_GMMProblem(data=df, W=I, theta0=theta0_matrix)

    myopts = default_gmm_opts(
        path = "C:/git-repos/GMMTools.jl/examples/temp/", # replace with target folder on your machine
        write_iter=true, # save results to file option
        clean_iter=true, # save results to file option
        overwrite=false,  # save results to file option
        trace=1
        )

    myfit = fit(myprob, ols_moments, opts=myopts)
    vcov_simple(myprob, ols_moments, myfit)
    GMMTools.regtable(myfit)