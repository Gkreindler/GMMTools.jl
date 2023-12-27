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
    
    # n by 2 matrix of moments
    moms = hcat(resids, resids .* data.acceleration)
    
    return moms
end

# initial parameter guess
    Random.seed!(123)
    theta0 = random_initial_conditions([0.0, 0.0], 20)

### using Optim.jl
    # estimation options
    myopts = GMMTools.GMMOptions(
                    path="C:/git-repos/GMMTools.jl/examples/temp/", 
                    optimizer=:lsqfit,
                    theta_lower=[-Inf, -Inf],
                    theta_upper=[Inf, Inf],
                    optim_autodiff=:forward,
                    # write_moms=false,
                    write_iter=false,
                    clean_iter=true,
                    overwrite=false,
                    # optim_opts=(show_trace=false,), # additional options for LsqFit in a NamedTuple
                    trace=0)

    # estimate model
    myfit = GMMTools.fit(df, ols_moments_fn, theta0, mode=:twostep, opts=myopts)

    myopts.path *= "step1/"
    myfit2 = GMMTools.read_fit(myopts)
   
### using Optim.jl
    # estimation options
    myopts = GMMTools.GMMOptions(
                    path="C:/git-repos/GMMTools.jl/examples/temp/", 
                    optimizer=:optim,
                    optim_algo=LBFGS(), 
                    optim_autodiff=:forward,
                    write_iter=true,
                    clean_iter=true,
                    overwrite=true,
                    trace=0)

    # estimate model
    myfit = GMMTools.fit(df, ols_moments_fn, theta0, mode=:twostep, opts=myopts);

# compute asymptotic variance-covariance matrix and save in myfit.vcov
    vcov_simple(df, ols_moments_fn, myfit)


    GMMTools.write(myfit.vcov, myopts)

# print table with results
    regtable(myfit) |> display

    regtable(reg_ols, myfit) |> display # works
    # regtable(reg_ols, myfit; render   = AsciiTable()) # doesn't work. why?

# compute Bayesian (weighted) bootstrap inference and save in myfit.vcov (this will overwrite existing vcov files)
    myopts.trace = 1
    myopts.write_iter = false # time consuming to write individual iterations to file (500 x 20)

    # first generate bootstrap weights. In case estimation is interrupted, running this code again generates exactly the same weights, so we can continue where we left off.
    bweights_matrix = boot_weights(df, myfit, nboot=100, method=:simple, rng_initial_seed=1234) 

    vcov_bboot(df, ols_moments_fn, theta0, myfit, bweights_matrix, opts=myopts)
    regtable(myfit) # print table with new bootstrap SEs -- very similar to asymptotic SEs in this case. Nice!


# read vcov with bootstrop from file
    myfit.vcov = GMMTools.read_vcov(myopts)
    regtable(myfit) |> display

    # using Plots
    # histogram(myfit.vcov[:boot_fits].all_theta_hat[:, 1])

# bootstrap with weightes drawn at the level of clusters defined by the variable df.cylinders
    # myopts.trace = 0
    # vcov_bboot(df, ols_moments_fn, theta0, myfit, boot_weights=:cluster, cluster_var=:cylinders, nboot=500, opts=myopts)
    # myfit.vcov

    # GMMTools.regtable(myfit)

