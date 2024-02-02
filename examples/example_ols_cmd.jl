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

    # outcome if exp() of linear combination of parameters
    df.outcome = @. exp( 0.05 * df.mpg + 0.1 * df.acceleration + df.constant)

    reg_ols = reg(df, term(:outcome) ~ term(:mpg) + term(:acceleration))
    regtable(reg_ols)

    data_moms = [reg_ols.coef[1] reg_ols.coef[2] reg_ols.coef[3]]

# define moments for OLS regression
# residuals orthogonal to the constant and to the variable (acceleration)
# this must always return a Matrix (rows = observations, columns = moments)
function cmd_moments_fn(data, theta)

    data.outcome = @. exp( theta[1] * data.mpg + theta[2] * data.acceleration + theta[3] * data.constant)

    myreg = reg(data, term(:outcome) ~ term(:mpg) + term(:acceleration))

    return [myreg.coef[1] myreg.coef[2] myreg.coef[3]] .- data_moms
end

    cmd_moments_fn(df, [0.05, 0.1, 1.0])

# initial parameter guess
    Random.seed!(123)
    theta0 = random_initial_conditions([10.0, 0.0, 0.0], 1)

### using Optim.jl
    # estimation options
    myopts = GMMTools.GMMOptions(
                    path="C:/git-repos/GMMTools.jl/examples/temp/", 
                    optimizer=:lsqfit,
                    theta_lower=[-Inf, -Inf, -Inf],
                    theta_upper=[Inf, Inf, Inf],
                    # optim_autodiff=:forward,
                    write_iter=false,
                    clean_iter=true,
                    overwrite=true,
                    throw_errors=false,
                    # optim_opts=(show_trace=false,), # additional options for LsqFit in a NamedTuple
                    trace=0)

    # estimate model
    myfit = GMMTools.fit(df, cmd_moments_fn, theta0, mode=:cmd, opts=myopts)


# compute asymptotic variance-covariance matrix and save in myfit.vcov
    vcov_simple(df, ols_moments_fn, myfit, opts=myopts)

    GMMTools.write(myfit.vcov, myopts.path)

    # read vcov from file
    mypath = "C:/git-repos/GMMTools.jl/examples/temp/"
    myvcov2 = GMMTools.read_vcov(mypath)
    myfit.vcov = myvcov2

# print table with results
    regtable(myfit) |> display

    regtable(reg_ols, myfit) |> display # works
    # regtable(reg_ols, myfit; render   = AsciiTable()) # doesn't work. why?

# compute Bayesian (weighted) bootstrap inference and save in myfit.vcov (this will overwrite existing vcov files)
    myopts.trace = 1
    myopts.write_iter = false # time consuming to write individual iterations to file

    # first generate bootstrap weights. In case estimation is interrupted, running this code again generates exactly the same weights, so we can continue where we left off.
    bweights_matrix = boot_weights(df, myfit, nboot=100, method=:simple, rng_initial_seed=1234) 

    vcov_bboot(df, ols_moments_fn, theta0, myfit, bweights_matrix, opts=myopts)
    regtable(myfit) # print table with new bootstrap SEs -- very similar to asymptotic SEs in this case. Nice!


# read vcov with bootstrop from file
    myfit.vcov = GMMTools.read_vcov(myopts.path);
    regtable(myfit) |> display

### Factors in optimization
    myopts = GMMTools.GMMOptions(
                    path="C:/git-repos/GMMTools.jl/examples/temp/", 
                    optimizer=:lsqfit,
                    theta_lower=[-Inf, -Inf],
                    theta_upper=[Inf, Inf],
                    optim_autodiff=:forward,
                    # optim_opts=(show_trace=false,), # additional options for LsqFit in a NamedTuple
                    write_iter=false,
                    clean_iter=true,
                    overwrite=true,
                    throw_errors=false,
                    theta_factors=[0.25, 4.0], # ! this is new. Ask theta_1 to be divided by 4 before feeding it to the optimizer, and ask theta_2 to be multiplied by 4
                    trace=1)

    # estimate model
    myfit = GMMTools.fit(df, ols_moments_fn, theta0, mode=:twostep, opts=myopts)
    vcov_simple(df, ols_moments_fn, myfit, opts=myopts)

    myopts.trace = 1
    vcov_bboot(df, ols_moments_fn, theta0, myfit, bweights_matrix, opts=myopts)

    regtable(myfit) |> display # ! same results as before