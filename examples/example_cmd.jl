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
using Distributions

# load data, originally from: https://www.kaggle.com/datasets/uciml/autompg-dataset/?select=auto-mpg.csv 
    df = CSV.read("examples/auto-mpg.csv", DataFrame)
    df[!, :constant] .= 1.0

    # outcome if exp() of linear combination of parameters
    df.outcome = @. exp( 0.05 * df.mpg + 0.1 * df.acceleration + df.constant)

    reg_ols = reg(df, term(:outcome) ~ term(:mpg) + term(:acceleration))
    regtable(reg_ols)

    # Save the data reduced form moments (π in the Woolridge notation) and their variance-covariance matrix (Ξ in the Wooldridge notation)
    mymoms_data = Matrix(Transpose(reg_ols.coef))
    mymoms_data_vcov = reg_ols.vcov

# define moments for OLS regression
# residuals orthogonal to the constant and to the variable (acceleration)
# this must always return a Matrix (rows = observations, columns = moments)
function cmd_moments_fn(data, moms_data, theta)

    data.outcome = @. exp( theta[1] * data.mpg + theta[2] * data.acceleration + theta[3] * data.constant)

    myreg = reg(data, term(:outcome) ~ term(:mpg) + term(:acceleration))

    @assert size(moms_data, 1) == 1 # return a 1 x M row vector

    # return a 1 x M row vector
    return Matrix(Transpose(myreg.coef)) .- moms_data
end

    theta_true = [0.05, 0.1, 1.0]
    cmd_moments_fn(df, mymoms_data, theta_true)

# initial parameter guess
    Random.seed!(123)
    theta0 = random_initial_conditions([0.1, 0.1, 0.1], 20)

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
                    throw_errors=true,
                    # optim_opts=(show_trace=false,), # additional options for LsqFit in a NamedTuple
                    trace=2)

    # estimate model
    myfit = GMMTools.fit_cmd(df, cmd_moments_fn, mymoms_data, mymoms_data_vcov, theta0, opts=myopts)

    display(myfit.theta_hat)
    display(cmd_moments_fn(df, mymoms_data, myfit.theta_hat))



# compute asymptotic variance-covariance matrix and save in myfit.vcov
    # TODO: add asymptotic variance

    
# Propagate multivariate normal uncertainty in reduced form moments to uncertainty in parameters

    ### Generate values of the data moments drawns from the multivariate normal distribution around estimates
        n_moms = size(mymoms_data, 2)        
        mymvn = MvNormal(zeros(n_moms), mymoms_data_vcov)

        n_boot = 100
        mymoms_data_matrix = zeros(n_boot, n_moms)
        for i=1:n_boot
            mymoms_data_matrix[i:i, :] .= mymoms_data .+ rand(mymvn)'
        end
        

    # run 
    myopts.trace = 1
    myopts.write_iter = false # time consuming to write individual iterations to file

    vcov_cmd(df, cmd_moments_fn, mymoms_data_matrix, mymoms_data_vcov, theta0, myfit, opts=myopts)

    # ! Stars are off because of DOF adjustments with n_obs = 1
    myfit.n_obs = 100
    regtable(myfit) # print table with new bootstrap SEs -- very similar to asymptotic SEs in this case. Nice!


    