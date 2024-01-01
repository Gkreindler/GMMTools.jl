module GMMTools

using Distributed

using DataFrames
using LinearAlgebra
using Statistics # means

using StatsAPI

# for (Bayesian) bootstrap
using StatsBase # take samples
using StatsModels
using Distributions # Dirichlet distribution for bootstrap

using DelimitedFiles
using CSV 
using JSON

using Random

# optimization backends
using Optim
using LsqFit

# optimization gradients
using FiniteDiff
using ForwardDiff

# ? should drop this?
using ProgressMeter

# for tables
using Vcov # needed for regression table
using RegressionTables


export GMMOptions,
        random_initial_conditions, theta_add_fixed_values,
        default_gmm_opts, default_optim_opts,
        fit, 
        vcov_simple, vcov_bboot, boot_weights,
        regtable

include("functions_estimation.jl")
include("optimization_backends.jl")
include("functions_inference.jl")
include("functions_regtable.jl")
include("utilities.jl")
include("io.jl")

end


