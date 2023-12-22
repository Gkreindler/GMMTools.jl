module GMMTools

# Write your package code here.
using Distributed # for parallel
# using Future: randjump # will use for generating random seeds for parallel bootstrap runs

using DataFrames
using LinearAlgebra
using Statistics # means

using StatsAPI

# for (Bayesian) bootstrap
using StatsBase # take samples
using Distributions # Dirichlet distribution for bootstrap

using DelimitedFiles
using CSV 
using JSON

using Random

using Optim

using FiniteDiff
using ForwardDiff

using Vcov # needed for regression table
using RegressionTables
# import ..RegressionTables: regtable, asciiOutput


export GMMFit, table, 
        random_initial_conditions,theta_add_fixed_values,
        default_gmm_opts, default_optim_opts,
        fit, 
        vcov_simple, vcov_bboot,
        regtable

include("functions_estimation.jl")
include("functions_inference.jl")
include("functions_regtable.jl")
include("utilities.jl")

end


