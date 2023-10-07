module GMMTools

# Write your package code here.
using Distributed
using Future: randjump

using DataFrames
using LinearAlgebra
using Statistics # means

using StatsAPI

# for (Bayesian) bootstrap
using StatsBase # take samples
using Distributions # Dirichlet distribution

using CSV 
using JSON

using Random

using Optim

using FiniteDiff
using ForwardDiff

using Vcov # needed for regression table
using RegressionTables
# import ..RegressionTables: regtable, asciiOutput


export GMMProblem, create_GMMProblem, GMMResult, table, random_theta0, 
       fit, 
       vcov_simple, vcov_bboot,
       regtable

include("functions_gmm.jl")
include("functions_inference.jl")
include("gmm_table.jl")

end


