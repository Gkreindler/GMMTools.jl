module GMMTools

# Write your package code here.
using Distributed
using Future: randjump

using DataFrames
using LinearAlgebra
using Statistics # means
using StatsBase # need to take bootstrap samples
using StatsAPI

using CSV 
using JSON

using Random

using Optim

using Vcov # needed for regression table
using RegressionTables
# import ..RegressionTables: regtable, asciiOutput


export GMMProblem, create_GMMProblem, GMMResult, table, random_theta0, fit, regtable

include("functions_gmm.jl")
include("gmm_table.jl")

end


