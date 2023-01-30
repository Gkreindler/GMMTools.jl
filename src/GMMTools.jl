module GMMTools

# Write your package code here.
using Future

using DataFrames
using LinearAlgebra
using Statistics # means
using StatsBase # need to take bootstrap samples

using CSV 
using JSON

using Random

using FiniteDifferences
using LsqFit # install version that accepts MaxTime. Run "add LsqFit#master" as of Jan 2023


export run_estimation, run_inference, random_initial_conditions

include("gmm_tools.jl")

end
