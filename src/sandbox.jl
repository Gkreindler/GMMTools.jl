using Revise
using StatsBase
using StatsAPI
using Vcov
using DataFrames
using RegressionTables



Base.@kwdef struct GMMResults <: RegressionModel
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix
    vcov_type::CovarianceEstimator
    nclusters::Union{NamedTuple, Nothing} = nothing

    esample::BitVector      # Is the row of the original dataframe part of the estimation sample?
    residuals::Union{AbstractVector, Nothing} = nothing
    fe::DataFrame
    fekeys::Vector{Symbol}


    coefnames::Vector       # Name of coefficients
    responsename::Union{String, Symbol} # Name of dependent variable
    # formula::FormulaTerm        # Original formula
    # formula_schema::FormulaTerm # Schema for predict
    contrasts::Dict

    nobs::Int64             # Number of observations
    dof::Int64              # Number parameters estimated - has_intercept. Used for p-value of F-stat.
    dof_fes::Int64          # Number of fixed effects
    dof_residual::Int64     # dof used for t-test and p-value of F-stat. nobs - degrees of freedoms with simple std
    rss::Float64            # Sum of squared residuals
    tss::Float64            # Total sum of squares

    F::Float64              # F statistics
    p::Float64              # p value for the F statistics

    # for FE
    iterations::Int         # Number of iterations
    converged::Bool         # Has the demeaning algorithm converged?
    r2_within::Union{Float64, Nothing} = nothing      # within r2 (with fixed effect

    # for IV
    F_kp::Union{Float64, Nothing} = nothing           # First Stage F statistics KP
    p_kp::Union{Float64, Nothing} = nothing           # First Stage p value KP
end


has_iv(m::GMMResults) = m.F_kp !== nothing
has_fe(m::GMMResults) = false

# function GMMResults()
#     return GMMResults()
# end


StatsAPI.coef(m::GMMResults) = m.coef
StatsAPI.coefnames(m::GMMResults) = m.coefnames
StatsAPI.responsename(m::GMMResults) = m.responsename
StatsAPI.vcov(m::GMMResults) = m.vcov
StatsAPI.nobs(m::GMMResults) = m.nobs
StatsAPI.dof(m::GMMResults) = m.dof
StatsAPI.dof_residual(m::GMMResults) = m.dof_residual
StatsAPI.r2(m::GMMResults) = r2(m, :devianceratio)
StatsAPI.islinear(m::GMMResults) = true
StatsAPI.deviance(m::GMMResults) = rss(m)
StatsAPI.nulldeviance(m::GMMResults) = m.tss
StatsAPI.rss(m::GMMResults) = m.rss
StatsAPI.mss(m::GMMResults) = nulldeviance(m) - rss(m)
# StatsModels.formula(m::GMMResults) = m.formula_schema
dof_fes(m::GMMResults) = m.dof_fes

myr = GMMResults(
    coef = [0.0, 1.0],
    vcov = [0.4 0.2; 0.2 0.1] .^ 2,
    vcov_type=Vcov.simple(),
    esample=[],
    fe=DataFrame(),
    fekeys=[],
    coefnames=["coef1", "coef2"],
    responsename="my outcome",
    # formula::FormulaTerm        # Original formula
    # formula_schema::FormulaTerm # Schema for predict
    contrasts=Dict(),
    nobs=5,
    dof=1,
    dof_fes=1,
    dof_residual=1,
    rss=0.00,
    tss=0.00,
    F=0.1,
    p=0.1,
    iterations=5,         # Number of iterations
    converged=true        # Has the demeaning algorithm converged?
)


regtable(myr; renderSettings = asciiOutput(), labels = Dict("__LABEL_ESTIMATOR_OLS__" => "GMM"))
