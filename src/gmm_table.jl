
Base.@kwdef struct GMMResultTable <: RegressionModel
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


has_iv(m::GMMResultTable) = m.F_kp !== nothing
has_fe(m::GMMResultTable) = false

StatsAPI.coef(m::GMMResultTable) = m.coef
StatsAPI.coefnames(m::GMMResultTable) = m.coefnames
StatsAPI.responsename(m::GMMResultTable) = m.responsename
StatsAPI.vcov(m::GMMResultTable) = m.vcov
StatsAPI.nobs(m::GMMResultTable) = m.nobs
StatsAPI.dof(m::GMMResultTable) = m.dof
StatsAPI.dof_residual(m::GMMResultTable) = m.dof_residual
StatsAPI.r2(m::GMMResultTable) = r2(m, :devianceratio)
StatsAPI.islinear(m::GMMResultTable) = true
StatsAPI.deviance(m::GMMResultTable) = rss(m)
StatsAPI.nulldeviance(m::GMMResultTable) = m.tss
StatsAPI.rss(m::GMMResultTable) = m.rss
StatsAPI.mss(m::GMMResultTable) = nulldeviance(m) - rss(m)
# StatsModels.formula(m::GMMResultTable) = m.formula_schema
dof_fes(m::GMMResultTable) = m.dof_fes

function GMMResultTable(r::GMMResult)
    
    GMMResultTable(
    coef = r.theta_hat,
    vcov = diagm(r.theta_hat),
    vcov_type=Vcov.simple(),
    esample=[],
    fe=DataFrame(),
    fekeys=[],
    coefnames=r.theta_names,
    responsename="",
    # formula::FormulaTerm        # Original formula
    # formula_schema::FormulaTerm # Schema for predict
    contrasts=Dict(),
    nobs=1, # ! this only works for dataframes
    dof=1,
    dof_fes=1,
    dof_residual=1,
    rss=0.00,
    tss=0.00,
    F=0.0,
    p=0.0,
    iterations=5, 
    converged=true)
end

function regtable(r::GMMResult)
    RegressionTables.regtable(GMMResultTable(r);  
        labels = Dict("__LABEL_ESTIMATOR_OLS__" => "GMM"), 
        renderSettings = asciiOutput())
end


