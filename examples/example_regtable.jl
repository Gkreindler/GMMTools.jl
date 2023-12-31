using Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using Revise
using CSV
using DataFrames
using FixedEffectModels # for benchmarking
using RegressionTables

using GMMTools

# load data, originally from: https://www.kaggle.com/datasets/uciml/autompg-dataset/?select=auto-mpg.csv 
    df = CSV.read("examples/auto-mpg.csv", DataFrame)
    df[!, :constant] .= 1.0


# Run plain OLS for comparison
    reg_ols = reg(df, term(:mpg) ~ term(:acceleration))
    regtable(reg_ols)

    # read fit from file
    mypath = "C:/git-repos/GMMTools.jl/examples/temp/step2/"
    myfit = GMMTools.read_fit(mypath)

    # read vcov from file
    mypath = "C:/git-repos/GMMTools.jl/examples/temp/"
    myfit.vcov = GMMTools.read_vcov(mypath)

    myfit.theta_names = ["(Intercept)", "acceleration"]

# print table with results (combine both OLS and GMM results)
    # want to use a new label for the estimator: doesn't work
    # RegressionTables.label_distribution(render::AbstractRenderType, d::GMMTools.GMMFit) = "GMM"
    # default: print_estimator_section = false

    # mylabels = Dict("theta_1" => "(Intercept)", "theta_2" => "acceleration")
    regtable(reg_ols, myfit, render = AsciiTable(), below_statistic = ConfInt) |> display
    
    
