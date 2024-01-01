
using GMMTools
using Test

@testset "GMMTools.jl" begin
    # Write your tests here.
    @test include("test_example_ols.jl")
    @test include("test_example_ols_2step.jl")
end
