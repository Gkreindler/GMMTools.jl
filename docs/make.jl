push!(LOAD_PATH, "..")

using Documenter

makedocs(
    sitename = "GMMTools.jl",
    pages = Any[
        "Introduction" => "index.md",
        "Logit tutorial" => "tutorials/logit.md"],
    format = Documenter.HTML(
        canonical = "https://Gkreindler.github.io/GMMTools.jl/stable/"
    ))

deploydocs(
    repo = "https://github.com/Gkreindler/GMMTools.jl.git",
    target = "build",
    deps = nothing,
    make = nothing)
