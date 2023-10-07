#= 
TODOS:
1. add try ... catch blocks
1. add optimizer options / variants. Where to save the optimizer-specific options? kwargs.
1. save bootstrap weights?
1. add parameter names in the problem object
1. switch to PrettyTables.jl (cf Peter's suggestion)
=#

Base.@kwdef mutable struct GMMBootResults
    all_results
    all_results_allbootruns
end

function Base.show(io::IO, r::GMMBootResults)
    println("Baysian bootstrap results")
    display(r.all_results)
    display(r.all_results_allbootruns)
end

function write(boot_results::GMMBootResults, opts::GMMOptions)
    if opts.path == ""
        return
    end

    # temp_df = DataFrame(boot_results.table, :auto)

    CSV.write(opts.path * "results_boot.csv", boot_results.all_results)
    CSV.write(opts.path * "results_boot_all.csv", boot_results.all_results_allbootruns)
end


###

function jacobian(problem::GMMProblem, mom_fn::Function, myfit::GMMResult)
    
    moment_loaded = theta -> mean(mom_fn(problem, theta), dims=1)

    try
        # try automatic differentiation
        return ForwardDiff.jacobian(moment_loaded, myfit.theta_hat)
    catch
        # fall back on finite differences
        println("jacobian: AD failed, falling back on finite differences")
        return FiniteDiff.finite_difference_jacobian(moment_loaded, myfit.theta_hat)
    end

end

function vcov_simple(problem::GMMProblem, mom_fn::Function, myfit::GMMResult)
    
    # jacobian
    J = jacobian(problem, mom_fn, myfit)
    # display(J)

    # weight matrix
    W = problem.W
    # display(W)

    # simple, general: (JWJ')⁻¹ * JWΣW'J * (JWJ')⁻¹

    # estimate Σ
    m = mom_fn(problem, myfit.theta_hat)
    N = size(m, 1)
    Σ = Transpose(m) * m / N
    # display(Σ)

    invJWJ = inv(J * W * J')
    
    V = invJWJ * J * W * Σ * W' * J * invJWJ ./ N

    myfit.vcov = Dict(
        :method => Vcov.simple(), #"plain",
        :V => V,
        :J => J,
        :W => W,
        :Σ => Σ,
        :N => N, # TODO : should really move this up to main fit object
        :ses => sqrt.(diag(V))
    )

    return
end

function process_boot_results(lb::Vector{GMMResult})

    boot_results_allruns = []
    nboot = length(lb)
    for i=1:nboot
        temp_df = copy(lb[i].all_results)
        temp_df[!, :boot_idx] .= i
        push!(boot_results_allruns, temp_df)
    end
    boot_results_allruns = vcat(boot_results_allruns...)
    
    # vcat([b.all_results for b=lb]...)

    mysample = boot_results_allruns.is_optimum .== 1
    boot_results = copy(boot_results_allruns[mysample, :])

    # nparams = length(lb[1].theta_hat)
    # nboot = length(lb)

    # res_table = Matrix{Float64}(undef, nboot, nparams)
    # for i=1:nboot
    #     res_table[i, :] = lb[i].theta_hat
    #     lb[i].all_results[!, :boot_idx] .= i
    # end

    return GMMBootResults(
        all_results = boot_results,
        all_results_allbootruns = boot_results_allruns
    )
end




"""
Bayesian bootstrap
"""
function bboot(
    problem::GMMProblem,
    mom_fn::Function; 
    boot_fn::Union{Function, Nothing}=nothing, 
    nboot=100, 
    cluster_var=nothing, 
    run_parallel=true,
    opts::GMMOptions=GMMOptions())

    if opts.path != ""
        bootpath = opts.path * "__boot__"
        isdir(bootpath) || mkdir(bootpath)
    end

    if !isnothing(cluster_var)
        cluster_values = unique(problem.data[:, cluster_var])

        ### ___idx_cluster___ has the index in the cluster_values vector of this row's value        
            # drop column if already in the df
            ("___idx_cluster___" in names(problem.data)) && select!(problem.data, Not("___idx_cluster___"))

            # join
            temp_df = DataFrame(string(cluster_var) => cluster_values, "___idx_cluster___" => 1:length(cluster_values))
            leftjoin!(problem.data, temp_df, on=cluster_var)
    else
        cluster_values=nothing
    end

    #         
    if !run_parallel
        list_of_boot_results = Vector{GMMResult}(undef, nboot)
        for i=1:nboot
            list_of_boot_results[i] = bboot(
                i, 
                problem, 
                mom_fn,
                opts, 
                boot_fn=boot_fn, 
                cluster_var=cluster_var)
        end
    else
        list_of_boot_results = pmap( 
            i -> bboot(i, 
                    problem, 
                    mom_fn,
                    opts, 
                    boot_fn=boot_fn,
                    cluster_var=cluster_var, 
                    cluster_values=cluster_values), 
            1:nboot)
    end

    boot_results = process_boot_results(list_of_boot_results)

    write(boot_results, opts)

    # delete all intermediate files with individual iteration results
    if opts.clean_iter 
        print("Deleting individual boot files...")
        rm(opts.path * "__boot__/", force=true, recursive=true)
        println(" Done.")
    end

    return boot_results
end

function bboot(
    idx::Int64, 
    problem::GMMProblem, 
    mom_fn::Function,
    opts::GMMOptions; 
    boot_fn::Union{Function, Nothing}=nothing,
    cluster_var=nothing, 
    cluster_values)

    println("bootstrap run ", idx)

    # boot_fn
    
    if isnothing(cluster_var)
        problem.weights = rand(Dirichlet(nrow(problem.data), 1.0))
    else
        cluster_level_weights = rand(Dirichlet(length(cluster_values), 1.0))

        # one step "join" to get the weight for the appropriate cluster
        problem.weights .= cluster_level_weights[problem.data.___idx_cluster___]
    end

    # normalizing weights is important for numerical precision
    problem.weights = problem.weights ./ mean(problem.weights)

    # path for saving results
    new_opts = copy(opts)
    new_opts.path *= "__boot__/boot_" * string(idx) * "_"

    return fit(problem, mom_fn, run_parallel=false, opts=new_opts)
end






# ! move to table?
# Returns a matrix with all bootstrap estimates (for computing CIs of other stats)
function theta_hat_boot(rb::Union{GMMBootResults, GMMResult})
    x = parse_vector.(rb.all_results.theta_hat)
    x = hcat(x...) |> Transpose |> Matrix

    return x
end

# Returns confidence intervals (95%)
function cis(rb::GMMBootResults; ci_levels=[2.5, 97.5])

    nparams = length(rb.all_results[1, :theta_hat])

    theta_hat_boot = theta_hat_boot(rb)

    cis = []
    for i=1:nparams
        cil, cih = percentile(theta_hat_boot[:, i], ci_levels)
        push!(cis, (cil, cih))
    end
    
    return cis
end

# Returns standard errors (SD of bootstrap estimates)
function stderr(rb::GMMBootResults)

    nparams = length(rb.all_results[1, :theta_hat])

    theta_hat_boot = theta_hat_boot(rb)

    stderrors = zeros(nparams)
    for i=1:nparams
        stderrors[i] = std(theta_hat_boot[:, i])
    end
    
    return stderrors
end

